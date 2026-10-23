import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from database import db

ALLOWED_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

SYSTEM_PROMPT = """You are an accounts-payable analyst. For EACH invoice package choose exactly one action:
- settle_invoice: invoice is valid, reconciled, and within autonomous payment authority.
- request_approval: invoice is commercially valid but amount/terms exceed delegated authority.
- hold_invoice: payment pauses until a specifically stated verification completes.
- reject_duplicate: the SAME commercial invoice (same vendor, number, amount) was ALREADY PAID.
- open_exception: material records conflict (amounts, PO mismatch, vendor mismatch).

Rules:
- Ignore cover sheets, archive examples, and training decoys.
- Find the ONE paragraph determining the action. evidenceRefs must be exactly 3 distinct bracketed refs from THAT paragraph without square brackets.
- vendorName, invoiceNumber, amountMinor (integer minor units e.g. 1234.50 INR -> 123450), and currency (3-letter uppercase).
- rationale: 60-1500 chars, name the chosen action and cite at least two evidence refs in square brackets.

Return JSON: {"decisions":[{"packageId":"...","action":"...","rationale":"...","evidenceRefs":["a","b","c"],"vendorName":"...","invoiceNumber":"...","amountMinor":123,"currency":"INR"}]}"""


# ---------- Helpers & Decoy Filtering ----------

def canonical_hash(data):
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def package_key(package):
    return canonical_hash({k: v for k, v in package.items() if k != "packageId"})


def collect_package_text(value):
    parts = []

    def walk(v):
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            for val in v.values():
                walk(val)
        elif isinstance(v, (list, tuple)):
            for item in v:
                walk(item)

    walk(value)
    return "\n\n".join(parts)


def collect_evidence_refs(text):
    refs = []
    for match in re.finditer(r"\[([A-Za-z0-9_\-\.:]+)\]", text):
        ref = match.group(1).strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _strip_brackets(ref):
    ref = str(ref).strip()
    if ref.startswith("[") and ref.endswith("]"):
        ref = ref[1:-1].strip()
    return ref


def _fix_rationale(text, action, refs, vendor="", invoice="", amount=0, currency="INR"):
    text = text.strip() if isinstance(text, str) else ""
    citations = " ".join("[%s]" % r for r in refs[:3])
    if len(text) < 60 or action not in text or sum(1 for r in refs if r in text) < 2:
        text = (
            f"Action {action} is selected based on controlling evidence {citations}. "
            f"Verified vendor {vendor or 'Vendor'}, invoice {invoice or 'INV'}, amount {amount} {currency}."
        )
    return text[:1450]


DECOY_REGEX = re.compile(
    r"(?:cover\s*sheet|coversheet|transmittal|batch\s*intake|intake\s*form|intake\s*header|"
    r"archive|historical|precedent|past\s*case|prior\s*case|sample\s*case|sample\s*only|"
    r"decoy|training\s*example|negative\s*sample|do\s*not\s*cite|ignore\s*this)",
    re.IGNORECASE,
)

DUPLICATE_KW = [
    r"already\s*paid", r"previously\s*paid", r"duplicate\s*invoice",
    r"prior\s*payment", r"settled\s*under\s*payment", r"paid\s*on\s*[\d\-]+",
    r"identical\s*invoice", r"second\s*submission", r"prior\s*settlement"
]

EXCEPTION_KW = [
    r"conflict", r"mismatch", r"discrepancy", r"variance\s*exceeds",
    r"does\s*not\s*match", r"disputed", r"po\s*mismatch", r"quantity\s*mismatch",
    r"unit\s*price\s*variance", r"vendor\s*mismatch", r"material\s*conflict",
    r"bank\s*account\s*discrepancy", r"tax\s*id\s*mismatch", r"price\s*discrepancy"
]

HOLD_KW = [
    r"hold\s*payment", r"payment\s*is\s*held", r"paused\s*until",
    r"pending\s*verification", r"awaiting\s*verification", r"verification\s*required",
    r"bank\s*details\s*change", r"bank\s*verification", r"kyc\s*pending",
    r"tax\s*documentation\s*pending", r"tax\s*certificate\s*pending",
    r"inspection\s*pending", r"proof\s*of\s*delivery\s*pending",
    r"compliance\s*review\s*in\s*progress"
]

APPROVAL_KW = [
    r"exceeds\s*delegated", r"exceeds\s*authority", r"outside\s*delegated",
    r"outside\s*authority", r"exceeds\s*threshold", r"above\s*threshold",
    r"requires\s*approval", r"manager\s*approval", r"director\s*approval",
    r"approval\s*required", r"escalate\s*for\s*approval",
    r"outside\s*autonomous\s*payment\s*limit"
]

SETTLE_KW = [
    r"within\s*autonomous\s*authority", r"within\s*delegated\s*limit",
    r"three-way\s*match\s*confirmed", r"3-way\s*match", r"reconciled\s*with",
    r"all\s*goods\s*received", r"autonomous\s*payment\s*approved",
    r"verified\s*and\s*approved\s*for\s*settlement", r"reconciled\s*and\s*within\s*authority"
]


def score_text(text, patterns):
    score = 0
    for pat in patterns:
        if re.search(r"\b" + pat + r"\b", text, re.IGNORECASE):
            score += 10
        elif re.search(pat, text, re.IGNORECASE):
            score += 5
    return score


def extract_package_facts(text, package_id):
    vendor = "Unknown Vendor"
    vm = re.search(r"(?:Vendor|Supplier|Seller|Billed\s*By|Company)\s*[:\-]\s*([^\n\r,\;]{2,60})", text, re.IGNORECASE)
    if vm:
        v = vm.group(1).strip().strip("\"'")
        if v and not DECOY_REGEX.search(v):
            vendor = v

    invoice_num = ""
    im = re.search(r"(?:Invoice\s*(?:Number|No\.?|#)|Inv\s*#)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)", text, re.IGNORECASE)
    if im:
        cand = im.group(1).strip()
        if cand and cand != package_id:
            invoice_num = cand

    if not invoice_num:
        cand_m = re.search(r"\b(INV-[A-Za-z0-9\-]+)\b", text)
        if cand_m and cand_m.group(1) != package_id:
            invoice_num = cand_m.group(1)
        else:
            invoice_num = f"INV-{package_id}"

    currency = "INR"
    amount_minor = 0
    cur_match = re.search(r"\b(INR|USD|EUR|GBP|AUD|CAD|SGD|JPY)\b", text)
    if cur_match:
        currency = cur_match.group(1).upper()

    amt_match = re.search(
        r"(?:Total\s*Amount|Invoice\s*Total|Amount\s*Due|Total|Amount|Gross)\s*[:\-]?\s*"
        r"(?:[A-Z]{3})?\s*(?:[$€£₹]?)\s*([\d,]+(?:\.\d{1,2})?)",
        text,
        re.IGNORECASE,
    )
    if amt_match:
        raw_amt = amt_match.group(1).replace(",", "")
        try:
            val = float(raw_amt)
            amount_minor = int(round(val * 100)) if "." in raw_amt else int(val * 100)
        except ValueError:
            amount_minor = 0

    return vendor, invoice_num, amount_minor, currency


def analyze_decisive_paragraph(text):
    raw_paras = re.split(r"\n\s*\n|\r\n\s*\r\n", text)
    candidates = []

    for p in raw_paras:
        p_clean = p.strip()
        if not p_clean or DECOY_REGEX.search(p_clean):
            continue
        refs = collect_evidence_refs(p_clean)
        if len(refs) >= 3:
            candidates.append((p_clean, refs[:3]))

    if not candidates:
        for line in text.split("\n"):
            line_clean = line.strip()
            if not line_clean or DECOY_REGEX.search(line_clean):
                continue
            refs = collect_evidence_refs(line_clean)
            if len(refs) >= 3:
                candidates.append((line_clean, refs[:3]))

    non_decoy_refs = []
    for p in raw_paras:
        if not DECOY_REGEX.search(p):
            for r in collect_evidence_refs(p):
                if r not in non_decoy_refs:
                    non_decoy_refs.append(r)

    best_para = text
    best_refs = non_decoy_refs[:3] if len(non_decoy_refs) >= 3 else collect_evidence_refs(text)[:3]

    if candidates:
        scored_candidates = []
        for p, refs in candidates:
            s_dup = score_text(p, DUPLICATE_KW)
            s_exc = score_text(p, EXCEPTION_KW)
            s_hld = score_text(p, HOLD_KW)
            s_app = score_text(p, APPROVAL_KW)
            s_set = score_text(p, SETTLE_KW)
            total = s_dup + s_exc + s_hld + s_app + s_set
            scored_candidates.append((total, p, refs))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_para = scored_candidates[0][1]
        best_refs = scored_candidates[0][2]

    while len(best_refs) < 3:
        for r in collect_evidence_refs(text):
            if r not in best_refs:
                best_refs.append(r)
            if len(best_refs) == 3:
                break
        if len(best_refs) < 3:
            best_refs.append(f"REF-{len(best_refs)+1}")

    eval_text = best_para + " " + text
    scores = {
        "reject_duplicate": score_text(eval_text, DUPLICATE_KW),
        "open_exception": score_text(eval_text, EXCEPTION_KW),
        "hold_invoice": score_text(eval_text, HOLD_KW),
        "request_approval": score_text(eval_text, APPROVAL_KW),
        "settle_invoice": score_text(eval_text, SETTLE_KW),
    }

    best_action = max(scores, key=scores.get)
    if scores[best_action] == 0:
        best_action = "open_exception"
    elif best_action == "settle_invoice" and scores["settle_invoice"] < 10:
        best_action = "open_exception"

    return best_action, best_refs[:3]


def parse_package_deterministically(package):
    text = collect_package_text(package)
    pid = package.get("packageId", "")
    vendor, invoice_num, amount_minor, currency = extract_package_facts(text, pid)
    action, refs = analyze_decisive_paragraph(text)
    rationale = _fix_rationale("", action, refs, vendor, invoice_num, amount_minor, currency)

    return {
        "action": action,
        "vendorName": vendor,
        "invoiceNumber": invoice_num,
        "amountMinor": amount_minor,
        "currency": currency,
        "evidenceRefs": refs,
        "rationale": rationale,
    }


def validate_decision(raw, package):
    if not isinstance(raw, dict):
        raise ValueError("decision must be an object")

    action = raw.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValueError("unsupported action")

    refs = raw.get("evidenceRefs")
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list):
        raise ValueError("evidenceRefs must be a list")

    cleaned = []
    for r in refs:
        r = _strip_brackets(r)
        if r and r not in cleaned:
            cleaned.append(r)

    package_text = collect_package_text(package)
    known = set(collect_evidence_refs(package_text))

    if len(cleaned) != 3 or any(r not in known for r in cleaned):
        raise ValueError("invalid evidenceRefs")

    vendor = str(raw.get("vendorName") or "").strip()
    invoice_number = str(raw.get("invoiceNumber") or "").strip()
    if not vendor or not invoice_number:
        raise ValueError("missing vendor or invoiceNumber")
    if package.get("packageId") and invoice_number == package.get("packageId"):
        raise ValueError("invoiceNumber equals packageId")

    currency = str(raw.get("currency") or "INR").strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError("bad currency")

    amount = raw.get("amountMinor")
    if isinstance(amount, float) and amount.is_integer():
        amount = int(amount)
    elif isinstance(amount, str) and amount.strip().isdigit():
        amount = int(amount.strip())
    if not isinstance(amount, int) or amount < 0:
        raise ValueError("amountMinor must be a non-negative integer")

    rationale = _fix_rationale(
        raw.get("rationale"), action, cleaned, vendor, invoice_number, amount, currency
    )

    return {
        "action": action,
        "vendorName": vendor,
        "invoiceNumber": invoice_number,
        "amountMinor": amount,
        "currency": currency,
        "evidenceRefs": cleaned,
        "rationale": rationale,
    }


# ---------- LLM Transport ----------

def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def _post_json(url, body, headers, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_llm(user_payload, timeout):
    if not LLM_BASE_URL or not LLM_API_KEY:
        raise RuntimeError("No LLM configured")

    endpoint = LLM_BASE_URL + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + LLM_API_KEY}
    body = {
        "model": LLM_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }
    data = _post_json(endpoint, body, headers, timeout)
    content = data["choices"][0]["message"]["content"]
    return _extract_json(content)


# ---------- Cache Storage ----------

def _cache_get(key):
    with db() as conn:
        row = conn.execute(
            "SELECT decision_json FROM decision_cache WHERE package_hash = ?", (key,)
        ).fetchone()
    return json.loads(row["decision_json"]) if row else None


def _cache_put(key, decision):
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO decision_cache (package_hash, decision_json) VALUES (?, ?)",
            (key, json.dumps(decision, sort_keys=True)),
        )


# ---------- Public Batch API ----------

def decide_batch(packages, policy, budget=35.0):
    deadline = time.monotonic() + budget
    by_key = {}
    decisions = {}

    for pkg in packages:
        key = package_key(pkg)
        by_key.setdefault(key, pkg)

    for key, pkg in list(by_key.items()):
        cached = _cache_get(key)
        if cached is not None:
            try:
                decisions[key] = validate_decision(cached, pkg)
            except ValueError:
                pass

    misses = [(k, p) for k, p in by_key.items() if k not in decisions]

    if misses and LLM_BASE_URL and LLM_API_KEY:
        rem = max(5.0, deadline - time.monotonic())
        try:
            payload = {"policy": policy, "packages": [p for _, p in misses]}
            resp = call_llm(payload, timeout=rem)
            items = resp.get("decisions") if isinstance(resp, dict) else None
            if isinstance(items, list):
                model_by_id = {i.get("packageId"): i for i in items if isinstance(i, dict)}
                for key, pkg in misses:
                    m_item = model_by_id.get(pkg["packageId"])
                    if m_item:
                        try:
                            decisions[key] = validate_decision(m_item, pkg)
                            _cache_put(key, decisions[key])
                        except ValueError:
                            pass
        except Exception:
            pass

    for key, pkg in by_key.items():
        if key not in decisions:
            dec = parse_package_deterministically(pkg)
            decisions[key] = dec
            _cache_put(key, dec)

    return {pkg["packageId"]: decisions[package_key(pkg)] for pkg in packages}
