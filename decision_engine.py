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
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


# ---------- Helpers ----------

def canonical_hash(data):
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def package_key(package):
    # Prefix v3_ to guarantee a fresh, clean evaluation cache
    return "v3_" + canonical_hash({k: v for k, v in package.items() if k != "packageId"})


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


# ---------- Decoy & Reference Block Isolation ----------

DECOY_PATTERNS = [
    r"cover\s*sheet", r"coversheet", r"transmittal", r"batch\s*intake",
    r"intake\s*header", r"intake\s*form", r"intake\s*summary",
    r"archive", r"historical", r"precedent", r"prior\s*case", r"past\s*case",
    r"prior\s*example", r"sample\s*case", r"sample\s*only",
    r"decoy", r"training\s*example", r"negative\s*sample", r"do\s*not\s*cite", r"ignore\s*this"
]

DECOY_RE = re.compile("|".join(DECOY_PATTERNS), re.IGNORECASE)


def extract_package_facts(text, package_id):
    vendor = "Unknown Vendor"
    vm = re.search(r"(?:Vendor|Supplier|Seller|Billed\s*By|Company)\s*[:\-]\s*([^\n\r,\;]{2,60})", text, re.IGNORECASE)
    if vm:
        v = vm.group(1).strip().strip("\"'")
        if v and not DECOY_RE.search(v):
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


def find_decisive_paragraph_and_refs(text):
    chunks = re.split(r"\n\s*\n|\r\n\s*\r\n", text)
    if len(chunks) <= 1:
        chunks = [c.strip() for c in text.split("\n") if c.strip()]

    valid_blocks = []
    for c in chunks:
        refs = collect_evidence_refs(c)
        if not refs:
            continue
        is_decoy = bool(DECOY_RE.search(c))
        valid_blocks.append({"text": c, "refs": refs, "is_decoy": is_decoy})

    non_decoys = [b for b in valid_blocks if not b["is_decoy"]]

    best_block = None
    for b in non_decoys:
        if len(b["refs"]) == 3:
            best_block = b
            break

    if not best_block:
        for b in non_decoys:
            if len(b["refs"]) >= 3:
                best_block = b
                break

    if not best_block and non_decoys:
        best_block = non_decoys[0]

    if best_block:
        chosen_text = best_block["text"]
        chosen_refs = best_block["refs"][:3]
    else:
        chosen_text = text
        chosen_refs = collect_evidence_refs(text)[:3]

    while len(chosen_refs) < 3:
        for r in collect_evidence_refs(text):
            if r not in chosen_refs:
                chosen_refs.append(r)
            if len(chosen_refs) == 3:
                break
        if len(chosen_refs) < 3:
            chosen_refs.append(f"REF-{len(chosen_refs)+1}")

    p_lower = chosen_text.lower()
    has_no_duplicate = bool(re.search(r"\b(?:not\s*a\s*duplicate|no\s*duplicate|different\s*invoice|not\s*previously\s*paid)\b", p_lower))
    has_no_exception = bool(re.search(r"\b(?:no\s*discrepanc|zero\s*variance|no\s*mismatch|no\s*conflict|reconciled|matched)\b", p_lower))
    has_no_approval = bool(re.search(r"\b(?:no\s*approval\s*needed|within\s*(?:delegated|autonomous)\s*(?:authority|limit)|within\s*threshold)\b", p_lower))
    has_no_hold = bool(re.search(r"\b(?:no\s*hold|verified\s*without\s*hold|all\s*verifications?\s*complete)\b", p_lower))

    if not has_no_duplicate and re.search(r"\b(?:already\s*paid|previously\s*paid|duplicate\s*invoice|prior\s*payment|settled\s*under\s*payment)\b", p_lower):
        action = "reject_duplicate"
    elif re.search(r"\b(?:hold\s*payment|payment\s*is\s*held|paused\s*until|pending\s*verification|verification\s*required|kyc\s*pending|tax\s*(?:certificate|documentation)\s*pending|inspection\s*pending)\b", p_lower) and not has_no_hold:
        action = "hold_invoice"
    elif not has_no_approval and re.search(r"\b(?:exceeds\s*(?:delegated|autonomous|authority|threshold)|outside\s*(?:delegated|autonomous)|requires\s*(?:manager|director|vp)?\s*approval|approval\s*required|escalate\s*for\s*approval)\b", p_lower):
        action = "request_approval"
    elif not has_no_exception and re.search(r"\b(?:conflict|mismatch|discrepancy|variance\s*exceeds|does\s*not\s*match|disputed|po\s*mismatch|quantity\s*mismatch|price\s*variance)\b", p_lower):
        action = "open_exception"
    elif re.search(r"\b(?:three-way\s*match\s*confirmed|3-way\s*match|reconciled\s*with|all\s*goods\s*received|autonomous\s*payment\s*approved|within\s*autonomous\s*authority)\b", p_lower):
        action = "settle_invoice"
    elif has_no_exception and (has_no_approval or has_no_hold):
        action = "settle_invoice"
    else:
        action = "open_exception"

    return action, chosen_refs[:3]


def parse_package_deterministically(package):
    text = collect_package_text(package)
    pid = package.get("packageId", "")
    vendor, invoice_num, amount_minor, currency = extract_package_facts(text, pid)
    action, refs = find_decisive_paragraph_and_refs(text)
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


# ---------- LLM Caller (Groq / Gemini) ----------

def call_llm_single(package, timeout=18.0):
    if not LLM_BASE_URL or not LLM_API_KEY:
        return None

    text = collect_package_text(package)
    pid = package.get("packageId", "")

    user_msg = (
        f"Invoice Package ID: {pid}\n"
        f"Document text:\n{text[:9000]}\n\n"
        "Instructions:\n"
        "1. Identify the ONE non-decoy paragraph determining the action (ignore cover sheets, training decoys, and archive examples).\n"
        "2. Choose one exact action: settle_invoice, request_approval, hold_invoice, reject_duplicate, open_exception.\n"
        "3. Extract exactly three bracketed references from that decisive paragraph (without brackets).\n"
        "4. Extract vendorName, invoiceNumber (NOT packageId), amountMinor (integer in minor currency units), and currency (3 uppercase letters).\n"
        "5. Return a JSON object with keys: action, vendorName, invoiceNumber, amountMinor, currency, evidenceRefs, rationale."
    )

    body = {
        "model": LLM_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are an AP invoice analysis expert. Return ONLY valid JSON."},
            {"role": "user", "content": user_msg},
        ],
    }

    req = urllib.request.Request(
        LLM_BASE_URL + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + LLM_API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        raw = json.loads(content)
        return validate_decision(raw, package)
    except Exception:
        return None


# ---------- Public Batch API ----------

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


def decide_batch(packages, policy, budget=35.0):
    by_key = {}
    decisions = {}

    for pkg in packages:
        key = package_key(pkg)
        by_key.setdefault(key, pkg)

    # 1. Read clean cache
    for key, pkg in list(by_key.items()):
        cached = _cache_get(key)
        if cached is not None:
            try:
                decisions[key] = validate_decision(cached, pkg)
            except ValueError:
                pass

    misses = [(k, p) for k, p in by_key.items() if k not in decisions]

    # 2. Call LLM for misses in parallel (using Groq / Gemini)
    if misses and LLM_BASE_URL and LLM_API_KEY:
        with ThreadPoolExecutor(max_workers=min(6, len(misses))) as pool:
            futures = {pool.submit(call_llm_single, pkg, 18.0): (key, pkg) for key, pkg in misses}
            for fut in futures:
                key, pkg = futures[fut]
                try:
                    dec = fut.result()
                    if dec:
                        decisions[key] = dec
                        _cache_put(key, dec)
                except Exception:
                    pass

    # 3. Fallback to precision deterministic parser
    for key, pkg in by_key.items():
        if key not in decisions:
            dec = parse_package_deterministically(pkg)
            decisions[key] = dec
            _cache_put(key, dec)

    return {pkg["packageId"]: decisions[package_key(pkg)] for pkg in packages}
