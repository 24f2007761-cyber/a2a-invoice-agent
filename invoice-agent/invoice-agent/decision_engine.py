"""AI decision layer: one batched model call per batch, cached by package content."""
import hashlib
import json
import logging
import os
import re
import time

import database as db

log = logging.getLogger("engine")
ACTIONS = ("settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception")
REF_RE = re.compile(r"\[[^\[\]\n]{1,200}\]")
FACT_KEYS = ("vendorName", "invoiceNumber", "amountMinor", "currency")

SYSTEM = """You are an accounts-payable reviewer. For EACH invoice package choose exactly ONE action:
- settle_invoice: valid, reconciled, and within autonomous payment authority.
- request_approval: commercially valid but the amount/terms are outside delegated authority.
- hold_invoice: payment pauses until a stated verification completes (bank-detail callback, missing PO/receipt, tax/KYC check...).
- reject_duplicate: the SAME commercial invoice (same vendor, invoice number and amount) was already paid. A payment for a different invoice number is NOT a duplicate.
- open_exception: material records conflict (amount, PO, vendor, currency, receipt mismatch) and need an exception workflow.

Rules:
- The documents mix real facts with archive examples, old/superseded policy revisions, training decoys, negations ("no approval needed", "was not paid") and irrelevant action words. Only the CURRENT evidence for THIS invoice under the stated policy revision counts.
- Ignore the cover sheet / summary paragraph, archive examples, training examples, and other revisions.
- Exactly one paragraph per package is the decisive paragraph: the one that states the current evidence determining the action. Return its id (e.g. "P4").
- facts: vendorName, invoiceNumber (strings), amountMinor (integer minor units, e.g. 12345.67 INR -> 1234567), currency (ISO code) of THIS invoice.
- reason: 1-2 sentences naming the decisive facts.

Return ONLY JSON: {"decisions":[{"index":<int>,"action":"<action>","paragraphId":"P<n>","reason":"...","facts":{"vendorName":"","invoiceNumber":"","amountMinor":0,"currency":""}}]}
Include every package index given."""


def canonical(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def package_key(pkg):
    clean = {k: v for k, v in pkg.items() if k != "packageId"}
    return hashlib.sha256(canonical(clean).encode()).hexdigest()


def split_paragraphs(pkg):
    out = []

    def add(path, text):
        for chunk in re.split(r"\n\s*\n", text):
            chunk = chunk.strip()
            if not chunk:
                continue
            if len(REF_RE.findall(chunk)) > 3:
                out.extend((path, ln.strip()) for ln in chunk.split("\n") if ln.strip())
            else:
                out.append((path, chunk))

    def walk(v, path):
        if isinstance(v, str):
            add(path, v)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{path}/{i}")
        elif isinstance(v, dict):
            for k, x in v.items():
                if k != "packageId":
                    walk(x, f"{path}/{k}")
        elif v is not None and not isinstance(v, bool):
            add(path, str(v))

    walk(pkg, "")
    return out


def _structured_facts(pkg):
    """Take facts from top-level (or one-level nested) fields when present."""
    found = {}
    layers = [pkg] + [v for v in pkg.values() if isinstance(v, dict)]
    for layer in layers:
        for k in FACT_KEYS:
            if k not in found and layer.get(k) not in (None, ""):
                found[k] = layer[k]
    return found


def _norm_facts(f):
    try:
        return {
            "vendorName": str(f["vendorName"]),
            "invoiceNumber": str(f["invoiceNumber"]),
            "amountMinor": int(round(float(str(f["amountMinor"]).replace(",", "")))),
            "currency": str(f["currency"]).upper(),
        }
    except Exception:
        return None


def _chat(user, timeout):
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY") or os.getenv("AIPIPE_TOKEN") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://aipipe.org/openai/v1"),
        timeout=timeout,
        max_retries=0,
    )
    r = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content


def _stub(paras_by_i):
    ds = []
    for i, paras in paras_by_i.items():
        n = next((k for k, (_, t) in enumerate(paras) if len(REF_RE.findall(t)) == 3), 0)
        ds.append({"index": i, "action": "hold_invoice", "paragraphId": f"P{n}", "reason": "stub decision.",
                   "facts": {"vendorName": "Stub Vendor", "invoiceNumber": "S-1", "amountMinor": 100, "currency": "INR"}})
    return json.dumps({"decisions": ds})


def _ask(items, policy_rev, timeout):
    paras_by_i, blocks = {}, []
    for i, pkg in items:
        paras = split_paragraphs(pkg)
        paras_by_i[i] = paras
        body = "\n".join(f"#P{n} ({p}): {t}" for n, (p, t) in enumerate(paras))
        blocks.append(f"=== PACKAGE index={i} ===\n{body}")
    if os.getenv("LLM_STUB") == "1":
        raw = _stub(paras_by_i)
    else:
        raw = _chat(f"Current policy revision: {policy_rev}\n\n" + "\n\n".join(blocks), timeout)
    out = {}
    pkgs = dict(items)
    for d in json.loads(raw).get("decisions", []):
        try:
            i = int(d["index"])
            if i not in paras_by_i or d.get("action") not in ACTIONS:
                continue
            n = int(re.search(r"\d+", str(d["paragraphId"])).group())
            refs = list(dict.fromkeys(REF_RE.findall(paras_by_i[i][n][1])))
            if not refs:
                continue
            facts = _structured_facts(pkgs[i])
            for k in FACT_KEYS:
                facts.setdefault(k, (d.get("facts") or {}).get(k))
            facts = _norm_facts(facts)
            if not facts:
                continue
            out[i] = {"action": d["action"], "refs": refs, "facts": facts,
                      "reason": str(d.get("reason", ""))[:600]}
        except Exception:
            log.exception("bad decision item")
    return out


def decide(policy_rev, packages):
    """Return one decision per package (cached by canonical package content)."""
    start = time.time()
    keys = [package_key(p) for p in packages]
    results, pending = [None] * len(packages), []
    for i, k in enumerate(keys):
        c = db.get_decision(k)
        if c:
            results[i] = c
        else:
            pending.append(i)
    for _ in range(2):
        if not pending or (time.time() - start > 15 and _ == 1):
            break
        got = _ask([(i, packages[i]) for i in pending], policy_rev, max(5, 38 - (time.time() - start)))
        for i, dec in got.items():
            results[i] = dec
            db.put_decision(keys[i], dec)
        pending = [i for i in pending if results[i] is None]
    if pending:
        raise RuntimeError("some packages could not be decided")
    return results


def build_proposal(batch_id, package_id, dec):
    action, f = dec["action"], dec["facts"]
    action_id = "act-" + hashlib.sha256(f"{batch_id}|{package_id}|{action}".encode()).hexdigest()[:24]
    rationale = (
        f"Proposing {action} for invoice {f['invoiceNumber']} from {f['vendorName']} "
        f"({f['amountMinor']} minor units {f['currency']}). {dec.get('reason', '')} "
        f"Decisive evidence: {'; '.join(dec['refs'])}."
    )[:1500]
    if len(rationale) < 60:
        rationale = rationale.ljust(60, ".")
    return {"packageId": package_id, "actionId": action_id, "action": action,
            "facts": f, "evidenceRefs": dec["refs"], "rationale": rationale}
