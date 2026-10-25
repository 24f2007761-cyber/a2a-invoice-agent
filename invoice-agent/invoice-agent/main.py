"""GA5 invoice action agent - A2A 1.0 HTTP+JSON."""
import asyncio
import datetime as dt
import hashlib
import json
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.exceptions import HTTPException as StarletteHTTPException

import database as db
import decision_engine as engine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("a2a")

BASE = "/a2a"  # submit https://<your-host>/a2a as the base URL
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROP_MT = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MT = "application/vnd.ga5.invoice-action-receipts+json"
RESULT_MT = "application/vnd.ga5.invoice-action-results+json"
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

app = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None)
_locks = {}


def lock_for(key):
    return _locks.setdefault(key, asyncio.Lock())


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def jresp(obj, status=200, mt="application/a2a+json"):
    return Response(json.dumps(obj, separators=(",", ":")), status_code=status, media_type=mt)


def err(status, code, msg):
    return jresp({"error": {"code": status, "status": code, "message": msg}, "code": code}, status)


def guard(request: Request, post=False):
    """-> (principal, error_response)"""
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not token:
        return None, err(401, "UNAUTHENTICATED", "Authentication required.")
    if request.headers.get("a2a-version", "").strip() != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "Unsupported A2A version.")
    if post and not request.headers.get("content-type", "").lower().startswith("application/a2a+json"):
        return None, err(415, "UNSUPPORTED_MEDIA_TYPE", "Use application/a2a+json.")
    return hashlib.sha256(token.encode()).hexdigest(), None


@app.exception_handler(StarletteHTTPException)
async def http_exc(request, exc):
    return err(exc.status_code, "ERROR", "Request failed.")


@app.get("/healthz")
@app.get("/")
async def health():
    return {"ok": True}


# ---------------------------------------------------------------- Agent Card
def card(request: Request):
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    return {
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one business action per invoice with cited evidence, "
                       "and executes only grader-accepted proposals.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": f"https://{host}{BASE}", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [BATCH_MT, RESULT_MT],
        "defaultOutputModes": [PROP_MT, RECEIPT_MT],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice action decisions",
            "description": "Choose settle, approve, hold, reject-duplicate or exception for each invoice with evidence.",
            "tags": ["invoice", "accounts-payable", "approval", "duplicate-detection"],
        }],
    }


@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    return jresp(card(request), mt="application/json")


@app.get(BASE + "/.well-known/agent-card.json")
async def agent_card2(request: Request):
    return jresp(card(request), mt="application/json")


# ------------------------------------------------------------------- Tasks
@app.get(BASE + "/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    principal, e = guard(request)
    if e:
        return e
    t = db.get_task(principal, task_id)
    return jresp(t) if t else err(404, "NOT_FOUND", "Task not found.")


@app.get(BASE + "/tasks")
async def list_tasks(request: Request):
    principal, e = guard(request)
    if e:
        return e
    return jresp({"tasks": db.list_tasks(principal)})


@app.post(BASE + "/tasks/{task_id}:cancel")
async def cancel(task_id: str, request: Request):
    principal, e = guard(request)
    if e:
        return e
    async with lock_for(("t", principal, task_id)):
        t = db.get_task(principal, task_id)
        if not t:
            return err(404, "NOT_FOUND", "Task not found.")
        if t["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_CANCELABLE", "Task is already terminal.")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        with db.tx():
            db.save_task(principal, t)
        return jresp(t)


# ------------------------------------------------------------ message:send
def find_part(msg, mt):
    return next((p for p in msg["parts"] if isinstance(p, dict) and p.get("mediaType") == mt
                 and isinstance(p.get("data"), dict)), None)


@app.post(BASE + "/message:send")
async def message_send(request: Request):
    principal, e = guard(request, post=True)
    if e:
        return e
    try:
        body = json.loads(await request.body())
    except Exception:
        return err(400, "INVALID_ARGUMENT", "Malformed JSON.")
    msg = body.get("message") if isinstance(body, dict) else None
    if (not isinstance(msg, dict) or not isinstance(msg.get("messageId"), str) or not msg["messageId"]
            or msg.get("role") != "ROLE_USER" or not isinstance(msg.get("parts"), list) or not msg["parts"]):
        return err(400, "INVALID_ARGUMENT", "Invalid message.")
    mid = msg["messageId"]
    h = hashlib.sha256(engine.canonical(msg).encode()).hexdigest()  # message only; configuration ignored

    async with lock_for(("m", principal, mid)):
        rec = db.get_message(principal, mid)
        if rec:
            if rec["hash"] != h:
                return err(409, "IDEMPOTENCY_CONFLICT", "Message ID reused with different content.")
            t = db.get_task(principal, rec["task_id"])
            return jresp({"task": t})
        if msg.get("taskId"):
            return await handle_result(principal, msg, mid, h)
        return await handle_batch(principal, msg, mid, h)


async def handle_batch(principal, msg, mid, h):
    part = find_part(msg, BATCH_MT)
    data = part["data"] if part else None
    if (not data or not isinstance(data.get("batchId"), str) or not isinstance(data.get("packages"), list)
            or not data["packages"]):
        return err(400, "INVALID_ARGUMENT", "Invalid invoice batch.")
    pkgs = data["packages"]
    ids = [p.get("packageId") if isinstance(p, dict) else None for p in pkgs]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        return err(400, "INVALID_ARGUMENT", "Invalid or duplicate package IDs.")
    try:
        decisions = await asyncio.to_thread(engine.decide, data.get("policyRevision"), pkgs)
    except Exception:
        log.exception("decision step failed")
        return err(500, "INTERNAL", "Processing failed.")
    proposals = [engine.build_proposal(data["batchId"], pid, d) for pid, d in zip(ids, decisions)]
    task_id, ctx = "task-" + uuid.uuid4().hex, msg.get("contextId") or "ctx-" + uuid.uuid4().hex
    hist = dict(msg, taskId=task_id, contextId=ctx)
    task = {
        "id": task_id, "contextId": ctx,
        "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
        "artifacts": [{"artifactId": "proposals-" + uuid.uuid4().hex[:12], "name": "invoice-action-proposals",
                       "parts": [{"mediaType": PROP_MT, "data": {"batchId": data["batchId"], "proposals": proposals}}]}],
        "history": [hist],
    }
    with db.tx():
        db.save_task(principal, task)
        db.save_message(principal, mid, h, task_id)
    return jresp({"task": task})


async def handle_result(principal, msg, mid, h):
    task_id = msg["taskId"]
    async with lock_for(("t", principal, task_id)):
        task = db.get_task(principal, task_id)
        if not task:
            return err(404, "NOT_FOUND", "Task not found.")
        if msg.get("contextId") != task["contextId"]:
            return err(400, "INVALID_ARGUMENT", "Context mismatch.")
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_TERMINAL", "Task is already terminal.")
        part = find_part(msg, RESULT_MT)
        stored = task["artifacts"][0]["parts"][0]["data"]
        if not part or part["data"].get("batchId") != stored["batchId"] or not isinstance(part["data"].get("results"), list):
            return err(400, "INVALID_ARGUMENT", "Invalid results.")
        by_pkg = {p["packageId"]: p for p in stored["proposals"]}
        seen, accepted = set(), []
        for r in part["data"]["results"]:
            if not isinstance(r, dict):
                return err(400, "INVALID_ARGUMENT", "Invalid result.")
            prop = by_pkg.get(r.get("packageId"))
            if (not prop or r["packageId"] in seen or r.get("actionId") != prop["actionId"]
                    or r.get("action") != prop["action"] or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                return err(400, "INVALID_ARGUMENT", "Result does not match proposal.")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                if not isinstance(r.get("receiptNonce"), str) or not r["receiptNonce"]:
                    return err(400, "INVALID_ARGUMENT", "Missing receipt nonce.")
                accepted.append((prop, r["receiptNonce"]))
        executions = []
        with db.tx():
            for prop, nonce in accepted:  # the action gate: only ACCEPTED proposals run
                db.record_execution(principal, prop["actionId"], prop["packageId"], prop["action"], nonce)
                executions.append({"packageId": prop["packageId"], "actionId": prop["actionId"],
                                   "action": prop["action"], "receiptNonce": nonce,
                                   "facts": prop["facts"], "evidenceRefs": prop["evidenceRefs"]})
            task["artifacts"].append({
                "artifactId": "receipts-" + uuid.uuid4().hex[:12], "name": "invoice-action-receipts",
                "parts": [{"mediaType": RECEIPT_MT, "data": {"batchId": stored["batchId"], "executions": executions}}]})
            task["history"].append(msg)
            task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
            db.save_task(principal, task)
            db.save_message(principal, mid, h, task_id)
        return jresp({"task": task})
