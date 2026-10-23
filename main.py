import hashlib
import hmac
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from database import db, init_db, transaction
from decision_engine import ALLOWED_ACTIONS, canonical_hash, decide_batch

BASE_URL = os.getenv("BASE_URL", "https://your-public-host.example/a2a/").rstrip("/") + "/"
BASE_PATH = urlparse(BASE_URL).path.rstrip("/")

BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSAL_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RESULT_MEDIA = "application/vnd.ga5.invoice-action-results+json"
RECEIPT_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"

TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}
MAX_RESPONSE_BYTES = 450 * 1024

app = FastAPI(title="Invoice Action Agent", redirect_slashes=False)
router = APIRouter(prefix=BASE_PATH) if BASE_PATH else APIRouter()


@app.on_event("startup")
def startup_event():
    init_db()


# ---------- Responses & Errors ----------

def a2a_response(data, status_code=200):
    return JSONResponse(
        content=data,
        status_code=status_code,
        media_type="application/a2a+json",
    )


def error_response(status, code):
    return a2a_response({"error": {"code": code, "status": status}}, status)


@app.exception_handler(HTTPException)
async def http_exc(_request, exc):
    code = exc.detail if isinstance(exc.detail, str) and exc.detail.isupper() else "ERROR"
    if exc.status_code == 404 and code == "ERROR":
        code = "NOT_FOUND"
    return error_response(exc.status_code, code)


@app.exception_handler(RequestValidationError)
async def validation_exc(_request, _exc):
    return error_response(400, "INVALID_REQUEST")


@app.exception_handler(Exception)
async def unhandled_exc(_request, _exc):
    return error_response(500, "INTERNAL")


# ---------- Authentication & Helpers ----------

def authenticate(authorization: str, version: str):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "UNAUTHENTICATED")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(401, "UNAUTHENTICATED")
    if version and version != "1.0":
        raise HTTPException(400, "UNSUPPORTED_VERSION")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def require_a2a_type(content_type: str):
    if (content_type or "").split(";")[0].strip().lower() != "application/a2a+json":
        raise HTTPException(415, "UNSUPPORTED_MEDIA_TYPE")


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dumps(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


_msg_locks = {}
_msg_locks_guard = threading.Lock()

_task_locks = {}
_task_locks_guard = threading.Lock()


def key_lock(principal: str, message_id: str):
    with _msg_locks_guard:
        return _msg_locks.setdefault((principal, message_id), threading.Lock())


def task_lock(task_id: str):
    with _task_locks_guard:
        return _task_locks.setdefault(task_id, threading.Lock())


def find_part(message, media):
    for part in message.get("parts") or []:
        if isinstance(part, dict) and part.get("mediaType") == media and isinstance(part.get("data"), dict):
            return part["data"]
    return None


def get_owned_task(conn, task_id: str, principal: str):
    row = conn.execute(
        "SELECT task_json FROM tasks WHERE task_id = ? AND principal = ?",
        (task_id, principal),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "NOT_FOUND")
    return json.loads(row["task_json"])


def lookup_message(principal: str, message_id: str):
    with db() as conn:
        return conn.execute(
            "SELECT message_hash, task_id, response_json FROM messages WHERE principal = ? AND message_id = ?",
            (principal, message_id),
        ).fetchone()


def validate_batch(batch):
    if not isinstance(batch, dict) or not isinstance(batch.get("batchId"), str) or not batch["batchId"]:
        raise HTTPException(400, "INVALID_BATCH")
    packages = batch.get("packages")
    if not isinstance(packages, list) or not packages:
        raise HTTPException(400, "INVALID_BATCH")
    ids = []
    for p in packages:
        if not isinstance(p, dict) or not isinstance(p.get("packageId"), str) or not p["packageId"]:
            raise HTTPException(400, "INVALID_BATCH")
        ids.append(p["packageId"])
    if len(ids) != len(set(ids)):
        raise HTTPException(400, "INVALID_BATCH")


def make_agent_card(target_url: str = None):
    effective_url = target_url or BASE_URL
    return {
        "name": "Invoice Action Agent",
        "description": "Reviews invoice evidence, decides actions, and safely executes accepted proposals.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [BATCH_MEDIA],
        "defaultOutputModes": [PROPOSAL_MEDIA, RECEIPT_MEDIA],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Proposes settle, approval, hold, duplicate-reject or exception actions for invoice packages.",
            "tags": ["invoice", "payments", "finance"],
            "inputModes": [BATCH_MEDIA],
            "outputModes": [PROPOSAL_MEDIA, RECEIPT_MEDIA],
        }],
        "supportedInterfaces": [{
            "url": effective_url,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }],
    }


# ---------- Core Logic ----------

def create_task(principal, message, message_id, message_hash):
    batch = find_part(message, BATCH_MEDIA)
    if batch is None:
        raise HTTPException(400, "INVALID_MESSAGE")
    validate_batch(batch)

    policy = {k: v for k, v in batch.items() if k != "packages"}
    decisions = decide_batch(batch["packages"], policy)

    proposals = []
    for pkg in batch["packages"]:
        d = decisions[pkg["packageId"]]
        proposals.append({
            "packageId": pkg["packageId"],
            "actionId": "act_" + uuid.uuid4().hex,
            "action": d["action"],
            "facts": {
                "vendorName": d["vendorName"],
                "invoiceNumber": d["invoiceNumber"],
                "amountMinor": d["amountMinor"],
                "currency": d["currency"],
            },
            "evidenceRefs": d["evidenceRefs"],
            "rationale": d["rationale"],
        })

    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
        "history": [message],
        "artifacts": [{
            "artifactId": str(uuid.uuid4()),
            "name": "invoice-action-proposals",
            "parts": [{
                "mediaType": PROPOSAL_MEDIA,
                "data": {"batchId": batch["batchId"], "proposals": proposals},
            }],
        }],
    }
    response = {"task": task}

    with transaction() as conn:
        conn.execute(
            "INSERT INTO tasks (task_id, context_id, principal, batch_id, state, task_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, context_id, principal, batch["batchId"], task["status"]["state"], dumps(task)),
        )
        conn.execute(
            "INSERT INTO messages (principal, message_id, message_hash, task_id, response_json) VALUES (?, ?, ?, ?, ?)",
            (principal, message_id, message_hash, task_id, dumps(response)),
        )
    return response


def continue_task(principal, message, message_id, message_hash):
    task_id = message.get("taskId")
    context_id = message.get("contextId")
    if not isinstance(task_id, str) or not isinstance(context_id, str) or not context_id:
        raise HTTPException(400, "INVALID_MESSAGE")
    data = find_part(message, RESULT_MEDIA)
    if data is None or not isinstance(data.get("results"), list) or not data["results"]:
        raise HTTPException(400, "INVALID_MESSAGE")

    with task_lock(task_id):
        with transaction() as conn:
            task = get_owned_task(conn, task_id, principal)
            if task.get("contextId") != context_id:
                raise HTTPException(400, "CONTEXT_MISMATCH")
            if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
                raise HTTPException(409, "TASK_NOT_AWAITING_RESULTS")

            proposal_data = None
            for art in task.get("artifacts", []):
                for part in art.get("parts") or []:
                    if part.get("mediaType") == PROPOSAL_MEDIA and isinstance(part.get("data"), dict):
                        proposal_data = part["data"]
                        break
                if proposal_data:
                    break

            if not proposal_data:
                raise HTTPException(409, "PROPOSALS_MISSING")
            if data.get("batchId") != proposal_data.get("batchId"):
                raise HTTPException(400, "BATCH_MISMATCH")

            index = {p["packageId"]: p for p in proposal_data["proposals"]}
            seen, executions, stored = set(), [], []
            for r in data["results"]:
                if not isinstance(r, dict):
                    raise HTTPException(400, "INVALID_RESULT")
                pid = r.get("packageId")
                prop = index.get(pid)
                if prop is None or pid in seen:
                    raise HTTPException(400, "PACKAGE_MISMATCH")
                if r.get("actionId") != prop["actionId"]:
                    raise HTTPException(400, "ACTION_ID_MISMATCH")
                if r.get("action") != prop["action"] or r.get("action") not in ALLOWED_ACTIONS:
                    raise HTTPException(400, "ACTION_MISMATCH")
                outcome = r.get("outcome")
                if outcome not in {"ACCEPTED", "REJECTED"}:
                    raise HTTPException(400, "INVALID_OUTCOME")
                nonce = r.get("receiptNonce")
                if outcome == "ACCEPTED":
                    if not isinstance(nonce, str) or not nonce:
                        raise HTTPException(400, "RECEIPT_MISSING")
                    executions.append({
                        "packageId": pid,
                        "actionId": prop["actionId"],
                        "action": prop["action"],
                        "receiptNonce": nonce,
                        "facts": prop["facts"],
                        "evidenceRefs": prop["evidenceRefs"],
                    })
                seen.add(pid)
                stored.append((pid, prop["actionId"], outcome, nonce if outcome == "ACCEPTED" else None))

            task["history"].append(message)
            task["artifacts"].append({
                "artifactId": str(uuid.uuid4()),
                "name": "invoice-action-receipts",
                "parts": [{
                    "mediaType": RECEIPT_MEDIA,
                    "data": {"batchId": proposal_data["batchId"], "executions": executions},
                }],
            })
            task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
            response = {"task": task}

            conn.execute(
                "UPDATE tasks SET state = ?, task_json = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE task_id = ? AND principal = ?",
                ("TASK_STATE_COMPLETED", dumps(task), task_id, principal),
            )
            for pid, aid, outcome, nonce in stored:
                conn.execute(
                    "INSERT OR IGNORE INTO task_results (task_id, package_id, action_id, outcome, receipt_nonce) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (task_id, pid, aid, outcome, nonce),
                )
            conn.execute(
                "INSERT INTO messages (principal, message_id, message_hash, task_id, response_json) VALUES (?, ?, ?, ?, ?)",
                (principal, message_id, message_hash, task_id, dumps(response)),
            )
    return response


def handle_message(principal, message):
    message_id = message["messageId"]
    message_hash = canonical_hash(message)

    with key_lock(principal, message_id):
        row = lookup_message(principal, message_id)
        if row is not None:
            if row["message_hash"] != message_hash:
                raise HTTPException(409, "IDEMPOTENCY_CONFLICT")
            task_id = row["task_id"]
            with db() as conn:
                task_row = conn.execute(
                    "SELECT task_json FROM tasks WHERE task_id = ? AND principal = ?",
                    (task_id, principal),
                ).fetchone()
            if task_row:
                return {"task": json.loads(task_row["task_json"])}
            return json.loads(row["response_json"])

        if message.get("taskId"):
            return continue_task(principal, message, message_id, message_hash)
        return create_task(principal, message, message_id, message_hash)


# ---------- Routes ----------

@app.get("/.well-known/agent-card.json")
def agent_card(request: Request):
    origin = f"{request.url.scheme}://{request.url.netloc}"
    req_base = f"{origin}/a2a/"
    env_url = os.getenv("BASE_URL", "").strip()
    if env_url and request.url.netloc in env_url:
        card_url = env_url.rstrip("/") + "/"
    else:
        card_url = req_base
    return a2a_response(make_agent_card(card_url))


@router.post("/message:send")
async def send_message(
    request: Request,
    authorization: str = Header(default=None),
    a2a_version: str = Header(default=None, alias="A2A-Version"),
    content_type: str = Header(default=None),
):
    principal = authenticate(authorization, a2a_version)
    require_a2a_type(content_type)
    try:
        body = json.loads(await request.body())
    except Exception:
        raise HTTPException(400, "INVALID_JSON")

    message = body.get("message") if isinstance(body, dict) else None
    if (
        not isinstance(message, dict)
        or message.get("role") != "ROLE_USER"
        or not isinstance(message.get("messageId"), str)
        or not message["messageId"]
        or not isinstance(message.get("parts"), list)
    ):
        raise HTTPException(400, "INVALID_MESSAGE")

    result = await run_in_threadpool(handle_message, principal, message)
    return a2a_response(result)


@router.get("/tasks")
def list_tasks(
    authorization: str = Header(default=None),
    a2a_version: str = Header(default=None, alias="A2A-Version"),
):
    principal = authenticate(authorization, a2a_version)
    with db() as conn:
        rows = conn.execute(
            "SELECT task_json FROM tasks WHERE principal = ? ORDER BY created_at DESC, rowid DESC",
            (principal,),
        ).fetchall()

    tasks, total = [], 0
    for row in rows:
        task = json.loads(row["task_json"])
        size = len(dumps(task))
        if total + size > MAX_RESPONSE_BYTES:
            slim = dict(task, history=[])
            size = len(dumps(slim))
            if total + size > MAX_RESPONSE_BYTES:
                break
            task = slim
        total += size
        tasks.append(task)
    return a2a_response({"tasks": tasks})


@router.get("/tasks/{task_id}")
def get_task(
    task_id: str,
    authorization: str = Header(default=None),
    a2a_version: str = Header(default=None, alias="A2A-Version"),
):
    principal = authenticate(authorization, a2a_version)
    with db() as conn:
        task = get_owned_task(conn, task_id, principal)
    return a2a_response(task)


@router.post("/tasks/{task_id}:cancel")
def cancel_task(
    task_id: str,
    authorization: str = Header(default=None),
    a2a_version: str = Header(default=None, alias="A2A-Version"),
):
    principal = authenticate(authorization, a2a_version)
    with task_lock(task_id):
        with transaction() as conn:
            task = get_owned_task(conn, task_id, principal)
            if task["status"]["state"] in TERMINAL:
                raise HTTPException(409, "TASK_TERMINAL")
            task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
            conn.execute(
                "UPDATE tasks SET state = ?, task_json = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE task_id = ? AND principal = ?",
                ("TASK_STATE_CANCELED", dumps(task), task_id, principal),
            )
    return a2a_response(task)


app.include_router(router)
if BASE_PATH:
    app.include_router(router, prefix="")
