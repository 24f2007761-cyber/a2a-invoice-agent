"""SQLite persistence: tasks, message idempotency, decision cache, execution ledger."""
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager

DB_PATH = os.getenv("DB_PATH", "invoice_agent.db")
_lock = threading.RLock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_conn.row_factory = sqlite3.Row
_conn.execute("PRAGMA journal_mode=WAL")
_conn.executescript(
    """
CREATE TABLE IF NOT EXISTS tasks(
  principal TEXT NOT NULL, id TEXT NOT NULL, state TEXT NOT NULL,
  created REAL NOT NULL, body TEXT NOT NULL, PRIMARY KEY(principal, id));
CREATE TABLE IF NOT EXISTS messages(
  principal TEXT NOT NULL, message_id TEXT NOT NULL, hash TEXT NOT NULL,
  task_id TEXT NOT NULL, PRIMARY KEY(principal, message_id));
CREATE TABLE IF NOT EXISTS decisions(key TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS executions(
  principal TEXT NOT NULL, action_id TEXT NOT NULL, package_id TEXT, action TEXT,
  receipt_nonce TEXT, created REAL, PRIMARY KEY(principal, action_id));
"""
)


@contextmanager
def tx():
    with _lock:
        _conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            _conn.execute("COMMIT")
        except Exception:
            _conn.execute("ROLLBACK")
            raise


def get_task(principal, task_id):
    with _lock:
        r = _conn.execute(
            "SELECT body FROM tasks WHERE principal=? AND id=?", (principal, task_id)
        ).fetchone()
    return json.loads(r["body"]) if r else None


def list_tasks(principal):
    with _lock:
        rows = _conn.execute(
            "SELECT body FROM tasks WHERE principal=? ORDER BY created DESC", (principal,)
        ).fetchall()
    return [json.loads(r["body"]) for r in rows]


def save_task(principal, task):
    with _lock:
        _conn.execute(
            "INSERT INTO tasks(principal,id,state,created,body) VALUES(?,?,?,?,?) "
            "ON CONFLICT(principal,id) DO UPDATE SET state=excluded.state, body=excluded.body",
            (principal, task["id"], task["status"]["state"], time.time(), json.dumps(task)),
        )


def get_message(principal, message_id):
    with _lock:
        r = _conn.execute(
            "SELECT hash, task_id FROM messages WHERE principal=? AND message_id=?",
            (principal, message_id),
        ).fetchone()
    return dict(r) if r else None


def save_message(principal, message_id, h, task_id):
    with _lock:
        _conn.execute(
            "INSERT OR IGNORE INTO messages VALUES(?,?,?,?)", (principal, message_id, h, task_id)
        )


def get_decision(key):
    with _lock:
        r = _conn.execute("SELECT body FROM decisions WHERE key=?", (key,)).fetchone()
    return json.loads(r["body"]) if r else None


def put_decision(key, dec):
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO decisions VALUES(?,?)", (key, json.dumps(dec))
        )


def record_execution(principal, action_id, package_id, action, nonce):
    """Ledger of carried-out actions; PRIMARY KEY makes a repeat a no-op."""
    with _lock:
        cur = _conn.execute(
            "INSERT OR IGNORE INTO executions VALUES(?,?,?,?,?,?)",
            (principal, action_id, package_id, action, nonce, time.time()),
        )
        return cur.rowcount == 1
