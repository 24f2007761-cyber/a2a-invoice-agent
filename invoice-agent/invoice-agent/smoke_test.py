"""python smoke_test.py http://127.0.0.1:8000"""
import sys, uuid, httpx
root = sys.argv[1].rstrip("/"); base = root + "/a2a"
H = lambda tok="tok-A": {"Authorization": f"Bearer {tok}", "A2A-Version": "1.0", "Content-Type": "application/a2a+json"}
c = httpx.Client(timeout=60)
card = c.get(root + "/.well-known/agent-card.json").json(); print("card ok", card["supportedInterfaces"][0]["url"])
print("no auth ->", c.get(base + "/tasks", headers={"A2A-Version": "1.0"}).status_code)
print("bad version ->", c.get(base + "/tasks", headers={**H(), "A2A-Version": "0.3"}).status_code)
pk = lambda i: {"packageId": f"p{i}", "vendorName": "Acme", "invoiceNumber": f"INV-{i}", "amountMinor": 1000 + i, "currency": "INR",
                "docs": [{"text": "Cover [COVER-1]"}, {"text": f"Archive [A-{i}] [B-{i}] [C-{i}]\n\nCurrent [E1-{i}] [E2-{i}] [E3-{i}]"}]}
mid = uuid.uuid4().hex
msg = {"messageId": mid, "role": "ROLE_USER", "parts": [{"mediaType": "application/vnd.ga5.invoice-claim-batch+json",
       "data": {"batchId": "b1", "policyRevision": "r7", "packages": [pk(i) for i in range(3)]}}]}
r1 = c.post(base + "/message:send", headers=H(), json={"message": msg, "configuration": {"returnImmediately": False}})
t = r1.json()["task"]; print(r1.status_code, t["status"]["state"])
r2 = c.post(base + "/message:send", headers=H(), json={"configuration": {"returnImmediately": True}, "message": dict(reversed(list(msg.items())))})
print("replay same task:", r2.json()["task"]["id"] == t["id"])
bad = dict(msg, parts=[{**msg["parts"][0], "data": {"batchId": "other", "packages": [pk(9)]}}])
print("conflict ->", c.post(base + "/message:send", headers=H(), json={"message": bad}).status_code)
print("outsider get ->", c.get(f"{base}/tasks/{t['id']}", headers=H("tok-B")).status_code, c.get(base + "/tasks", headers=H("tok-B")).json())
props = t["artifacts"][0]["parts"][0]["data"]["proposals"]
res = {"message": {"messageId": uuid.uuid4().hex, "taskId": t["id"], "contextId": t["contextId"], "role": "ROLE_USER",
       "parts": [{"mediaType": "application/vnd.ga5.invoice-action-results+json", "data": {"batchId": "b1", "results": [
           {"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
            "outcome": "ACCEPTED" if i else "REJECTED", "receiptNonce": f"n{i}"} for i, p in enumerate(props)]}}]}}
r3 = c.post(base + "/message:send", headers=H(), json=res).json()["task"]
print(r3["status"]["state"], [len(a["parts"][0]["data"].get("executions", [])) for a in r3["artifacts"][1:]])
print("cancel after done ->", c.post(f"{base}/tasks/{t['id']}:cancel", headers=H()).status_code)
