"""Tamper-evident incident ledger: each record's hash covers the previous record's hash."""
import hashlib
import json
import threading

from . import db

GENESIS = "0" * 64
_lock = threading.Lock()


def _canon(body: dict) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def _hash(prev: str, body_str: str) -> str:
    return hashlib.sha256((prev + body_str).encode()).hexdigest()


def add(body: dict) -> dict:
    with _lock:
        last = db.one("SELECT seq, hash FROM incidents ORDER BY seq DESC LIMIT 1")
        prev = last["hash"] if last else GENESIS
        seq = (last["seq"] if last else 0) + 1
        body = {**body, "seq": seq}
        body_str = _canon(body)
        h = _hash(prev, body_str)
        db.execute("INSERT INTO incidents(seq, ts, body, prev_hash, hash) VALUES(?,?,?,?,?)",
                   (seq, body.get("ts"), body_str, prev, h))
        return {**body, "prev_hash": prev, "hash": h}


def list_all() -> list:
    rows = db.query("SELECT seq, body, prev_hash, hash FROM incidents ORDER BY seq DESC")
    return [{**json.loads(r["body"]), "prev_hash": r["prev_hash"], "hash": r["hash"]} for r in rows]


def verify() -> dict:
    prev = GENESIS
    rows = db.query("SELECT seq, body, prev_hash, hash FROM incidents ORDER BY seq ASC")
    for r in rows:
        if r["prev_hash"] != prev or _hash(prev, r["body"]) != r["hash"]:
            return {"ok": False, "broken_at": r["seq"], "records": len(rows)}
        prev = r["hash"]
    return {"ok": True, "records": len(rows)}


def tamper(seq: int) -> bool:
    """Demo only: edit a stored record without re-hashing, to prove verify() catches it."""
    row = db.one("SELECT body FROM incidents WHERE seq = ?", (seq,))
    if not row:
        return False
    body = json.loads(row["body"])
    body["note"] = str(body.get("note", "")) + " (edited later)"
    db.execute("UPDATE incidents SET body = ? WHERE seq = ?", (_canon(body), seq))
    return True
