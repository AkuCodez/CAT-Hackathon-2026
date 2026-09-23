import os
import sqlite3
import threading

DB_PATH = os.environ.get("CAB_DB", os.path.join(os.path.dirname(__file__), "..", "cab.db"))
_lock = threading.RLock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row

SCHEMA = """
CREATE TABLE IF NOT EXISTS operators(id TEXT PRIMARY KEY, skill TEXT, bias REAL DEFAULT 1.0, safety REAL DEFAULT 100);
CREATE TABLE IF NOT EXISTS machines(id TEXT PRIMARY KEY, age INTEGER);
CREATE TABLE IF NOT EXISTS telemetry(id INTEGER PRIMARY KEY, ts TEXT, machine_id TEXT, operator_id TEXT,
  engine_on INTEGER, idle INTEGER, fuel_l REAL, load_cycle INTEGER, belt_fastened INTEGER);
CREATE TABLE IF NOT EXISTS proximity(id INTEGER PRIMARY KEY, ts TEXT, nearest_m REAL, count INTEGER, source TEXT);
CREATE TABLE IF NOT EXISTS imu_event(id INTEGER PRIMARY KEY, ts TEXT, type TEXT, magnitude REAL, tilt_deg REAL);
CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, sort INTEGER, task_type TEXT, zone TEXT, estimated_min INTEGER,
  status TEXT DEFAULT 'todo', start_clock INTEGER, predicted_min REAL, predicted_base REAL, actual_min INTEGER, weather TEXT);
CREATE TABLE IF NOT EXISTS task_history(id INTEGER PRIMARY KEY, task_id TEXT, task_type TEXT, weather TEXT,
  operator_skill TEXT, machine_age INTEGER, start_hour INTEGER, estimated_min INTEGER, actual_min INTEGER, ts TEXT);
CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY, ts TEXT, clock TEXT, key TEXT, severity TEXT,
  category TEXT, title TEXT, action TEXT);
CREATE TABLE IF NOT EXISTS training(id INTEGER PRIMARY KEY, operator_id TEXT, module TEXT, reason TEXT,
  status TEXT, clock TEXT, UNIQUE(operator_id, module));
CREATE TABLE IF NOT EXISTS incidents(seq INTEGER PRIMARY KEY, ts TEXT, body TEXT, prev_hash TEXT, hash TEXT);
"""

TODAY_TASKS = [
    ("T101", 1, "Earth Excavation", "Pit A", 60),
    ("T102", 2, "Trenching", "Drain line, north side", 45),
    ("T103", 3, "Material Loading", "Stockpile 2", 30),
    ("T104", 4, "Grading", "Access road", 35),
    ("T105", 5, "Demolition", "Old shed, east boundary", 90),
]


def execute(sql, params=()):
    with _lock:
        cur = _conn.execute(sql, params)
        _conn.commit()
        return cur.lastrowid


def query(sql, params=()):
    with _lock:
        return [dict(r) for r in _conn.execute(sql, params).fetchall()]


def one(sql, params=()):
    rows = query(sql, params)
    return rows[0] if rows else None


def seed_shift():
    with _lock:
        _conn.execute("INSERT OR IGNORE INTO operators(id, skill) VALUES('OP1001', 'Intermediate')")
        _conn.execute("INSERT OR IGNORE INTO machines(id, age) VALUES('EXC001', 4)")
        if not _conn.execute("SELECT 1 FROM tasks LIMIT 1").fetchone():
            _conn.executemany("INSERT INTO tasks(id, sort, task_type, zone, estimated_min) VALUES(?,?,?,?,?)",
                              TODAY_TASKS)
        _conn.commit()


def init():
    with _lock:
        _conn.executescript(SCHEMA)
        _conn.commit()
    seed_shift()


def reset_shift():
    """Clear shift data for a fresh run. Keeps task_history so the model can keep learning."""
    with _lock:
        for t in ("telemetry", "proximity", "imu_event", "tasks", "alerts", "training", "incidents"):
            _conn.execute(f"DELETE FROM {t}")
        _conn.execute("UPDATE operators SET safety = 100")
        _conn.commit()
    seed_shift()
