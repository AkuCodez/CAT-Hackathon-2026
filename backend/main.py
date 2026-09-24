"""Cab Companion backend. Run from repo root: uvicorn backend.main:app --reload --host 0.0.0.0"""
import asyncio
import collections
import datetime as dt
import json
import os
import time
import urllib.request
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, ledger
from .features import window_features
from .ml import ML
from .rules import Rules, zones

OP, MACHINE = "OP1001", "EXC001"
DEMO_MODE = os.environ.get("DEMO_MODE", "1") == "1"
SITE_LAT = float(os.environ.get("SITE_LAT", "12.9692"))
SITE_LON = float(os.environ.get("SITE_LON", "79.1559"))
FRONTEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
DEDUCT = {"belt": {"critical": 8, "warning": 3}, "prox": {"critical": 10, "warning": 2},
          "tilt": {"critical": 10, "warning": 3}, "harsh": {"warning": 2}}


class Live:
    def __init__(self):
        self.hist = collections.deque(maxlen=240)
        self.engine_on, self.idle, self.belt = False, False, True
        self.clock, self.last_tel = 480, 0.0
        self.persons, self.cam_t = [], 0.0
        self.tilt, self.phone_t = None, 0.0
        self.harsh = collections.deque(maxlen=500)
        self.harsh_pending = 0
        self.since_break = 0
        self.weather, self.weather_mode, self.weather_detail, self.weather_t = "Sunny", "auto", "Not fetched yet", 0.0
        self.banner = None
        self.anomaly = {"score": None, "flagged": False}
        self.live_alerts = []


L, R, M = Live(), Rules(), ML()
clients: set = set()


def fmt(minutes: int) -> str:
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def window(n: int = 30) -> dict:
    rows = list(L.hist)[-n:]
    f = window_features(rows)
    eng = sum(r["engine_on"] for r in rows)
    fuel = sum(r["fuel_l"] for r in rows)
    return {"len": len(rows), "engine_min": eng, "ratio": f["idle_share"], "fpc": f["fuel_per_cycle"],
            "cycles": f["cycles"], "unbelted": f["unbelted_min"], "fuel": round(fuel, 2)}


def harsh_count(seconds: float = 30) -> int:
    cutoff = time.time() - seconds
    return sum(1 for t in L.harsh if t >= cutoff)


def nearest():
    if time.time() - L.cam_t > 3 or not L.persons:
        return None
    return min(p["distance_m"] for p in L.persons)


def tilt():
    return L.tilt if time.time() - L.phone_t <= 5 else None


def operator():
    return db.one("SELECT * FROM operators WHERE id = ?", (OP,))


def machine():
    return db.one("SELECT * FROM machines WHERE id = ?", (MACHINE,))


# ---------- Alerts, training, incidents ----------
def telemetry_snapshot() -> dict:
    w = window(10)
    return {"window_min": w["len"], "engine_min": w["engine_min"], "idle_share": round(w["ratio"], 2),
            "fuel_l": w["fuel"], "load_cycles": w["cycles"], "unbelted_min": w["unbelted"],
            "nearest_person_m": nearest(), "tilt_deg": tilt(), "weather": L.weather}


def assign_training(module: str, reason: str):
    exists = db.one("SELECT 1 FROM training WHERE operator_id = ? AND module = ?", (OP, module))
    if not exists:
        db.execute("INSERT INTO training(operator_id, module, reason, status, clock) VALUES(?,?,?,?,?)",
                   (OP, module, reason, "assigned", fmt(L.clock)))


def raise_alert(a: dict):
    db.execute("INSERT INTO alerts(ts, clock, key, severity, category, title, action) VALUES(?,?,?,?,?,?,?)",
               (now_iso(), fmt(L.clock), a["key"], a["severity"], a["category"], a["title"], a["action"]))
    d = DEDUCT.get(a["key"], {}).get(a["severity"], 0)
    if d:
        db.execute("UPDATE operators SET safety = MAX(0, safety - ?) WHERE id = ?", (d, OP))
    if a.get("module"):
        assign_training(a["module"], a["title"])
    if a["severity"] == "critical":
        L.banner = {"key": a["key"], "title": a["title"], "action": a["action"], "id": time.time()}
        ledger.add({"ts": now_iso(), "clock": fmt(L.clock), "type": f"{a['category']} alert",
                    "severity": "critical", "note": a["title"], "source": "Automatic",
                    "snapshot": telemetry_snapshot()})


def evaluate():
    w = window(30)
    ctx = {"telemetry_live": time.time() - L.last_tel < 15, "engine_on": L.engine_on, "idle": L.idle,
           "belt": L.belt, "nearest": nearest(), "tilt_deg": tilt(), "weather": L.weather, "win": w,
           "since_break": L.since_break, "harsh_count": harsh_count(), "anomaly_flagged": L.anomaly["flagged"]}
    new, live = R.evaluate(ctx)
    for a in new:
        raise_alert(a)
    L.live_alerts = live
    if L.banner and L.banner["key"] not in R.active:
        L.banner = None


# ---------- Tasks ----------
def task_views():
    op, m = operator(), machine()
    rows = db.query("SELECT * FROM tasks ORDER BY sort")
    cursor, out = L.clock, []
    for t in rows:
        v = dict(t)
        if t["status"] == "done":
            v["scheduled"] = fmt(t["start_clock"])
        elif t["status"] == "active":
            v["scheduled"] = fmt(t["start_clock"])
            v["elapsed_min"] = L.clock - t["start_clock"]
            cursor = max(L.clock, t["start_clock"] + int(t["predicted_min"]))
        else:
            p = M.predict_task(t["task_type"], L.weather, op["skill"], m["age"], cursor // 60,
                               t["estimated_min"], op["bias"])
            v["prediction"], v["scheduled"] = p, fmt(cursor)
            cursor += int(round(p["mid"]))
        out.append(v)
    return out, fmt(cursor)


# ---------- Live snapshot + websocket ----------
def idle_series(points: int = 120) -> list:
    h = list(L.hist)
    out = []
    for i in range(max(0, len(h) - points), len(h)):
        sl = h[max(0, i - 29):i + 1]
        eng = sum(x["engine_on"] for x in sl)
        out.append(round(sum(x["engine_on"] and x["idle"] for x in sl) / eng, 3) if eng else 0)
    return out


def snapshot() -> dict:
    op = operator()
    tot = db.one("SELECT SUM(engine_on) e, SUM(engine_on * idle) i, SUM(load_cycle) c, SUM(fuel_l) f FROM telemetry")
    e, i = tot["e"] or 0, tot["i"] or 0
    efficiency = round(max(0, min(100, 100 - max(0, (i / e if e else 0) - 0.15) * 200)))
    tasks, finish = task_views()
    now = time.time()
    return {
        "clock": fmt(L.clock), "clock_min": L.clock,
        "machine": {"id": MACHINE, "age": machine()["age"], "engine_on": L.engine_on, "idle": L.idle, "belt": L.belt},
        "operator": {"id": OP, "skill": op["skill"], "bias": round(op["bias"], 3), "safety": round(op["safety"])},
        "efficiency": efficiency, "shift_cycles": tot["c"] or 0, "shift_fuel_l": round(tot["f"] or 0, 2),
        "persons": L.persons if now - L.cam_t <= 3 else [], "nearest": nearest(), "zones": zones(L.weather),
        "tilt_deg": tilt(), "harsh_count": harsh_count(), "since_break": L.since_break,
        "weather": {"value": L.weather, "mode": L.weather_mode, "detail": L.weather_detail},
        "window": window(30), "anomaly": L.anomaly, "idle_series": idle_series(),
        "live_alerts": L.live_alerts, "banner": L.banner,
        "recent_alerts": db.query("SELECT * FROM alerts ORDER BY id DESC LIMIT 25"),
        "training": db.query("SELECT module, reason, status, clock FROM training WHERE operator_id = ? ORDER BY id", (OP,)),
        "tasks": tasks, "expected_finish": finish,
        "incident_count": db.one("SELECT COUNT(*) n FROM incidents")["n"],
        "sources": {"telemetry": now - L.last_tel < 15, "camera": now - L.cam_t < 3,
                    "phone": now - L.phone_t < 5, "weather_auto": L.weather_mode == "auto" and L.weather_t > 0},
        "model": "quantile-gbm" if M.ok else "fallback-formula",
    }


async def broadcast():
    if not clients:
        return
    msg = json.dumps(snapshot(), default=str)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


WEATHER_CODES = {"Rainy": set(range(51, 68)) | set(range(80, 83)) | set(range(95, 100)),
                 "Cloudy": {2, 3, 45, 48}}


def fetch_weather():
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={SITE_LAT}&longitude={SITE_LON}"
           f"&current=weather_code,wind_speed_10m,temperature_2m")
    with urllib.request.urlopen(url, timeout=6) as r:
        cur = json.load(r)["current"]
    code, wind = cur["weather_code"], cur["wind_speed_10m"]
    value = "Rainy" if code in WEATHER_CODES["Rainy"] else "Windy" if wind >= 30 else \
        "Cloudy" if code in WEATHER_CODES["Cloudy"] else "Sunny"
    return value, f"Live: {cur['temperature_2m']}°C, wind {wind} km/h"


async def weather_loop():
    while True:
        if L.weather_mode == "auto":
            try:
                L.weather, L.weather_detail = await asyncio.to_thread(fetch_weather)
                L.weather_t = time.time()
            except Exception as e:
                L.weather_detail = f"Live weather unavailable ({type(e).__name__}); using {L.weather}"
        await asyncio.sleep(600)


async def tick_loop():
    while True:
        try:
            evaluate()
            await broadcast()
        except Exception as e:
            print("[tick] error:", e)
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app):
    db.init()
    t1, t2 = asyncio.create_task(tick_loop()), asyncio.create_task(weather_loop())
    yield
    t1.cancel()
    t2.cancel()


app = FastAPI(title="Cab Companion API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- Ingestion ----------
class Telemetry(BaseModel):
    machine_id: str = MACHINE
    operator_id: str = OP
    ts: Optional[str] = None
    engine_on: bool
    idle: bool
    fuel_l: float
    load_cycle: int = 0
    belt_fastened: bool


class Person(BaseModel):
    distance_m: float
    bearing_deg: float = 180


class Proximity(BaseModel):
    persons: List[Person] = []
    source: str = "camera"


class Imu(BaseModel):
    tilt_deg: Optional[float] = None
    harsh: bool = False
    magnitude: float = 0


@app.post("/telemetry")
def post_telemetry(t: Telemetry):
    if t.ts and len(t.ts) >= 16:
        try:
            L.clock = int(t.ts[11:13]) * 60 + int(t.ts[14:16])
        except ValueError:
            L.clock += 1
    else:
        L.clock += 1
    row = t.model_dump()
    harsh_in_row, L.harsh_pending = L.harsh_pending, 0
    L.hist.append({"engine_on": int(t.engine_on), "idle": int(t.idle), "fuel_l": t.fuel_l,
                   "load_cycle": t.load_cycle, "belt_fastened": int(t.belt_fastened), "harsh_event": harsh_in_row})
    L.engine_on, L.idle, L.belt, L.last_tel = t.engine_on, t.idle and t.engine_on, t.belt_fastened, time.time()
    if t.engine_on:
        L.since_break += 1
    db.execute("INSERT INTO telemetry(ts, machine_id, operator_id, engine_on, idle, fuel_l, load_cycle, belt_fastened) "
               "VALUES(?,?,?,?,?,?,?,?)", (row["ts"] or now_iso(), t.machine_id, t.operator_id, int(t.engine_on),
                                           int(t.idle), t.fuel_l, t.load_cycle, int(t.belt_fastened)))
    if len(L.hist) >= 20:
        feats = window_features(list(L.hist)[-30:])
        L.anomaly = M.anomaly(feats)
    return {"ok": True, "clock": fmt(L.clock)}


@app.post("/proximity")
def post_proximity(p: Proximity):
    L.persons = [x.model_dump() for x in p.persons]
    L.cam_t = time.time()
    n = min((x.distance_m for x in p.persons), default=None)
    if n is not None:
        db.execute("INSERT INTO proximity(ts, nearest_m, count, source) VALUES(?,?,?,?)",
                   (now_iso(), n, len(p.persons), p.source))
    return {"ok": True, "zones": zones(L.weather)}


@app.post("/imu")
def post_imu(i: Imu):
    L.phone_t = time.time()
    if i.tilt_deg is not None:
        L.tilt = i.tilt_deg
    if i.harsh:
        L.harsh.append(time.time())
        L.harsh_pending += 1
        db.execute("INSERT INTO imu_event(ts, type, magnitude, tilt_deg) VALUES(?,?,?,?)",
                   (now_iso(), "harsh", i.magnitude, i.tilt_deg))
    return {"ok": True}


@app.get("/zones")
def get_zones():
    return zones(L.weather)


# ---------- Operator actions ----------
class WeatherIn(BaseModel):
    mode: str = "manual"
    weather: Optional[str] = None


@app.post("/weather")
async def set_weather(w: WeatherIn):
    if w.mode == "auto":
        L.weather_mode = "auto"
        try:
            L.weather, L.weather_detail = await asyncio.to_thread(fetch_weather)
            L.weather_t = time.time()
        except Exception as e:
            L.weather_detail = f"Live weather unavailable ({type(e).__name__}); using {L.weather}"
    elif w.weather in ("Sunny", "Cloudy", "Rainy", "Windy"):
        L.weather_mode, L.weather, L.weather_detail = "manual", w.weather, "Set by operator"
    else:
        raise HTTPException(400, "weather must be Sunny, Cloudy, Rainy or Windy")
    return {"weather": L.weather, "mode": L.weather_mode, "detail": L.weather_detail}


@app.post("/break")
def take_break():
    L.since_break = 0
    return {"ok": True}


@app.post("/banner/ack")
def ack_banner():
    L.banner = None
    return {"ok": True}


@app.post("/tasks/{task_id}/start")
def start_task(task_id: str):
    if db.one("SELECT 1 FROM tasks WHERE status = 'active'"):
        raise HTTPException(409, "Finish the current task first")
    t = db.one("SELECT * FROM tasks WHERE id = ? AND status = 'todo'", (task_id,))
    if not t:
        raise HTTPException(404, "Task not found or already started")
    op, m = operator(), machine()
    p = M.predict_task(t["task_type"], L.weather, op["skill"], m["age"], L.clock // 60, t["estimated_min"], op["bias"])
    db.execute("UPDATE tasks SET status='active', start_clock=?, predicted_min=?, predicted_base=?, weather=? WHERE id=?",
               (L.clock, p["mid"], p["base_mid"], L.weather, task_id))
    return {"ok": True, "prediction": p}


@app.post("/tasks/{task_id}/complete")
def complete_task(task_id: str):
    t = db.one("SELECT * FROM tasks WHERE id = ? AND status = 'active'", (task_id,))
    if not t:
        raise HTTPException(404, "Task is not in progress")
    actual = max(1, L.clock - t["start_clock"])
    op, m = operator(), machine()
    ratio = actual / max(t["predicted_base"], 1)
    bias = max(0.7, min(1.5, 0.7 * op["bias"] + 0.3 * ratio))
    db.execute("UPDATE tasks SET status='done', actual_min=? WHERE id=?", (actual, task_id))
    db.execute("UPDATE operators SET bias=? WHERE id=?", (bias, OP))
    db.execute("INSERT INTO task_history(task_id, task_type, weather, operator_skill, machine_age, start_hour, "
               "estimated_min, actual_min, ts) VALUES(?,?,?,?,?,?,?,?,?)",
               (task_id, t["task_type"], t["weather"], op["skill"], m["age"], t["start_clock"] // 60,
                t["estimated_min"], actual, now_iso()))
    return {"ok": True, "actual_min": actual, "new_bias": round(bias, 3)}


@app.post("/training/{module}/complete")
def complete_training(module: str):
    db.execute("INSERT OR IGNORE INTO training(operator_id, module, reason, status, clock) VALUES(?,?,?,?,?)",
               (OP, module, "Self-selected", "assigned", fmt(L.clock)))
    row = db.one("SELECT status FROM training WHERE operator_id=? AND module=?", (OP, module))
    if row and row["status"] != "done":
        db.execute("UPDATE training SET status='done' WHERE operator_id=? AND module=?", (OP, module))
        db.execute("UPDATE operators SET safety = MIN(100, safety + 3) WHERE id = ?", (OP,))
    return {"ok": True}


class PredictIn(BaseModel):
    task_type: Literal["Earth Excavation", "Trenching", "Material Loading", "Grading", "Demolition"]
    weather: Literal["Sunny", "Cloudy", "Rainy", "Windy"]
    operator_skill: Literal["Expert", "Intermediate", "Beginner"]
    machine_age: int = 3
    start_hour: int = 8
    estimated_min: float = 60.0


@app.post("/predict")
def predict(p: PredictIn):
    return M.predict_task(**p.model_dump())


@app.get("/model/info")
def model_info():
    return {"loaded": M.ok, "meta": M.meta}


# ---------- Incidents ----------
class IncidentIn(BaseModel):
    type: str
    severity: str = "warning"
    note: str
    source: str = "Operator"


@app.get("/incidents")
def get_incidents():
    return ledger.list_all()


@app.post("/incidents")
def post_incident(i: IncidentIn):
    return ledger.add({"ts": now_iso(), "clock": fmt(L.clock), **i.model_dump(), "snapshot": telemetry_snapshot()})


@app.get("/incidents/verify")
def verify_incidents():
    return ledger.verify()


@app.post("/incidents/{seq}/tamper")
def tamper_incident(seq: int):
    if not DEMO_MODE:
        raise HTTPException(403, "Disabled outside demo mode")
    if not ledger.tamper(seq):
        raise HTTPException(404, "No such record")
    return {"ok": True}


# ---------- Admin ----------
@app.post("/admin/reset")
def reset():
    db.reset_shift()
    global L, R
    weather = (L.weather, L.weather_mode, L.weather_detail, L.weather_t)
    L, R = Live(), Rules()
    L.weather, L.weather_mode, L.weather_detail, L.weather_t = weather
    return {"ok": True}


@app.get("/admin/export-history")
def export_history():
    """Completed real tasks, in the training CSV format, to feed back into ml/train.py."""
    return db.query("SELECT task_id, task_type, weather, operator_skill, machine_age, start_hour, estimated_min, "
                    "actual_min, 'backend' AS source FROM task_history")


@app.get("/state")
def state():
    return snapshot()


@app.websocket("/live")
async def live(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        await ws.send_text(json.dumps(snapshot(), default=str))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


@app.get("/")
def root():
    return RedirectResponse("/app/")


app.mount("/app", StaticFiles(directory=FRONTEND, html=True), name="app")
