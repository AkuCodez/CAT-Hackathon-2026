# Cab Companion: Smart Operator Assistant

An in-cab assistant for construction machine operators. It reads live machine telemetry, a rear camera and a phone motion sensor, predicts task times with a trained model, raises safety and behaviour alerts, recommends training from what happens on shift, and keeps a tamper-evident incident record.

## Workflow: Sense → Understand → Act → Learn

| Stage | What happens | Code |
|---|---|---|
| Sense | Telemetry, camera person detection, phone tilt and jerks, live weather | `sensors/`, `frontend/phone.html`, `backend/main.py` (ingestion) |
| Understand | Rules, anomaly model, task-time model | `backend/rules.py`, `backend/ml.py` |
| Act | Alerts on screen and by voice, task dashboard, automatic incident records | `frontend/index.html`, `backend/ledger.py` |
| Learn | Alerts assign lessons; completed tasks update the operator's pace and feed retraining | `backend/main.py`, `ml/export_history.py` |

```
replay.py (telemetry) ──POST /telemetry──┐
vision.py (webcam+YOLO) ─POST /proximity─┤
phone.html (IMU) ─────────POST /imu──────┼──> FastAPI ──> SQLite
Open-Meteo (weather) ─────────────────── ┘     │  rules + ML + hash-chained ledger
                                               └──WS /live──> index.html (operator UI)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python ml/generate_data.py          # synthetic data calibrated to the brief's sample tables
python ml/train.py                  # trains models into ml/models/
```

## Run (each in its own terminal, from the repo root)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
python sensors/replay.py            # type: work, idle, unbelt, belt, off, on, fast, slow, quit
```

Open http://localhost:8000 for the operator UI.

Optional sensors:

```bash
pip install -r requirements-vision.txt
python sensors/vision.py --facing rear     # stand 2 m away and press c to calibrate
```

For the phone, run `ngrok http 8000` and open `https://<your-ngrok-url>/app/phone.html` on the phone. Phone motion sensors only work over HTTPS, and iPhones ask for permission when you tap **Start sensors**.

Live weather uses Open-Meteo (no key needed). It defaults to Vellore; set your site with `SITE_LAT` and `SITE_LON` environment variables. If the internet is down, choose the weather manually in the header.

## Retrain from real shifts

```bash
python ml/export_history.py         # pulls completed tasks from the backend
python ml/train.py                  # retrains using synthetic + real tasks
```

Restart the backend to load the new models.

## API contract

| Method | Path | Body / purpose |
|---|---|---|
| POST | `/telemetry` | `{ts, engine_on, idle, fuel_l, load_cycle, belt_fastened}` one row per minute |
| POST | `/proximity` | `{persons: [{distance_m, bearing_deg}]}` bearing 0 = front, 180 = rear |
| POST | `/imu` | `{tilt_deg}` every second, `{harsh: true, magnitude}` on a jerk |
| POST | `/weather` | `{mode: "auto"}` or `{mode: "manual", weather}` |
| POST | `/tasks/{id}/start`, `/tasks/{id}/complete` | task lifecycle; complete updates pace |
| POST | `/predict` | `{task_type, weather, operator_skill, machine_age, start_hour, estimated_min}` |
| POST | `/training/{module}/complete` | lesson finished |
| GET/POST | `/incidents` | list or add a record |
| GET | `/incidents/verify` | checks the hash chain |
| POST | `/incidents/{seq}/tamper` | demo only (`DEMO_MODE=1`): edits a record without re-hashing |
| POST | `/break`, `/banner/ack`, `/admin/reset` | operator and admin actions |
| GET | `/state`, `/model/info`, `/zones` | debugging and UI data |
| WS | `/live` | full state pushed every second |

Interactive docs: http://localhost:8000/docs

## Where to change things

| Change | File |
|---|---|
| Alert thresholds, new rules | `backend/rules.py` (`conditions`) |
| Stop and caution zone sizes | `backend/rules.py` (`zones`) |
| Score penalties | `backend/main.py` (`DEDUCT`) |
| Today's tasks, operator, machine | `backend/db.py` (`TODAY_TASKS`, `seed_shift`) |
| Model features | `backend/features.py` (shared by training and serving) |
| Lessons and quizzes | `frontend/index.html` (`MODULES`) |
| Voice commands | `frontend/index.html` (`handle`) |

## Team split

| Owner | Area |
|---|---|
| Backend + UI | `backend/`, `frontend/index.html` |
| ML | `ml/` |
| Vision | `sensors/vision.py` |
| Phone sensor + slides | `frontend/phone.html` |

## Honest limitations

The task and telemetry training data is synthetic, generated from the patterns in the hackathon's sample tables. Camera distance is estimated from bounding-box height and needs calibration per camera. The telemetry source is a replay script; a real deployment would replace it with the machine's telematics feed, and nothing else in the pipeline would change.
