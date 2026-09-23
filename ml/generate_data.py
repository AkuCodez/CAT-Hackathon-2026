"""Generate synthetic task and telemetry data calibrated to the hackathon sample tables.

Patterns taken from the brief:
- Experts in good weather finish on or under estimate; beginners run ~40% over.
- Rain, wind and older machines add time.
- Productive work: ~0.43 L fuel per load cycle. Long idle: ~2 L per hour, few cycles.
- Unfastened seatbelt co-occurs with long idle windows.
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

TYPES = {"Earth Excavation": (40, 90), "Trenching": (30, 70), "Material Loading": (20, 50),
         "Grading": (25, 60), "Demolition": (60, 150)}
SKILL = {"Expert": 0.95, "Intermediate": 1.05, "Beginner": 1.30}
WEATHER = {"Sunny": 1.00, "Cloudy": 1.05, "Rainy": 1.10, "Windy": 1.08}

BRIEF_TASKS = [
    ("T001", "Earth Excavation", "Sunny", "Expert", 2, 60, 58),
    ("T002", "Trenching", "Rainy", "Intermediate", 4, 45, 52),
    ("T003", "Material Loading", "Cloudy", "Beginner", 3, 30, 42),
    ("T004", "Grading", "Sunny", "Expert", 5, 35, 33),
    ("T005", "Demolition", "Windy", "Intermediate", 6, 90, 105),
]


def make_tasks(n=2000):
    rows = []
    for i in range(n):
        t = rng.choice(list(TYPES))
        lo, hi = TYPES[t]
        planned = int(max(lo, (rng.integers(lo, hi + 1) // 5) * 5))
        skill = rng.choice(list(SKILL), p=[0.3, 0.45, 0.25])
        weather = rng.choice(list(WEATHER), p=[0.45, 0.25, 0.18, 0.12])
        age = int(rng.integers(0, 11))
        hour = int(rng.integers(7, 18))
        f = SKILL[skill] * WEATHER[weather] * (1 + 0.015 * (age - 3))
        if weather == "Rainy" and t in ("Trenching", "Earth Excavation"):
            f *= 1.05
        if weather == "Windy" and t == "Demolition":
            f *= 1.04
        if hour >= 15:
            f *= 1.03
        actual = max(5, round(planned * f * rng.lognormal(0, 0.06)))
        rows.append(dict(task_id=f"S{i:05d}", task_type=t, weather=weather, operator_skill=skill,
                         machine_age=age, start_hour=hour, estimated_min=planned, actual_min=actual,
                         source="synthetic"))
    brief = [dict(task_id=r[0], task_type=r[1], weather=r[2], operator_skill=r[3], machine_age=r[4],
                  start_hour=9, estimated_min=r[5], actual_min=r[6], source="brief") for r in BRIEF_TASKS]
    return pd.DataFrame(brief + rows)


def make_telemetry(days=7, minutes_per_day=600):
    episodes = ["work", "short_idle", "long_idle_unbelted", "aggressive", "off"]
    probs = [0.55, 0.20, 0.10, 0.07, 0.08]
    rows = []
    for day in range(days):
        t = pd.Timestamp("2025-05-01 08:00") + pd.Timedelta(days=day)
        minute = 0
        while minute < minutes_per_day:
            ep = rng.choice(episodes, p=probs)
            dur = {"work": rng.integers(20, 91), "short_idle": rng.integers(3, 12),
                   "long_idle_unbelted": rng.integers(30, 61), "aggressive": rng.integers(15, 41),
                   "off": rng.integers(10, 31)}[ep]
            for _ in range(int(dur)):
                if minute >= minutes_per_day:
                    break
                engine_on = ep != "off"
                idle = ep in ("short_idle", "long_idle_unbelted")
                if not engine_on:
                    fuel, cycle = 0.0, 0
                elif idle:
                    fuel, cycle = 0.03 + rng.random() * 0.01, 0
                else:
                    fuel, cycle = 0.075 + rng.random() * 0.02, int(rng.random() < 0.2)
                if ep == "aggressive":
                    fuel *= 1.25
                belt = not (ep == "long_idle_unbelted" and rng.random() < 0.7) and not (rng.random() < 0.01)
                harsh = int(ep == "aggressive" and rng.random() < 0.25) or int(ep == "work" and rng.random() < 0.01)
                rows.append(dict(ts=(t + pd.Timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M"),
                                 machine_id="EXC001", operator_id="OP1001", engine_on=int(engine_on),
                                 idle=int(idle), fuel_l=round(fuel, 4), load_cycle=cycle,
                                 belt_fastened=int(belt), harsh_event=harsh, episode=ep))
                minute += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tasks = make_tasks()
    tel = make_telemetry()
    tasks.to_csv(os.path.join(OUT, "tasks.csv"), index=False)
    tel.to_csv(os.path.join(OUT, "telemetry.csv"), index=False)
    print(f"tasks.csv: {len(tasks)} rows | telemetry.csv: {len(tel)} rows")
    work = tel[(tel.episode == "work")]
    print(f"check: productive fuel per cycle = {work.fuel_l.sum() / max(work.load_cycle.sum(), 1):.2f} L")
