"""Generate synthetic task and telemetry data calibrated to the hackathon sample tables.

Patterns taken from the brief:
- Experts in good weather finish on or under estimate; beginners run ~40% over.
- Rain, wind and older machines add time.
- Productive work: ~0.43 L fuel per load cycle. Long idle: ~2 L per hour, few cycles.
- Unfastened seatbelt co-occurs with long idle windows.
- Harsh events are rare in normal work, cluster during aggressive operation, and
  become moderately more likely as fatigue risk rises late in a shift.

All rows produced here are synthetic. ``sim_weather``, ``sim_operator_skill``
and ``fatigue_risk`` are simulation context, not measured CAT fields.
"""
import os
import numpy as np
import pandas as pd

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


def make_tasks(n=2000, seed=42):
    rng = np.random.default_rng(seed)
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


def _fatigue_risk(minute: int, continuous_operation: int, minutes_per_day: int) -> float:
    """A smooth 0-1 simulation signal; it is not used as an anomaly-model feature."""
    shift_progress = minute / max(minutes_per_day - 1, 1)
    late_shift = np.clip((shift_progress - 0.55) / 0.45, 0, 1)
    long_run = np.clip((continuous_operation - 180) / 240, 0, 1)
    return float(np.clip(0.65 * late_shift + 0.35 * long_run, 0, 1))


def make_telemetry(days=7, minutes_per_day=600, seed=43):
    rng = np.random.default_rng(seed)
    episodes = ["work", "short_idle", "long_idle_unbelted", "aggressive", "off"]
    base_probs = np.array([0.55, 0.20, 0.10, 0.07, 0.08], dtype=float)
    weather_values, weather_probs = ["Sunny", "Cloudy", "Rainy", "Windy"], [0.45, 0.25, 0.18, 0.12]
    skill_values, skill_probs = ["Expert", "Intermediate", "Beginner"], [0.30, 0.45, 0.25]
    rows = []
    for day in range(days):
        t = pd.Timestamp("2025-05-01 08:00") + pd.Timedelta(days=day)
        sim_weather = rng.choice(weather_values, p=weather_probs)
        sim_skill = rng.choice(skill_values, p=skill_probs)
        minute, continuous_operation = 0, 0
        previous_harsh = False
        while minute < minutes_per_day:
            fatigue = _fatigue_risk(minute, continuous_operation, minutes_per_day)
            # Fatigue changes episode likelihoods gradually rather than creating a
            # deterministic late-shift "unsafe" block.
            probs = base_probs + fatigue * np.array([-0.07, 0.035, 0.010, 0.035, -0.010])
            probs = np.clip(probs, 0.01, None)
            probs /= probs.sum()
            ep = rng.choice(episodes, p=probs)
            dur = {"work": rng.integers(20, 91), "short_idle": rng.integers(3, 12),
                   "long_idle_unbelted": rng.integers(30, 61), "aggressive": rng.integers(15, 41),
                   "off": rng.integers(10, 31)}[ep]
            for _ in range(int(dur)):
                if minute >= minutes_per_day:
                    break
                fatigue = _fatigue_risk(minute, continuous_operation, minutes_per_day)
                engine_on = ep != "off"
                fatigue_idle = ep == "work" and rng.random() < 0.018 * fatigue
                idle = ep in ("short_idle", "long_idle_unbelted") or fatigue_idle
                if not engine_on:
                    fuel, cycle = 0.0, 0
                elif idle:
                    fuel, cycle = (0.03 + rng.random() * 0.01) * (1 + 0.05 * fatigue), 0
                else:
                    cycle_probability = 0.20 * (1 - 0.18 * fatigue)
                    fuel = (0.075 + rng.random() * 0.02) * (1 + 0.06 * fatigue)
                    cycle = int(rng.random() < cycle_probability)
                if ep == "aggressive":
                    fuel *= 1.25
                belt = not (ep == "long_idle_unbelted" and rng.random() < 0.7) and not (rng.random() < 0.01)

                harsh_probability = {
                    "work": 0.004,
                    "short_idle": 0.002,
                    "long_idle_unbelted": 0.004,
                    "aggressive": 0.12,
                    "off": 0.0,
                }[ep]
                if engine_on:
                    harsh_probability += fatigue * (0.018 if ep != "aggressive" else 0.16)
                    harsh_probability *= {"Expert": 0.75, "Intermediate": 1.0, "Beginner": 1.3}[sim_skill]
                    harsh_probability *= 1.2 if sim_weather in ("Rainy", "Windy") else 1.0
                if previous_harsh and engine_on:
                    harsh_probability += 0.04  # occasional short bursts, not a fixed label
                harsh = int(rng.random() < min(harsh_probability, 0.45))
                previous_harsh = bool(harsh)

                continuous_operation = continuous_operation + 1 if engine_on else 0
                rows.append(dict(ts=(t + pd.Timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M"),
                                 machine_id="EXC001", operator_id="OP1001", engine_on=int(engine_on),
                                 idle=int(idle), fuel_l=round(fuel, 4), load_cycle=cycle,
                                 belt_fastened=int(belt), harsh_event=harsh, episode=ep,
                                 sim_weather=sim_weather, sim_operator_skill=sim_skill,
                                 continuous_operation_min=continuous_operation,
                                 fatigue_risk=round(fatigue, 3)))
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
