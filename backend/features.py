import pandas as pd

CATS = {
    "task_type": ["Earth Excavation", "Trenching", "Material Loading", "Grading", "Demolition"],
    "weather": ["Sunny", "Cloudy", "Rainy", "Windy"],
    "operator_skill": ["Expert", "Intermediate", "Beginner"],
}
NUMERIC = ["machine_age", "start_hour", "estimated_min"]
WINDOW_FEATURES = ["idle_share", "fuel_per_cycle", "unbelted_min", "cycles", "harsh_count"]
ANOMALY_WINDOW_ROWS = 30


def featurize_tasks(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC].astype(float).copy()
    for col, values in CATS.items():
        for v in values:
            X[f"{col}={v}"] = (df[col] == v).astype(int)
    return X


def window_features(rows) -> dict:
    """Aggregate consecutive telemetry samples for anomaly inference.

    The project contract is one telemetry row per simulated/operational minute,
    so ``harsh_count`` means events observed in the same 30-row window as all
    other features. The feature names and order are shared with serving.
    """
    eng = sum(int(r["engine_on"]) for r in rows)
    idle = sum(int(r["engine_on"]) and int(r["idle"]) for r in rows)
    fuel = sum(float(r["fuel_l"]) for r in rows)
    cycles = sum(int(r["load_cycle"]) for r in rows)
    unbelted = sum(int(r["engine_on"]) and not int(r["belt_fastened"]) for r in rows)
    harsh = sum(int(r.get("harsh_event", 0)) for r in rows)
    return {
        "idle_share": idle / eng if eng else 0.0,
        "fuel_per_cycle": fuel / max(cycles, 1),
        "unbelted_min": unbelted,
        "cycles": cycles,
        "harsh_count": harsh,
    }
