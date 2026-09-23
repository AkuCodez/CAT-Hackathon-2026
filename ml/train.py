"""Train and evaluate Cab Companion's task-time and usage-anomaly models.

Validation metrics are measured on held-out synthetic data and demonstrate the
pipeline only; they are not estimates of performance on production CAT data.
The five rows copied from the hackathon brief are excluded from model fitting
and reported only as a sample sanity check.
"""
import json
import os
import platform
import sys
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from backend.features import (  # noqa: E402
    ANOMALY_WINDOW_ROWS,
    WINDOW_FEATURES,
    featurize_tasks,
    window_features,
)

DATA = os.path.join(ROOT, "ml", "data")
MODELS = os.path.join(ROOT, "ml", "models")
os.makedirs(MODELS, exist_ok=True)

BASELINE_QUANTILES = (0.10, 0.50, 0.90)
FINAL_QUANTILES = (0.05, 0.50, 0.95)
BASELINE_PARAMS = {"max_iter": 300, "learning_rate": 0.05, "random_state": 7}
FINAL_PARAMS = {
    "max_iter": 400,
    "learning_rate": 0.04,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 40,
    "l2_regularization": 0.1,
    "random_state": 7,
}
TEST_RANDOM_STATE = 17


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit_learn": version("scikit-learn"),
        "joblib": version("joblib"),
    }


def load_task_splits():
    """Return train, untouched synthetic test, brief sanity rows, and history."""
    df = pd.read_csv(os.path.join(DATA, "tasks.csv"))
    synthetic = df[df.source == "synthetic"].copy()
    brief = df[df.source == "brief"].copy()
    train, test = train_test_split(
        synthetic, test_size=0.20, random_state=TEST_RANDOM_STATE
    )

    extra_path = os.path.join(DATA, "task_history_export.csv")
    history = pd.DataFrame(columns=df.columns)
    if os.path.exists(extra_path) and os.path.getsize(extra_path):
        history = pd.read_csv(extra_path)
        if not history.empty:
            required = set(df.columns)
            missing = required - set(history.columns)
            if missing:
                raise ValueError(f"task history is missing columns: {sorted(missing)}")
            history = history[df.columns]
            train = pd.concat([train, history], ignore_index=True)
            print(f"including {len(history)} completed tasks from the backend")
    return train, test, brief, history


def fit_task_models(df: pd.DataFrame, quantiles, params):
    X = featurize_tasks(df)
    y = df.actual_min / df.estimated_min
    return {
        q: HistGradientBoostingRegressor(
            loss="quantile", quantile=q, **params
        ).fit(X, y)
        for q in quantiles
    }


def task_predictions(models, df: pd.DataFrame, quantiles):
    X = featurize_tasks(df)
    estimate = df.estimated_min.to_numpy(dtype=float)
    lo, mid, hi = (models[q].predict(X) * estimate for q in quantiles)
    return np.minimum(lo, hi), mid, np.maximum(lo, hi)


def task_metrics(models, df: pd.DataFrame, quantiles) -> dict:
    lo, mid, hi = task_predictions(models, df, quantiles)
    actual = df.actual_min.to_numpy(dtype=float)
    return {
        "mae_model_min": round(float(np.mean(np.abs(mid - actual))), 2),
        "coverage": round(float(np.mean((actual >= lo) & (actual <= hi))), 3),
        "mean_interval_width_min": round(float(np.mean(hi - lo)), 2),
    }


def save_task_models(models):
    """Save canonical q05/q50/q95 artifacts plus legacy serving aliases.

    backend/ml.py currently loads q10/q50/q90. q10 and q90 are compatibility
    aliases of the calibrated q05 and q95 bounds so the existing backend keeps
    working until Person 1 adopts the canonical filenames.
    """
    canonical = {5: models[0.05], 50: models[0.50], 95: models[0.95]}
    aliases = {10: models[0.05], 90: models[0.95]}
    for label, model in {**canonical, **aliases}.items():
        joblib.dump(model, os.path.join(MODELS, f"task_q{label:02d}.joblib"))


def train_task_models():
    train, test, brief, history = load_task_splits()
    baseline_models = fit_task_models(train, BASELINE_QUANTILES, BASELINE_PARAMS)
    final_models = fit_task_models(train, FINAL_QUANTILES, FINAL_PARAMS)
    baseline = task_metrics(baseline_models, test, BASELINE_QUANTILES)
    final = task_metrics(final_models, test, FINAL_QUANTILES)
    save_task_models(final_models)

    blo, bmid, bhi = task_predictions(final_models, brief, FINAL_QUANTILES)
    sample = test.sort_values("task_id").iloc[np.linspace(0, len(test) - 1, 12, dtype=int)]
    slo, smid, shi = task_predictions(final_models, sample, FINAL_QUANTILES)
    representative = [
        {
            "task_id": row.task_id,
            "planned": int(row.estimated_min),
            "actual": int(row.actual_min),
            "lower": round(float(lo), 1),
            "median": round(float(mid), 1),
            "upper": round(float(hi), 1),
        }
        for row, lo, mid, hi in zip(sample.itertuples(), slo, smid, shi)
    ]
    return {
        "features": list(featurize_tasks(train).columns),
        "training_rows": int(len(train)),
        "history_rows": int(len(history)),
        "test_rows": int(len(test)),
        "test_source": "held-out synthetic rows; brief rows excluded",
        "mae_planned_min": round(
            float(np.mean(np.abs(test.estimated_min - test.actual_min))), 2
        ),
        "mae_model_min": final["mae_model_min"],
        "interval_coverage": final["coverage"],
        "mean_interval_width_min": final["mean_interval_width_min"],
        "quantiles": list(FINAL_QUANTILES),
        "hyperparameters": FINAL_PARAMS,
        "baseline": {"quantiles": list(BASELINE_QUANTILES), **baseline},
        "interval_10_90_coverage": final["coverage"],  # deprecated UI compatibility key
        "artifact_contract": {
            "canonical": ["task_q05.joblib", "task_q50.joblib", "task_q95.joblib"],
            "legacy_aliases": {
                "task_q10.joblib": "task_q05.joblib",
                "task_q90.joblib": "task_q95.joblib",
            },
        },
        "brief_evaluation_label": "sample brief sanity check; not independent validation",
        "brief_rows": [
            {
                "task_id": row.task_id,
                "estimated": int(row.estimated_min),
                "actual": int(row.actual_min),
                "model": round(float(mid), 1),
                "lower": round(float(lo), 1),
                "upper": round(float(hi), 1),
            }
            for row, lo, mid, hi in zip(brief.itertuples(), blo, bmid, bhi)
        ],
        "representative_holdout_rows": representative,
    }


def build_anomaly_windows(tel: pd.DataFrame) -> pd.DataFrame:
    """Build 30-row windows without allowing windows to cross shift days."""
    tel = tel.copy()
    tel["day"] = tel.ts.astype(str).str[:10]
    rows = []
    for day, daily in tel.groupby("day", sort=True):
        records = daily.to_dict("records")
        for start in range(0, len(records) - ANOMALY_WINDOW_ROWS + 1, 10):
            window = records[start:start + ANOMALY_WINDOW_ROWS]
            features = window_features(window)
            episodes = {r["episode"] for r in window}
            rows.append({
                **features,
                "day": day,
                "start": start,
                "unsafe": int(bool(episodes & {"long_idle_unbelted", "aggressive"})),
            })
    return pd.DataFrame(rows)


def train_anomaly_model():
    tel = pd.read_csv(os.path.join(DATA, "telemetry.csv"))
    windows = build_anomaly_windows(tel)
    days = sorted(windows.day.unique())
    test_days = set(days[-2:])
    train = windows[~windows.day.isin(test_days)]
    test = windows[windows.day.isin(test_days)]
    model = IsolationForest(n_estimators=300, contamination=0.08, random_state=7)
    model.fit(train[WINDOW_FEATURES])
    joblib.dump(model, os.path.join(MODELS, "usage_iforest.joblib"))

    flagged = model.predict(test[WINDOW_FEATURES]) == -1
    labels = test.unsafe.to_numpy(dtype=bool)
    precision = float(labels[flagged].mean()) if flagged.any() else 0.0
    recall = float(flagged[labels].mean()) if labels.any() else 0.0
    return {
        "features": WINDOW_FEATURES,
        "window_rows": ANOMALY_WINDOW_ROWS,
        "window_semantics": "30 consecutive one-minute telemetry rows",
        "train_windows": int(len(train)),
        "test_windows": int(len(test)),
        "windows": int(len(test)),  # existing frontend compatibility
        "test_days": sorted(test_days),
        "decision_threshold": 0.0,
        "flag_rate": round(float(flagged.mean()), 3),
        "precision_vs_labelled_unsafe": round(precision, 3),
        "recall_vs_labelled_unsafe": round(recall, 3),
        "evaluation_note": "held-out synthetic days; not production performance",
    }


def smoke_test_artifacts() -> dict:
    names = [
        "task_q05.joblib", "task_q10.joblib", "task_q50.joblib",
        "task_q90.joblib", "task_q95.joblib", "usage_iforest.joblib",
    ]
    result = {}
    for name in names:
        obj = joblib.load(os.path.join(MODELS, name))
        result[name] = type(obj).__name__
    return result


if __name__ == "__main__":
    meta = {
        "data_disclosure": "Synthetic data calibrated from the hackathon sample; not production CAT data.",
        "runtime_versions": runtime_versions(),
        "task_model": train_task_models(),
        "anomaly_model": train_anomaly_model(),
    }
    meta["artifact_smoke_test"] = smoke_test_artifacts()
    with open(os.path.join(MODELS, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    task, anomaly = meta["task_model"], meta["anomaly_model"]
    print(
        f"Task time: planned MAE {task['mae_planned_min']} min, "
        f"baseline model {task['baseline']['mae_model_min']} min, "
        f"final model {task['mae_model_min']} min"
    )
    print(
        f"Interval: baseline {task['baseline']['coverage']:.1%} coverage / "
        f"{task['baseline']['mean_interval_width_min']} min wide -> "
        f"final {task['interval_coverage']:.1%} / {task['mean_interval_width_min']} min wide"
    )
    print(
        f"Anomaly held-out: {anomaly['test_windows']} windows, "
        f"flag rate {anomaly['flag_rate']:.1%}, "
        f"precision {anomaly['precision_vs_labelled_unsafe']:.1%}, "
        f"recall {anomaly['recall_vs_labelled_unsafe']:.1%}"
    )
    print("Artifact reload smoke test: OK")
    for row in task["brief_rows"]:
        print(
            f"  {row['task_id']} sanity check: planned {row['estimated']}, "
            f"actual {row['actual']}, model {row['model']}"
        )
