"""Train the task-time models (10th/50th/90th percentile) and the usage anomaly model."""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from backend.features import WINDOW_FEATURES, featurize_tasks, window_features  # noqa: E402

DATA = os.path.join(ROOT, "ml", "data")
MODELS = os.path.join(ROOT, "ml", "models")
os.makedirs(MODELS, exist_ok=True)


def train_task_models():
    df = pd.read_csv(os.path.join(DATA, "tasks.csv"))
    extra = os.path.join(DATA, "task_history_export.csv")
    if os.path.exists(extra):
        df = pd.concat([df, pd.read_csv(extra)], ignore_index=True)
        print(f"including {len(pd.read_csv(extra))} completed tasks from the backend")
    X = featurize_tasks(df)
    y = df.actual_min / df.estimated_min
    Xtr, Xte, ytr, yte, dtr, dte = train_test_split(X, y, df, test_size=0.2, random_state=7)
    models = {}
    for q in (0.1, 0.5, 0.9):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                          learning_rate=0.05, random_state=7)
        m.fit(Xtr, ytr)
        models[q] = m
        joblib.dump(m, os.path.join(MODELS, f"task_q{int(q * 100)}.joblib"))
    pred = models[0.5].predict(Xte) * dte.estimated_min
    lo = models[0.1].predict(Xte) * dte.estimated_min
    hi = models[0.9].predict(Xte) * dte.estimated_min
    mae_model = float(np.mean(np.abs(pred - dte.actual_min)))
    mae_plan = float(np.mean(np.abs(dte.estimated_min - dte.actual_min)))
    coverage = float(np.mean((dte.actual_min >= lo) & (dte.actual_min <= hi)))
    brief = df[df.source == "brief"]
    bp = models[0.5].predict(featurize_tasks(brief)) * brief.estimated_min
    return {
        "features": list(X.columns),
        "test_rows": int(len(Xte)),
        "mae_planned_min": round(mae_plan, 2),
        "mae_model_min": round(mae_model, 2),
        "interval_10_90_coverage": round(coverage, 3),
        "brief_rows": [
            {"task_id": r.task_id, "estimated": int(r.estimated_min), "actual": int(r.actual_min),
             "model": round(float(p), 1)} for r, p in zip(brief.itertuples(), bp)
        ],
    }


def train_anomaly_model():
    tel = pd.read_csv(os.path.join(DATA, "telemetry.csv"))
    records = tel.to_dict("records")
    feats, labels = [], []
    for start in range(0, len(records) - 30, 10):
        w = records[start:start + 30]
        feats.append(window_features(w))
        eps = {r["episode"] for r in w}
        labels.append(int(bool(eps & {"long_idle_unbelted", "aggressive"})))
    F = pd.DataFrame(feats)[WINDOW_FEATURES]
    iso = IsolationForest(n_estimators=200, contamination=0.08, random_state=7)
    iso.fit(F)
    joblib.dump(iso, os.path.join(MODELS, "usage_iforest.joblib"))
    flagged = iso.predict(F) == -1
    labels = np.array(labels)
    precision = float(labels[flagged].mean()) if flagged.any() else 0.0
    recall = float(flagged[labels == 1].mean()) if (labels == 1).any() else 0.0
    return {"features": WINDOW_FEATURES, "windows": int(len(F)),
            "flag_rate": round(float(flagged.mean()), 3),
            "precision_vs_labelled_unsafe": round(precision, 3),
            "recall_vs_labelled_unsafe": round(recall, 3)}


if __name__ == "__main__":
    meta = {"task_model": train_task_models(), "anomaly_model": train_anomaly_model()}
    with open(os.path.join(MODELS, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    t, a = meta["task_model"], meta["anomaly_model"]
    print(f"Task time: MAE planned {t['mae_planned_min']} min -> model {t['mae_model_min']} min, "
          f"10-90% interval coverage {t['interval_10_90_coverage']:.0%}")
    print(f"Anomaly: {a['windows']} windows, flag rate {a['flag_rate']:.0%}, "
          f"precision {a['precision_vs_labelled_unsafe']:.0%}, recall {a['recall_vs_labelled_unsafe']:.0%}")
    for r in t["brief_rows"]:
        print(f"  {r['task_id']}: planned {r['estimated']}, actual {r['actual']}, model {r['model']}")
