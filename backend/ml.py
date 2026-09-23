"""Loads trained models. Falls back to a calibrated formula if models are missing."""
import json
import os

import pandas as pd

from .features import WINDOW_FEATURES, featurize_tasks

MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "models"))
QUANTILES = (5, 50, 95)
FALLBACK = {"skill": {"Expert": 0.95, "Intermediate": 1.05, "Beginner": 1.30},
            "weather": {"Sunny": 1.00, "Cloudy": 1.05, "Rainy": 1.10, "Windy": 1.08}}
BASELINE = {"weather": "Sunny", "operator_skill": "Expert", "machine_age": 3, "start_hour": 10}
LABELS = {"weather": "{v} weather", "operator_skill": "{v} operator",
          "machine_age": "Machine age {v} yrs", "start_hour": "Starting at {v}:00"}


class ML:
    def __init__(self):
        self.meta, self.q, self.iso = {}, {}, None
        try:
            import joblib
            for q in QUANTILES:
                self.q[q] = joblib.load(os.path.join(MODELS, f"task_q{q}.joblib"))
            self.iso = joblib.load(os.path.join(MODELS, "usage_iforest.joblib"))
            with open(os.path.join(MODELS, "meta.json")) as f:
                self.meta = json.load(f)
            self.ok = True
        except Exception as e:  # models not trained yet
            print(f"[ml] models not loaded ({e}); using fallback formula. Run ml/train.py")
            self.ok = False

    def _ratios(self, rows):
        df = pd.DataFrame(rows)
        if not self.ok:
            out = []
            for r in rows:
                f = FALLBACK["skill"][r["operator_skill"]] * FALLBACK["weather"][r["weather"]]
                f *= 1 + 0.015 * (r["machine_age"] - 3)
                out.append((f * 0.92, f, f * 1.08))
            return out
        X = featurize_tasks(df).reindex(columns=self.meta["task_model"]["features"], fill_value=0)
        lo, mid, hi = (self.q[q].predict(X) for q in QUANTILES)
        return [tuple(sorted((a, b, c))) for a, b, c in zip(lo, mid, hi)]

    def predict_task(self, task_type, weather, operator_skill, machine_age, start_hour, estimated_min, bias=1.0):
        base = dict(task_type=task_type, weather=weather, operator_skill=operator_skill,
                    machine_age=int(machine_age), start_hour=int(start_hour), estimated_min=float(estimated_min))
        variants = [(k, v) for k, v in BASELINE.items() if base[k] != v]
        rows = [base] + [{**base, k: v} for k, v in variants]
        ratios = self._ratios(rows)
        lo, mid, hi = ratios[0]
        est = float(estimated_min)
        factors = []
        for (k, _), r in zip(variants, ratios[1:]):
            delta = est * (mid - r[1]) * bias
            if abs(delta) >= 0.1:
                factors.append({"label": LABELS[k].format(v=base[k]), "delta_min": round(delta, 1)})
        if abs(bias - 1) > 0.005:
            factors.append({"label": "Your recent pace", "delta_min": round(est * mid * (bias - 1), 1)})
        return {"low": round(est * lo * bias, 1), "mid": round(est * mid * bias, 1),
                "high": round(est * hi * bias, 1), "base_mid": round(est * mid, 1),
                "factors": factors, "model": "quantile-gbm" if self.ok else "fallback-formula"}

    def anomaly(self, features: dict) -> dict:
        if not self.iso:
            return {"score": None, "flagged": False}
        X = pd.DataFrame([features])[WINDOW_FEATURES]
        score = float(self.iso.decision_function(X)[0])
        return {"score": round(score, 3), "flagged": bool(self.iso.predict(X)[0] == -1)}
