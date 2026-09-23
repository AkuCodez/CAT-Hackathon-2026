"""Pull completed real tasks from the backend so train.py can learn from them."""
import csv
import os
import sys

import requests

api = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
response = requests.get(f"{api}/admin/export-history", timeout=5)
response.raise_for_status()
rows = response.json()
out = os.path.join(os.path.dirname(__file__), "data", "task_history_export.csv")
if rows:
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
elif os.path.exists(out):
    # An empty export must not silently leave stale history in the next retrain.
    os.remove(out)
print(f"exported {len(rows)} completed tasks to {out}. Now run: python ml/train.py")
