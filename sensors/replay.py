"""Telemetry source for the backend.

Replace this script with a real telematics feed later; the backend API stays the same.

  python sensors/replay.py                      # live machine, control it from the terminal
  python sensors/replay.py --mode csv            # replay ml/data/telemetry.csv
  python sensors/replay.py --events "30:idle,45:unbelt,70:belt,75:work"

Terminal commands while running: work, idle, unbelt, belt, off, on, fast, slow, quit
"""
import argparse
import csv
import datetime as dt
import os
import random
import sys
import threading
import time

import requests

p = argparse.ArgumentParser()
p.add_argument("--api", default="http://localhost:8000")
p.add_argument("--mode", choices=["sim", "csv"], default="sim")
p.add_argument("--file", default=os.path.join(os.path.dirname(__file__), "..", "ml", "data", "telemetry.csv"))
p.add_argument("--interval", type=float, default=1.0, help="real seconds per telemetry minute")
p.add_argument("--start", default="08:00")
p.add_argument("--events", default="", help='scripted changes, e.g. "30:idle,45:unbelt"')
p.add_argument("--minutes", type=int, default=0, help="stop after N minutes (0 = run forever)")
args = p.parse_args()

state = {"engine_on": True, "idle": False, "belt": True, "interval": args.interval, "quit": False}
events = {}
for item in filter(None, args.events.split(",")):
    m, cmd = item.split(":")
    events.setdefault(int(m), []).append(cmd.strip())


def apply(cmd: str):
    cmd = cmd.strip().lower()
    if cmd == "work":
        state.update(idle=False, engine_on=True)
    elif cmd == "idle":
        state.update(idle=True, engine_on=True)
    elif cmd == "unbelt":
        state["belt"] = False
    elif cmd == "belt":
        state["belt"] = True
    elif cmd == "off":
        state.update(engine_on=False, idle=False)
    elif cmd == "on":
        state["engine_on"] = True
    elif cmd == "fast":
        state["interval"] = max(0.05, state["interval"] / 4)
    elif cmd == "slow":
        state["interval"] = args.interval
    elif cmd == "quit":
        state["quit"] = True
    elif cmd:
        print("commands: work, idle, unbelt, belt, off, on, fast, slow, quit")


def keyboard():
    for line in sys.stdin:
        apply(line)
        print(f"  -> engine {'on' if state['engine_on'] else 'off'}, {'idle' if state['idle'] else 'working'}, "
              f"belt {'on' if state['belt'] else 'OFF'}")


def sim_row():
    if not state["engine_on"]:
        fuel, cycle = 0.0, 0
    elif state["idle"]:
        fuel, cycle = 0.03 + random.random() * 0.01, 0
    else:
        fuel, cycle = 0.075 + random.random() * 0.02, int(random.random() < 0.2)
    return {"engine_on": state["engine_on"], "idle": state["idle"], "fuel_l": round(fuel, 4),
            "load_cycle": cycle, "belt_fastened": state["belt"]}


def csv_rows():
    with open(args.file) as f:
        for r in csv.DictReader(f):
            yield {"engine_on": r["engine_on"] == "1", "idle": r["idle"] == "1", "fuel_l": float(r["fuel_l"]),
                   "load_cycle": int(r["load_cycle"]), "belt_fastened": r["belt_fastened"] == "1"}


def main():
    if sys.stdin.isatty():
        threading.Thread(target=keyboard, daemon=True).start()
    start = dt.datetime.combine(dt.date.today(), dt.datetime.strptime(args.start, "%H:%M").time())
    source = csv_rows() if args.mode == "csv" else None
    minute = 0
    print(f"Sending telemetry to {args.api} ({args.mode} mode). Type a command and press Enter.")
    while not state["quit"]:
        for cmd in events.get(minute, []):
            apply(cmd)
            print(f"[event @ minute {minute}] {cmd}")
        row = next(source, None) if source else sim_row()
        if row is None:
            print("CSV finished.")
            break
        row["ts"] = (start + dt.timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M")
        try:
            requests.post(f"{args.api}/telemetry", json=row, timeout=3)
        except requests.RequestException as e:
            print("backend not reachable:", e)
        if minute % 10 == 0:
            print(f"{row['ts'][11:]}  engine={'on' if row['engine_on'] else 'off'} idle={row['idle']} "
                  f"belt={row['belt_fastened']} fuel={row['fuel_l']}")
        minute += 1
        if args.minutes and minute >= args.minutes:
            break
        time.sleep(state["interval"])


if __name__ == "__main__":
    main()
