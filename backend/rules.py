"""Safety and behaviour rules. Each rule fires once when it becomes true or escalates."""
RANK = {"info": 0, "warning": 1, "critical": 2}


def zones(weather: str) -> dict:
    red = 5 + {"Rainy": 2, "Windy": 1}.get(weather, 0)
    return {"red": red, "amber": red + 5}


def conditions(c: dict) -> list:
    z, w, d = zones(c["weather"]), c["win"], c["nearest"]
    live = c["telemetry_live"]
    out = []

    def add(key, on, sev, title, action, module, cat):
        out.append(dict(key=key, on=bool(on), severity=sev, title=title, action=action, module=module, category=cat))

    add("belt", live and c["engine_on"] and not c["belt"], "warning" if c["idle"] else "critical",
        "Seatbelt unfastened with engine running" if c["idle"] else "Seatbelt unfastened while the machine is working",
        "Fasten your seatbelt before moving the machine.", "seatbelt", "Safety")
    if d is not None:
        add("prox", d < z["amber"], "critical" if d < z["red"] else "warning",
            f"Person inside the {z['red']} m stop zone" if d < z["red"] else f"Person approaching, {d:.1f} m away",
            "Stop swing and travel. Wait until the ground crew is clear." if d < z["red"]
            else "Slow down and confirm eye contact with the worker.", "crew", "Safety")
    else:
        add("prox", False, "info", "", "", None, "Safety")
    tilt = c["tilt_deg"]
    add("tilt", tilt is not None and abs(tilt) >= 10, "critical" if tilt is not None and abs(tilt) >= 15 else "warning",
        f"Machine tilted {abs(tilt or 0):.0f} degrees",
        "Stop, lower the bucket and move to level ground." if tilt is not None and abs(tilt) >= 15
        else "Slope is getting steep. Travel slowly and keep the bucket low.", "slope", "Safety")
    add("harsh", c["harsh_count"] >= 3, "warning", f"Harsh operation: {c['harsh_count']} sudden jerks in the last 30 s",
        "Smooth out lever inputs. Jerky swings stress the machine and unsettle the cab.", "smooth", "Behaviour")
    add("idle", w["len"] >= 20 and w["ratio"] > 0.4, "warning",
        f"Excessive idling: {w['ratio']:.0%} of the last 30 min",
        "If the wait is longer than 5 minutes, shut the engine down.", "idle", "Behaviour")
    add("fuel", w["len"] >= 20 and w["cycles"] <= 6 and w["fpc"] > 1.0, "warning",
        f"Fuel per load cycle {w['fpc']:.2f} L, baseline about 0.5 L",
        "Fuel is being used without productive work. Check for idling or waiting.", "idle", "Behaviour")
    add("anomaly", c["anomaly_flagged"], "warning", "Unusual usage pattern flagged by the model",
        "This 30-minute window doesn't look like your normal work. Review idling, belt use and jerky operation.",
        None, "Behaviour")
    add("fatigue", c["since_break"] >= 240, "warning", "4 hours without a break",
        "Take a 15-minute break. Say \"take a break\" to log it.", None, "Wellbeing")
    add("weather", c["weather"] in ("Rainy", "Windy"), "info", f"{c['weather']}: stop zone widened to {z['red']} m",
        "Wet ground. Reduce swing speed and stay back from trench edges." if c["weather"] == "Rainy"
        else "High wind. Keep loads low when slewing.", "weather", "Conditions")
    return out


class Rules:
    def __init__(self):
        self.active = {}

    def evaluate(self, ctx: dict):
        new, live = [], []
        for c in conditions(ctx):
            cur = self.active.get(c["key"])
            if not c["on"]:
                self.active.pop(c["key"], None)
                continue
            if cur is None or RANK[c["severity"]] > RANK[cur]:
                new.append(c)
            self.active[c["key"]] = c["severity"]
            live.append(c)
        live.sort(key=lambda x: -RANK[x["severity"]])
        return new, live
