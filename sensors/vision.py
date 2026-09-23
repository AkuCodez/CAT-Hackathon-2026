"""Webcam safety sensing with YOLOv8, in one of two modes.

  pip install ultralytics opencv-python requests
  python sensors/vision.py --facing rear      # proximity: people around the machine
  python sensors/vision.py --mode operator    # operator: phone use while the engine runs

Proximity: distance is estimated from the height of the person's bounding box: distance = K / box_height.
Calibrate once: stand at a known distance (default 2 m), fully in frame, and press "c".
Operator: a phone visible for 2 s while the engine is on logs a Distraction incident (at most one per 30 s).
Keys: c = calibrate, q = quit
"""
import argparse
import time

import cv2
import requests
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--api", default="http://localhost:8000")
p.add_argument("--cam", type=int, default=0)
p.add_argument("--k", type=float, default=1665.0, help="distance constant (px * m), set by calibration")
p.add_argument("--calib-distance", type=float, default=2.0)
p.add_argument("--fov", type=float, default=70.0, help="camera horizontal field of view, degrees")
p.add_argument("--facing", choices=["rear", "front"], default="rear")
p.add_argument("--conf", type=float, default=0.5)
p.add_argument("--scale", type=float, default=1.0, help="multiply distances, e.g. 3 to map a room to a site")
p.add_argument("--mode", choices=["proximity", "operator"], default="proximity")
p.add_argument("--phone-conf", type=float, default=0.35, help="phone detection confidence in operator mode")
args = p.parse_args()

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(args.cam)
K = args.k
zones = {"red": 5, "amber": 10}
last_post = last_zone = 0.0
offset = 180 if args.facing == "rear" else 0
PHONE = 67  # COCO class id for "cell phone"
phone_since = phone_seen = last_distraction = last_state = 0.0
engine_on = False

while True:
    ok, frame = cap.read()
    if not ok:
        print("camera read failed")
        break
    H, W = frame.shape[:2]
    now = time.time()
    heights = []

    if args.mode == "operator":
        # Known limit: any phone in view counts, including one on the dash; upgrade = pose keypoints to require it in hand
        phones = model(frame, classes=[PHONE], conf=args.phone_conf, verbose=False)[0].boxes.xyxy.tolist()
        for (x1, y1, x2, y2) in phones:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(frame, "PHONE", (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if phones:
            phone_seen = now
            phone_since = phone_since or now
        elif now - phone_seen > 0.7:  # tolerate single-frame detection dropouts
            phone_since = 0.0
        held = now - phone_since if phone_since else 0.0

        if now - last_state > 2:
            try:
                engine_on = requests.get(f"{args.api}/state", timeout=1).json()["machine"]["engine_on"]
            except (requests.RequestException, KeyError, ValueError):
                pass
            last_state = now

        if held >= 2 and engine_on and now - last_distraction > 30:
            try:
                requests.post(f"{args.api}/incidents", json={
                    "type": "Distraction", "severity": "warning", "source": "Cab camera",
                    "note": f"Phone in use for {held:.0f} s while engine running"}, timeout=1)
                last_distraction = now
            except requests.RequestException:
                pass
        if now - last_distraction < 3:
            cv2.putText(frame, "DISTRACTION LOGGED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        status = f"operator | engine {'on' if engine_on else 'off'} | phone {held:.1f} s | q=quit"
    else:
        res = model(frame, classes=[0], conf=args.conf, verbose=False)[0]
        persons = []
        for (x1, y1, x2, y2) in res.boxes.xyxy.tolist():
            h = max(1.0, y2 - y1)
            d = K / h * args.scale
            bearing = offset + ((x1 + x2) / 2 / W - 0.5) * args.fov
            persons.append({"distance_m": round(d, 2), "bearing_deg": round(bearing, 1)})
            heights.append(h)
            colour = (0, 0, 255) if d < zones["red"] else (0, 165, 255) if d < zones["amber"] else (0, 200, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3)
            cv2.putText(frame, f"{d:.1f} m", (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

        if now - last_post > 0.3:
            try:
                r = requests.post(f"{args.api}/proximity", json={"persons": persons, "source": "camera"}, timeout=1)
                zones = r.json().get("zones", zones)
            except requests.RequestException:
                pass
            last_post = now
        status = f"stop {zones['red']} m | caution {zones['amber']} m | c=calibrate q=quit"

    cv2.putText(frame, status, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Cab Companion - proximity", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c") and heights:
        K = max(heights) * args.calib_distance / args.scale
        print(f"calibrated: K = {K:.0f} (use --k {K:.0f} next time)")

cap.release()
cv2.destroyAllWindows()
