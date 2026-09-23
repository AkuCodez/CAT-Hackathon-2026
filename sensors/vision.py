"""Person detection for proximity safety, using a webcam and YOLOv8.

  pip install ultralytics opencv-python requests
  python sensors/vision.py --facing rear

Distance is estimated from the height of the person's bounding box: distance = K / box_height.
Calibrate once: stand at a known distance (default 2 m), fully in frame, and press "c".
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
p.add_argument("--k", type=float, default=900.0, help="distance constant (px * m), set by calibration")
p.add_argument("--calib-distance", type=float, default=2.0)
p.add_argument("--fov", type=float, default=70.0, help="camera horizontal field of view, degrees")
p.add_argument("--facing", choices=["rear", "front"], default="rear")
p.add_argument("--conf", type=float, default=0.5)
p.add_argument("--scale", type=float, default=1.0, help="multiply distances, e.g. 3 to map a room to a site")
args = p.parse_args()

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(args.cam)
K = args.k
zones = {"red": 5, "amber": 10}
last_post = last_zone = 0.0
offset = 180 if args.facing == "rear" else 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("camera read failed")
        break
    H, W = frame.shape[:2]
    res = model(frame, classes=[0], conf=args.conf, verbose=False)[0]
    persons, heights = [], []
    for (x1, y1, x2, y2) in res.boxes.xyxy.tolist():
        h = max(1.0, y2 - y1)
        d = K / h * args.scale
        bearing = offset + ((x1 + x2) / 2 / W - 0.5) * args.fov
        persons.append({"distance_m": round(d, 2), "bearing_deg": round(bearing, 1)})
        heights.append(h)
        colour = (0, 0, 255) if d < zones["red"] else (0, 165, 255) if d < zones["amber"] else (0, 200, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3)
        cv2.putText(frame, f"{d:.1f} m", (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

    now = time.time()
    if now - last_post > 0.3:
        try:
            r = requests.post(f"{args.api}/proximity", json={"persons": persons, "source": "camera"}, timeout=1)
            zones = r.json().get("zones", zones)
        except requests.RequestException:
            pass
        last_post = now

    cv2.putText(frame, f"stop {zones['red']} m | caution {zones['amber']} m | c=calibrate q=quit",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Cab Companion - proximity", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c") and heights:
        K = max(heights) * args.calib_distance / args.scale
        print(f"calibrated: K = {K:.0f} (use --k {K:.0f} next time)")

cap.release()
cv2.destroyAllWindows()
