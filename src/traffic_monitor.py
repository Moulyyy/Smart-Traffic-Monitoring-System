import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import uuid
import requests

# Resolve project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "best.pt"

# Telegram config (can be set via environment variables or .env)
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7709578331:AAGFslWozdLHcVXT5hphT27SpOis0w_Emvw')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '1855281040')

def send_telegram_message(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram Warning: Bot token or chat ID not configured.")
        return
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    data = {'chat_id': CHAT_ID, 'text': message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print(f"Telegram Error: {response.text}")
    except Exception as e:
        print(f"Exception in Telegram Notification: {e}")

# Load YOLOv8 model
model_path = os.getenv("YOLO_MODEL_PATH", str(DEFAULT_MODEL_PATH))
model = YOLO(model_path)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

print("Press 'q' to quit.")

# State tracking
crash_log = set()
prev_car_count = -1
prev_crash_state = None  # "crash" or "no_crash"

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # forward inference
    current_boxes = {}
    frame_copy = frame.copy()
    car_count = 0
    crash_happened = False

    if results and results.boxes is not None:
        for i in range(len(results.boxes)):
            box = results.boxes[i]
            cls_id = int(box.cls.item())
            if cls_id != 0:  # Assuming class 0 = car
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_id = str(uuid.uuid4())
            current_boxes[box_id] = (x1, y1, x2, y2)
            car_count += 1

            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_copy, f"Car", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Crash detection
    ids = list(current_boxes.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            box1, box2 = current_boxes[id1], current_boxes[id2]

            if iou(box1, box2) > 0.3:
                crash_happened = True
                key = tuple(sorted([id1, id2]))
                if key not in crash_log:
                    crash_log.add(key)
                    print("🚨 Crash Detected!")
                    cx = int((box1[0] + box1[2]) / 2)
                    cy = int((box1[1] + box1[3]) / 2)
                    cv2.putText(frame_copy, "CRASH", (cx - 20, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.circle(frame_copy, (cx, cy), 20, (0, 0, 255), 5)

    # 📨 Telegram Notifications
    send_message = False
    message = ""

    if car_count != prev_car_count:
        send_message = True
        message += f"🚗 Cars Detected: {car_count}\n"
        prev_car_count = car_count

    if crash_happened and prev_crash_state != "crash":
        send_message = True
        message += "🚨 Crash Detected!"
        prev_crash_state = "crash"
    elif not crash_happened and prev_crash_state != "no_crash":
        send_message = True
        message += "✅ No crash detected."
        prev_crash_state = "no_crash"

    if send_message and message.strip():
        send_telegram_message(message.strip())

    cv2.imshow("Crash Detection", frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
