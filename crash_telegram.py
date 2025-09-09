import cv2
import numpy as np
from ultralytics import YOLO
import uuid
import requests

# Telegram config
BOT_TOKEN = '7709578331:AAGFslWozdLHcVXT5hphT27SpOis0w_Emvw'
CHAT_ID = '1855281040'

def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    data = {'chat_id': CHAT_ID, 'text': message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print(f"Telegram Error: {response.text}")
    except Exception as e:
        print(f"Exception in Telegram Notification: {e}")

# Load YOLOv8 model
model = YOLO('best.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

print("Press 'q' to quit.")

# Crash log to avoid multiple notifications
crash_log = set()

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

    results = model.predict(source=frame, save=False, conf=0.3)
    detections = results[0].boxes

    current_boxes = {}
    frame_copy = frame.copy()
    car_count = 0

    for i, box in enumerate(detections):
        cls = int(box.cls[0])
        if cls != 0:  # Assuming class 0 is 'car'
            continue

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        box_id = str(uuid.uuid4())
        current_boxes[box_id] = (x1, y1, x2, y2)
        car_count += 1
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crash Detection
    ids = list(current_boxes.keys())
    new_crash_detected = False
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            box1, box2 = current_boxes[id1], current_boxes[id2]

            if iou(box1, box2) > 0.3:
                key = tuple(sorted([id1, id2]))
                if key not in crash_log:
                    crash_log.add(key)
                    print("🚨 Crash Detected!")
                    new_crash_detected = True
                    cx = int((box1[0] + box1[2]) / 2)
                    cy = int((box1[1] + box1[3]) / 2)
                    cv2.putText(frame_copy, "CRASH", (cx - 20, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.circle(frame_copy, (cx, cy), 20, (0, 0, 255), 5)

    # Send Telegram only for new crashes
    if new_crash_detected:
        msg = f"🚨 Crash detected!\nTotal Cars Detected: {car_count}"
        send_telegram_message(msg)

    cv2.imshow("Crash Detection", frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
