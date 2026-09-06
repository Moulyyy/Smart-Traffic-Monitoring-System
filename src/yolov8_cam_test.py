import os
from pathlib import Path
import cv2
from ultralytics import YOLO

# Resolve project root and model path
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "best.pt"

# Load YOLOv8 model
model_path = os.getenv("YOLO_MODEL_PATH", str(DEFAULT_MODEL_PATH))
model = YOLO(model_path)

# Open USB camera
cap = cv2.VideoCapture(0)  # Use the correct camera index (0 is usually the default)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model.predict(source=frame, save=False, conf=0.3)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
