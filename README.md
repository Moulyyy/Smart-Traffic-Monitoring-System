# Smart Traffic Monitoring System 🚦🏎️

[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Nano-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/)
[![Model](https://img.shields.io/badge/Model-YOLOv8-blue?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Telegram](https://img.shields.io/badge/Alerts-Telegram%20Bot-2CA5E0?logo=telegram&logoColor=white)](https://core.telegram.org/bots/api)

A real-time edge AI traffic monitoring and collision detection prototype powered by **YOLOv8** on an **NVIDIA Jetson Nano**. Designed for smart city intersections, the system monitors multi-lane vehicle flow, detects potential collisions, and automatically dispatches instant alert notifications via Telegram.

---

## 📌 Features

- **Real-Time Vehicle Detection**: Identifies and classifies vehicles at high frame rates using a fine-tuned YOLOv8 model.
- **Automated Collision Detection**: Real-time bounding box proximity and overlap evaluation to instantly spot accidents.
- **Instant Telegram Alerts**: Automated incident reporting with vehicle counts delivered directly to Telegram in under 3 seconds.
- **Edge AI Optimized**: Lightweight architecture designed for on-device inference on the NVIDIA Jetson Nano.

---

## 🛠️ Hardware & Prototype Setup

The system monitors a physical scaled-down junction with multi-lane markings and miniature vehicles:

<p align="center">
  <img src="docs/images/hardware_setup.jpg" alt="Hardware Setup" width="600" />
  <br />
  <em>NVIDIA Jetson Nano 4GB with overhead USB camera setup</em>
</p>

| Component | Description |
| :--- | :--- |
| **Compute Board** | NVIDIA Jetson Nano (4GB RAM, 128-core Maxwell GPU) |
| **Camera** | Overhead HD USB Camera |
| **Testbed** | Scaled intersection base with roadway lane markings |
| **Fleet** | 6 miniature vehicle models (Gallardo, Porsche 911, Bat Mobile, Ravenger, Honda S800, F1) |

---

## 🔄 System Flow

<p align="center">
  <img src="docs/images/system_flowchart.png" alt="System Flowchart" width="450" />
</p>

1. **Video Ingestion**: Live stream capture from the overhead camera.
2. **YOLOv8 Inference**: Real-time object detection and vehicle classification.
3. **Tracking & Crash Detection**: Continuous monitoring of vehicle count and collision overlap.
4. **Alert Dispatch**: Real-time notification transmission to the Telegram channel.

---

## 📊 Dataset & Model Performance

The model was trained on a custom labeled dataset annotated via Roboflow and trained for 100 epochs using Google Colab GPU acceleration.

<p align="center">
  <img src="docs/images/dataset_roboflow.jpg" alt="Roboflow Dataset" width="48%" />
  <img src="docs/images/training_curves.png" alt="Training Metrics" width="48%" />
</p>

| Metric | Result |
| :--- | :--- |
| **mAP@0.5** | **~85%** |
| **Precision** | **~88%** |
| **Recall** | **~86%** |
| **Edge Frame Rate** | **4–6 FPS** (Jetson Nano) |
| **Alert Latency** | **2–3 seconds** (Telegram) |

---

## 📺 Live Output

<p align="center">
  <img src="docs/images/detection_output.jpg" alt="Live Detection" width="58%" />
  <img src="docs/images/telegram_alerts.jpg" alt="Telegram Alerts" width="38%" />
  <br />
  <em>Live detection view with collision bounding box (left) and automated Telegram alert stream (right)</em>
</p>

---

## 📂 Project Structure

```plaintext
Smart-Traffic-Monitoring-System/
├── src/                       # Application source pipelines
│   ├── traffic_monitor.py     # Main real-time monitor with debounced Telegram alerts
│   ├── crash_telegram.py      # Core collision detection pipeline
│   ├── yolov8_cam_test.py     # Camera and inference test script
│   └── dataset_converter.py   # Dataset formatting utility
├── models/                    # Trained YOLOv8 weight checkpoints (.pt)
├── data/                      # Training and testing datasets
├── docs/                      # Presentation PDF and architecture images
├── .env.example               # Configuration template for Telegram credentials
├── requirements.txt           # Python project dependencies
└── README.md                  # Project overview and documentation
```

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Moulyyy/Smart-Traffic-Monitoring-System.git
cd Smart-Traffic-Monitoring-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Telegram Alerts (Optional)
Copy `.env.example` to `.env` and enter your bot details:
```bash
cp .env.example .env
```
```env
TELEGRAM_BOT_TOKEN="your_bot_token"
TELEGRAM_CHAT_ID="your_chat_id"
```

### 4. Run the Monitor
```bash
# Camera validation test
python src/yolov8_cam_test.py

# Main traffic & crash monitoring pipeline
python src/traffic_monitor.py
```

---

## 📄 References
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [NVIDIA Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
