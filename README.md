# Activity Recognition System for Game Control

Control a browser-based Super Mario game using your phone's accelerometer. Walk, jump, and idle in real life and Mario does the same on screen.

## How It Works

```
Phone (Android App)  --->  TCP Socket  --->  Python Server (PC)
   Accelerometer                              ML Model → Keyboard Input → Game
```

1. The **Android app** reads accelerometer data and streams it over TCP
2. The **Python server** receives the data, runs it through a trained ML model
3. The model classifies your motion as **IDLE**, **WALK**, or **JUMP**
4. Keyboard inputs are sent to the browser game accordingly

## Project Structure

| File                     | Description                                                      |
| ------------------------ | ---------------------------------------------------------------- |
| `android_app/`           | Android app (Kotlin) — streams accelerometer data over TCP       |
| `collect_data.py`        | Real-time data collection with keyboard labeling                 |
| `build_dataset.py`       | Preprocesses CSVs into windowed datasets with feature extraction |
| `train_baseline.py`      | Trains a Random Forest classifier                                |
| `train_lstm.py`          | Trains an LSTM classifier                                        |
| `live_rf_inference.py`   | Live inference with Random Forest + game controls                |
| `live_lstm_inference.py` | Live inference with LSTM + game controls                         |
| `imu_server.py`          | Simple server for visualizing raw accelerometer data             |

## Setup

### Prerequisites

- Python 3.11+
- Android phone with the app installed
- PC and phone on the same WiFi network

### Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Collect Training Data

```bash
python collect_data.py
```

- Connect the phone, then label your motions in real-time:
  - **I** = idle, **W** = walk, **J** = jump, **P** = pause, **Q** = quit
- Keep the phone in the **same position** (e.g., pocket) throughout collection

### 2. Build Dataset

```bash
python build_dataset.py
```

### 3. Train Model

**Random Forest:**

```bash
python train_baseline.py
```

**LSTM:**

```bash
python train_lstm.py
```

### 4. Play

```bash
python live_rf_inference.py
# or
python live_lstm_inference.py
```

1. Run the script and connect the phone
2. During the 5-second countdown, switch to the Mario game in your browser
3. Start streaming from the app — your movements control the game

### Controls

| Motion | Game Action      |
| ------ | ---------------- |
| IDLE   | Release all keys |
| WALK   | Hold RIGHT arrow |
| JUMP   | Press SPACE      |

## Android App

The app is in `android_app/`. Open it in Android Studio, build, and install on your phone. It provides:

- IP/Port input to connect to the PC server
- Start/Stop streaming toggle
- Streams accelerometer data as CSV over TCP: `timestamp,ax,ay,az`

## Config

Update `HOST` in the Python scripts to match your PC's local IP address:

```python
HOST = "0.0.0.0"  # Listens on all interfaces
```

Find your IP with `ipconfig` (Windows) or `ifconfig` (Mac/Linux).
