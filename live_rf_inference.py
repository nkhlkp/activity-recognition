import socket
import numpy as np
import joblib
from collections import deque
from statistics import mode
import pyautogui
import time

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


# ===============================
# CONFIG
# ===============================

HOST = "10.145.27.140"
PORT = 5000
   
WINDOW_SIZE = 48
STRIDE = 12
SMOOTHING_SIZE = 5

current_state = None
last_jump_time = 0
JUMP_COOLDOWN = 0.8  # seconds

# ===============================
# Load Model
# ===============================

clf = joblib.load("rf_model.pkl")
print("[INFO] Random Forest model loaded")

# ===============================
# Feature Extraction (must match build_dataset.py)
# ===============================

def extract_features(window):
    """Extract orientation-invariant features from a window of (48, 3) accel data."""
    ax, ay, az = window[:, 0], window[:, 1], window[:, 2]

    mag = np.sqrt(ax**2 + ay**2 + az**2)

    jerk_ax = np.diff(ax)
    jerk_ay = np.diff(ay)
    jerk_az = np.diff(az)
    jerk_mag = np.sqrt(jerk_ax**2 + jerk_ay**2 + jerk_az**2)

    features = [
        mag.mean(), mag.std(), mag.min(), mag.max(), np.ptp(mag),
        ax.std(), ay.std(), az.std(),
        jerk_mag.mean(), jerk_mag.std(), jerk_mag.max(),
        np.mean(ax**2), np.mean(ay**2), np.mean(az**2),
        np.sum(np.diff(np.sign(jerk_ax)) != 0),
        np.sum(np.diff(np.sign(jerk_ay)) != 0),
        np.sum(np.diff(np.sign(jerk_az)) != 0),
        mag.var(),
        np.percentile(mag, 75) - np.percentile(mag, 25),
    ]
    return np.array(features).reshape(1, -1)

# ===============================
# Buffers
# ===============================

data_buffer = deque(maxlen=WINDOW_SIZE)
prediction_buffer = deque(maxlen=SMOOTHING_SIZE)

sample_counter = 0

# ===============================
# TCP Server
# ===============================

LABELS = ["IDLE", "WALK", "JUMP"]

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print(f"[INFO] Listening on {HOST}:{PORT}")
conn, addr = server.accept()
print(f"[INFO] Connected by {addr}")
conn.settimeout(0.1)

print("[INFO] Switch to browser game NOW! Starting in 5 seconds...")
for i in range(5, 0, -1):
    print(f"  {i}...")
    time.sleep(1)
print("[INFO] GO!")

buffer = ""

# ===============================
# Real-Time Loop
# ===============================

while True:
    try:
        data = conn.recv(4096).decode("utf-8")
    except socket.timeout:
        continue

    if not data:
        print("[INFO] Connection closed")
        break

    buffer += data

    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.strip()
        parts = line.split(",")

        if len(parts) != 4:  # timestamp, ax, ay, az
            continue

        try:
            ax = float(parts[1])
            ay = float(parts[2])
            az = float(parts[3])

            data_buffer.append([ax, ay, az])
            sample_counter += 1

        except:
            continue

        # Only predict if we have full window
        if len(data_buffer) == WINDOW_SIZE and sample_counter % STRIDE == 0:

            window = np.array(data_buffer)

            # Extract orientation-invariant features
            window_feat = extract_features(windo  w)

            pred = clf.predict(window_feat)[0]
            proba = clf.predict_proba(window_feat)[0]

            prediction_buffer.append(pred)

            if len(prediction_buffer) == SMOOTHING_SIZE:
                final_pred = mode(prediction_buffer)
                conf = proba[int(final_pred)] * 100

                if final_pred == 0:  # IDLE
                    if current_state != "IDLE":
                        pyautogui.keyUp("right")
                        current_state = "IDLE"
                        print(f"IDLE | conf: {conf:5.1f}%")

                elif final_pred == 1:  # WALK
                    if current_state != "WALK":
                        pyautogui.keyDown("right")
                        current_state = "WALK"
                        print(f"WALK | conf: {conf:5.1f}%")
       
                elif final_pred == 2:  # JUMP
                    now = time.time()
                    if now - last_jump_time > JUMP_COOLDOWN:
                        pyautogui.keyDown("space")
                        time.sleep(0.5)
                        pyautogui.keyUp("space")
                        last_jump_time = now
                        print(f"JUMP | conf: {conf:5.1f}%")
            