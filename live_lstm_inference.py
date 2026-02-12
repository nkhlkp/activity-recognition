import socket
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from statistics import mode
import pyautogui
import time


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

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    "lstm_model.pth",
    map_location=device,
    weights_only=False
)


model = LSTMModel().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

mean = checkpoint["mean"]
std = checkpoint["std"]

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
                #       Real-Time Loop
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

            # Normalize
            window = (window - mean) / std

            window_tensor = torch.tensor(
                window[np.newaxis, :, :],
                dtype=torch.float32
            ).to(device)

            with torch.no_grad():
                output = model(window_tensor)
                pred = torch.argmax(output, dim=1).item()

            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            prediction_buffer.append(pred)

            if len(prediction_buffer) == SMOOTHING_SIZE:
                final_pred = mode(prediction_buffer)
                conf = probs[final_pred] * 100

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

                                   