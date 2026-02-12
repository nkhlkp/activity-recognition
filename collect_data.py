import socket
import time
import csv
import keyboard
import os

HOST = "0.0.0.0"
PORT = 5000

OUTPUT_DIR = "data"

# ===============================
# Setup
# ===============================

os.makedirs(f"{OUTPUT_DIR}/idle", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/walk", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/jump", exist_ok=True)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print(f"[INFO] Listening on {HOST}:{PORT}")
conn, addr = server.accept()
print(f"[INFO] Connected by {addr}")
conn.settimeout(0.05)

# ===============================
# CSV writers for each label
# ===============================

timestamp = int(time.time())
files = {}
writers = {}
counts = {}

for label in ["idle", "walk", "jump"]:
    path = f"{OUTPUT_DIR}/{label}/{label}_{timestamp}.csv"
    f = open(path, mode="w", newline="")
    w = csv.writer(f)
    w.writerow(["timestamp_ms", "ax", "ay", "az"])
    files[label] = f
    writers[label] = w
    counts[label] = 0

# ===============================
# Instructions
# ===============================

print()
print("=" * 50)
print("  REAL-TIME DATA COLLECTION")
print("=" * 50)
print()
print("  Hold your phone in your POCKET or HAND")
print("  and keep it the SAME WAY the whole time!")
print()
print("  Controls:")
print("    [I] = label as IDLE   (stand still)")
print("    [W] = label as WALK   (walk around)")
print("    [J] = label as JUMP   (jump!)")
print("    [P] = PAUSE           (stop labeling)")
print("    [Q] = QUIT & SAVE")
print()
print("=" * 50)
print()

current_label = None
buffer = ""

# ===============================
# Main Loop
# ===============================

try:
    while True:
        # Check keyboard input
        if keyboard.is_pressed("i"):
            if current_label != "idle":
                current_label = "idle"
                print(f"\n>>> Recording: IDLE")
        elif keyboard.is_pressed("w"):
            if current_label != "walk":
                current_label = "walk"
                print(f"\n>>> Recording: WALK")
        elif keyboard.is_pressed("j"):
            if current_label != "jump":
                current_label = "jump"
                print(f"\n>>> Recording: JUMP")
        elif keyboard.is_pressed("p"):
            if current_label is not None:
                current_label = None
                print(f"\n>>> PAUSED (no label)")
        elif keyboard.is_pressed("q"):
            print(f"\n>>> QUITTING...")
            break

        # Receive sensor data
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

            if len(parts) != 4:
                continue

            if current_label is None:
                continue  # Not labeling right now

            try:
                timestamp_ms = float(parts[0])
                ax = float(parts[1])
                ay = float(parts[2])
                az = float(parts[3])

                writers[current_label].writerow([timestamp_ms, ax, ay, az])
                counts[current_label] += 1

                # Print progress every 100 samples
                total = sum(counts.values())
                if total % 100 == 0:
                    print(f"  [{current_label.upper():>4s}] idle:{counts['idle']:>5d} | walk:{counts['walk']:>5d} | jump:{counts['jump']:>5d}", end="\r")

            except:
                continue

except KeyboardInterrupt:
    print("\n[INFO] Stopped by Ctrl+C")

finally:
    conn.close()
    server.close()
    for f in files.values():
        f.close()

    print()
    print("=" * 50)
    print("  DATA SAVED")
    print("=" * 50)
    for label in ["idle", "walk", "jump"]:
        print(f"  {label}: {counts[label]} samples")
    print("=" * 50)
    print()
    print("  Now run: python build_dataset.py")
    print("  Then:    python train_baseline.py")
