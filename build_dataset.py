import pandas as pd
import numpy as np
import os

WINDOW_SIZE = 48
STRIDE = 12

def load_file(path):
    df = pd.read_csv(path)
    return df[["ax", "ay", "az"]].values

def extract_features(window):
    """Extract orientation-invariant features from a window of (48, 3) accel data."""
    ax, ay, az = window[:, 0], window[:, 1], window[:, 2]

    # Acceleration magnitude (removes gravity direction dependence)
    mag = np.sqrt(ax**2 + ay**2 + az**2)

    # Jerk (rate of change) — captures sudden movements like jumps
    jerk_ax = np.diff(ax)
    jerk_ay = np.diff(ay)
    jerk_az = np.diff(az)
    jerk_mag = np.sqrt(jerk_ax**2 + jerk_ay**2 + jerk_az**2)

    features = [
        # Magnitude stats
        mag.mean(), mag.std(), mag.min(), mag.max(), np.ptp(mag),

        # Per-axis variance (captures vibration, not orientation)
        ax.std(), ay.std(), az.std(),

        # Jerk stats (captures sudden motion changes)
        jerk_mag.mean(), jerk_mag.std(), jerk_mag.max(),

        # Signal energy
        np.mean(ax**2), np.mean(ay**2), np.mean(az**2),

        # Zero crossing rate of jerk (how often motion changes direction)
        np.sum(np.diff(np.sign(jerk_ax)) != 0),
        np.sum(np.diff(np.sign(jerk_ay)) != 0),
        np.sum(np.diff(np.sign(jerk_az)) != 0),

        # Magnitude variance (key: idle=low, walk=medium, jump=high)
        mag.var(),

        # Interquartile range of magnitude
        np.percentile(mag, 75) - np.percentile(mag, 25),
    ]
    return np.array(features)

def create_windows(data, window_size, stride):
    windows = []
    for start in range(0, len(data) - window_size, stride):
        end = start + window_size
        windows.append(data[start:end])
    return np.array(windows)

def process_folder(folder_path, label):
    all_windows = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            path = os.path.join(folder_path, file)
            data = load_file(path)
            windows = create_windows(data, WINDOW_SIZE, STRIDE)
            all_windows.append(windows)

    if len(all_windows) == 0:
        return None, None

    X_raw = np.concatenate(all_windows)

    # Extract features from each window
    X_feat = np.array([extract_features(w) for w in X_raw])

    y = np.full(len(X_feat), label)
    return X_raw, X_feat, y


# Folder structure:
# data/
#   idle/
#   walk/
#   jump/

X_raw_idle, X_idle, y_idle = process_folder("data/idle", 0)
X_raw_walk, X_walk, y_walk = process_folder("data/walk", 1)
X_raw_jump, X_jump, y_jump = process_folder("data/jump", 2)

# Save raw windows (for LSTM if needed)
X_raw = np.concatenate([X_raw_idle, X_raw_walk, X_raw_jump])
np.save("X_raw.npy", X_raw)

# Save engineered features (for Random Forest)
X = np.concatenate([X_idle, X_walk, X_jump])
y = np.concatenate([y_idle, y_walk, y_jump])

print("Raw dataset shape:", X_raw.shape)
print("Feature dataset shape:", X.shape)
print("Labels shape:", y.shape)

np.save("X.npy", X)
np.save("y.npy", y)

print("Saved X.npy (features), X_raw.npy (raw windows), y.npy (labels)")

unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

