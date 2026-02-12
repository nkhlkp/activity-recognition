import pandas as pd
import numpy as np
import os

WINDOW_SIZE = 48
STRIDE = 12

def load_file(path):
    df = pd.read_csv(path)
    return df[["ax", "ay", "az"]].values

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

    X = np.concatenate(all_windows)
    y = np.full(len(X), label)
    return X, y


# Folder structure:
# data/
#   idle/
#   walk/
#   jump/

X_idle, y_idle = process_folder("data/idle", 0)
X_walk, y_walk = process_folder("data/walk", 1)
X_jump, y_jump = process_folder("data/jump", 2)

X = np.concatenate([X_idle, X_walk, X_jump])
y = np.concatenate([y_idle, y_walk, y_jump])

print("Final dataset shape:", X.shape)
print("Labels shape:", y.shape)

np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset saved as X.npy and y.npy")

unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

