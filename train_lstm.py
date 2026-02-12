import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1. Load Dataset
# =========================

X = np.load("X.npy")   # Shape: (N, 48, 3)
y = np.load("y.npy")   # Shape: (N,)

print("Dataset shape:", X.shape)

# =========================
# 2. Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 3. Normalize (VERY IMPORTANT)
# =========================

mean = X_train.mean(axis=(0, 1))
std = X_train.std(axis=(0, 1)) + 1e-8

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# =========================
# 4. Convert to PyTorch Tensors
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

# =========================
# 5. Define LSTM Model
# =========================

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
        out = out[:, -1, :]  # Last timestep
        out = self.fc(out)
        return out

model = LSTMModel().to(device)

# =========================
# 6. Training Setup
# =========================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# 7. Training Loop
# =========================

EPOCHS = 25

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# =========================
# 8. Evaluation
# =========================

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    preds = torch.argmax(outputs, dim=1)
    accuracy = (preds == y_test).float().mean()

print("\nTest Accuracy:", accuracy.item())

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test.cpu(),
    preds.cpu()
))

print("\nClassification Report:")
print(classification_report(
    y_test.cpu(),
    preds.cpu()
))

# =========================
# 9. Save Model
# =========================

torch.save({
    "model_state_dict": model.state_dict(),
    "mean": mean,
    "std": std
}, "lstm_model.pth")

print("\nModel saved as lstm_model.pth")
