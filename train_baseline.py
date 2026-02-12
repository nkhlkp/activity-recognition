import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

X = np.load("X.npy")
y = np.load("y.npy")

# X is already feature vectors (N, 19) — no flattening needed
X_flat = X

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

clf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

# Save model
joblib.dump(clf, "rf_model.pkl")
print("\nModel saved as rf_model.pkl")
