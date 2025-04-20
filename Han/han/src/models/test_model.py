import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import os

# Define base project directory and construct absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # One more dirname to go up to han folder
# Updated to match the correct location
DATA_DIR = os.path.join(BASE_DIR, "src", "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEST_DATA_PATH = os.path.join(DATA_DIR, "sign_mnist_test.csv")
# Changed to match training script
MODEL_PATH = os.path.join(MODEL_DIR, "sign_mnist_model.h5")

# Check if files exist
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load test dataset
test_df = pd.read_csv(TEST_DATA_PATH)
X = test_df.drop('label', axis=1).values
y = test_df['label'].values

# Normalize images
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)

# One-hot encode labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Load trained model
model = load_model(MODEL_PATH)

# Evaluate
loss, accuracy = model.evaluate(X, y)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Optional: Classification report
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true_classes, y_pred_classes))
