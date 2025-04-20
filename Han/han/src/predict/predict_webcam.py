import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "sign_mnist_model.h5")
model = load_model(model_path)

# Labels
labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]


def preprocess(img):
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img


# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not access the webcam.")
    exit()

print("✅ Camera initialized successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)  # Mirror view

    # ROI box
    x1, y1, x2, y2 = 100, 100, 324, 324  # Square box for 28x28 scaling
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    processed = preprocess(gray)

    # Predict
    prediction = model.predict(processed, verbose=0)
    pred_index = np.argmax(prediction)
    pred_label = labels[pred_index]
    confidence = prediction[0][pred_index]

    # Display prediction
    result_text = f"Prediction: {pred_label} ({confidence*100:.1f}%)"
    cv2.putText(frame, result_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show webcam feed in larger window
    resized_frame = cv2.resize(frame, (960, 720))
    cv2.imshow("✋ Hand Sign Recognition", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
