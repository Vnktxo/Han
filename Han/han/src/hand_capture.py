import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Setup paths for data folder and CSV file
data_dir = os.path.join(base_dir, "data")
image_dir = os.path.join(data_dir, "hand_images")
csv_file = os.path.join(data_dir, "hand_sign_data.csv")

# Create directories if they don't exist
os.makedirs(image_dir, exist_ok=True)

# Setup CSV if not exists
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_path", "label"] +
            [f"x{i}" for i in range(21)] +
            [f"y{i}" for i in range(21)] +
            [f"z{i}" for i in range(21)]
        )

# Init MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
print("Press a-z to label hand sign, q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            if key in range(97, 123):  # a-z keys
                label = chr(key)

                coords = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                ]).flatten()

                img_name = f"{label}_{len(os.listdir(image_dir))}.jpg"
                img_path = os.path.join(image_dir, img_name)
                cv2.imwrite(img_path, clean_frame)

                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([img_path, label] + coords.tolist())

                print(f"[✅] Saved {img_name} with label '{label}'")
                cv2.waitKey(300)

    cv2.imshow("Hand Sign Capture", frame)

    if key == ord('q'):
        print("[👋] Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
