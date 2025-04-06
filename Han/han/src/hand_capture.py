import cv2
import mediapipe as mp
import csv
import os
from datetime import datetime

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Create folders
csv_path = os.path.join("data", "hand_landmarks.csv")
img_dir = os.path.join("data", "images")
os.makedirs(img_dir, exist_ok=True)

# Create CSV if it doesn't exist
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["image_path", "label"] + [
            f"{coord}{i}" for i in range(21) for coord in ["x", "y", "z"]
        ]
        writer.writerow(header)

# Start camera
cap = cv2.VideoCapture(0)

print("📸 Starting capture. Press 's' to save a frame, 'q' to quit.")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, 
                                      mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and results.multi_hand_landmarks:
        label = input("📝 Enter label for this gesture: ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"{label}_{timestamp}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        cv2.imwrite(img_path, frame)

        for hand in results.multi_hand_landmarks:
            data = [img_path, label]
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])

            with open(csv_path, mode='a', newline='') as f:
                csv.writer(f).writerow(data)

            print(f"✅ Saved {img_filename} and landmarks")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
