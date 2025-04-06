import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# CSV setup
csv_file = "data/face_expression_data.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_path", "label"] +
            [f"x{i}" for i in range(468)] +
            [f"y{i}" for i in range(468)] +
            [f"z{i}" for i in range(468)]
        )

# Create image directory
image_dir = "data/face_images"
os.makedirs(image_dir, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
print(
    "Press a key to label (h: happy, s: sad, a: angry, n: neutral, etc). "
    "Press 'q' to quit."
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            coords = np.array([
                [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
            ]).flatten()

            # Show frame
            cv2.imshow("Face Expression Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
            elif key in [ord('h'), ord('s'), ord('a'), ord('n')]:  
                label_map = {
                    ord('h'): 'happy',
                    ord('s'): 'sad',
                    ord('a'): 'angry',
                    ord('n'): 'neutral'
                }
                label = label_map[key]
                img_name = f"{label}_{len(os.listdir(image_dir))}.jpg"
                img_path = os.path.join(image_dir, img_name)
                cv2.imwrite(img_path, frame)

                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([img_path, label] + coords.tolist())

    else:
        cv2.imshow("Face Expression Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
