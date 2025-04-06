import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), 
                                      thickness=1, 
                                      circle_radius=1)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Setup CSV
csv_path = "data/face_expression_data.csv"
os.makedirs("data/face_images", exist_ok=True)

if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_path", "label"] +
            [f"x{i}" for i in range(468)] +
            [f"y{i}" for i in range(468)] +
            [f"z{i}" for i in range(468)]
        )

# Label mapping
label_map = {
    ord('h'): 'happy',
    ord('s'): 'sad',
    ord('a'): 'angry',
    ord('n'): 'neutral'
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

print("✅ Webcam started. Press h/s/a/n to label, q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy()  # Save frame before drawing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    key = cv2.waitKey(1) & 0xFF

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw contour mesh for visual feedback
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                drawing_spec,
                drawing_spec
            )

            if key in label_map:
                label = label_map[key]
                coords = np.array([
                    [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
                ]).flatten()

                img_name = f"{label}_{len(os.listdir('data/face_images'))}.jpg"
                img_path = os.path.join("data/face_images", img_name)
                cv2.imwrite(img_path, clean_frame)

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([img_path, label] + coords.tolist())

                print(f"[📸] Captured '{label}' at {img_path}")
                cv2.waitKey(300)  # Small pause to avoid double trigger

    cv2.imshow("Face Expression Capture", frame)

    if key == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
