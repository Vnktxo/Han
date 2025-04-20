import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from deepface import DeepFace

# Load your trained hand sign model
# Corrected model path
model = load_model("c:/Users/venkatesh/Han/han/src/models/sign_mnist_model.h5")
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
           'I', 'L', 'Nothing', 'Space']  # Change if needed

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# OpenCV webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    hand_sign = "None"
    emotion = "Unknown"

    # --- Hand Sign Prediction ---
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_data = []
            y_data = []

            for lm in hand_landmarks.landmark:
                x_data.append(lm.x)
                y_data.append(lm.y)

            x_data = np.array(x_data)
            y_data = np.array(y_data)
            x_data = x_data - np.min(x_data)
            y_data = y_data - np.min(y_data)

            input_data = np.concatenate([x_data, y_data])
            input_data = np.reshape(input_data, (28, 28))  # Reshape to 28x28
            input_data = np.expand_dims(
                input_data, axis=-1)  # Add channel dimension
            input_data = np.expand_dims(
                input_data, axis=0)  # Add batch dimension

            try:
                prediction = model.predict(input_data)
                class_id = np.argmax(prediction)
                hand_sign = classes[class_id]
            except Exception as e:
                hand_sign = "Error"
                print(f"Prediction error: {e}")

            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

    # --- Facial Expression Prediction ---
    try:
        face_result = DeepFace.analyze(
            frame, actions=["emotion"], enforce_detection=False)
        emotion = face_result[0].get('dominant_emotion', "Unknown")
    except Exception as e:
        emotion = "No Face"
        print(f"Facial expression error: {e}")

    # --- Display Results ---
    cv2.putText(frame, f'Sign: {hand_sign}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Expression: {emotion}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("HandySense - Sign + Expression", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
