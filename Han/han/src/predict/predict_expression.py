from fer import FER
import cv2
import sys
import time


def initialize_webcam():
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return None

    # Lower buffer size for less latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Clear buffer
    for _ in range(3):
        cap.read()

    print("✅ Webcam started. Press 'q' to quit.")
    return cap


# Initialize FER model
emotion_detector = FER(mtcnn=True)

# Initialize emotion smoothing variables
emotion_history = []
MAX_HISTORY = 5
last_stable_emotion = None
last_stable_score = 0

# Start webcam
cap = initialize_webcam()
if cap is None:
    sys.exit(1)

try:
    last_process_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam.")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)  # Mirror for better UX
        current_time = time.time()

        # Process every 200ms (5 FPS) - this is key for stability
        if current_time - last_process_time >= 0.2:
            try:
                emotion, score = emotion_detector.top_emotion(frame)

                if emotion:
                    # Add current emotion to history
                    emotion_history.append((emotion, score))
                    if len(emotion_history) > MAX_HISTORY:
                        emotion_history.pop(0)

                    # Get most frequent emotion from history
                    emotions = [e[0] for e in emotion_history]
                    most_common = max(set(emotions), key=emotions.count)

                    # Calculate average score for most common emotion
                    avg_score = sum(
                        [s[1] for s in emotion_history if s[0] == most_common]
                    )
                    avg_score /= emotions.count(most_common)

                    # Only update display if emotion is stable
                    if emotions.count(most_common) >= 3:  # Require 3/5
                        last_stable_emotion = most_common
                        last_stable_score = avg_score

                    # Display the stable emotion
                    if last_stable_emotion:
                        # Add background rectangle
                        text = (
                            f"Emotion: {last_stable_emotion} "
                            f"({last_stable_score:.2f})"
                        )
                        (text_width, text_height), _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                        )
                        cv2.rectangle(
                            frame,
                            (10, 10),
                            (10 + text_width + 10, 10 + text_height + 10),
                            (255, 255, 255),
                            -1
                        )
                        cv2.putText(
                            frame, text, (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 0), 2
                        )

                last_process_time = current_time
            except Exception as e:
                print(f"⚠️ An error occurred: {e}")

        cv2.imshow("Facial Expression Recognition", frame)

        # Reduced delay for smoother display
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
