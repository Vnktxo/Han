import cv2
import time
from deepface import DeepFace


def initialize_webcam():
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def adjust_frame(frame):
    # Reduce brightness and contrast
    frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=0)
    return frame


# Initialize webcam
cap = initialize_webcam()
if cap is None:
    print("❌ Failed to open webcam")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

# Initialize tracking variables
last_process_time = time.time()
last_emotion = None
emotion_stability = 0
smooth_confidence = 0
fade_alpha = 0  # For text fade effect

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        frame = adjust_frame(frame)  # Adjust brightness/contrast
        current_time = time.time()

        # Process every 300ms for stability
        if current_time - last_process_time >= 0.3:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                # Get emotion and confidence
                emotions = result[0]['emotion']
                current_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                confidence = emotions[current_emotion]

                # Only process if confidence is high enough
                if confidence > 60:  # Increased confidence threshold
                    # Implement emotion stability
                    if current_emotion == last_emotion:
                        emotion_stability += 1
                        smooth_confidence = (0.8 * smooth_confidence +
                                             0.2 * confidence)
                        fade_alpha = min(fade_alpha + 0.1, 0.9)  # Fade in
                    else:
                        emotion_stability = max(
                            0, emotion_stability - 2)  # Slower reset
                        fade_alpha = max(0, fade_alpha - 0.2)  # Fade out

                    # Show emotion after 6 consistent detections
                    if emotion_stability > 5:  # Increased stability threshold
                        # Create overlay for semi-transparent background
                        overlay = frame.copy()
                        emotion_text = current_emotion.upper()
                        text = f"{emotion_text}: {smooth_confidence:.1f}%"
                        font = cv2.FONT_HERSHEY_DUPLEX
                        font_scale = 0.8  # Slightly smaller text
                        thickness = 2

                        # Position at bottom of frame
                        (text_width, text_height), _ = cv2.getTextSize(
                            text, font, font_scale, thickness)
                        # Center horizontally
                        text_x = (frame.shape[1] - text_width) // 2
                        text_y = frame.shape[0] - 30  # Bottom padding

                        # Draw semi-transparent background
                        cv2.rectangle(
                            overlay,
                            (text_x - 10, text_y - text_height - 10),
                            (text_x + text_width + 10, text_y + 10),
                            (0, 0, 0),
                            -1
                        )

                        # Apply transparency
                        cv2.addWeighted(overlay, fade_alpha,
                                        frame, 1 - fade_alpha, 0, frame)

                        # Draw text
                        cv2.putText(
                            frame,
                            text,
                            (text_x, text_y),
                            font,
                            font_scale,
                            (255, 255, 255),
                            thickness
                        )

                    last_emotion = current_emotion

                last_process_time = current_time

            except Exception:
                pass  # Silent fail for smooth operation

        cv2.imshow("Facial Expression Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 Exiting...")
            break

finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
