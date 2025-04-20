import cv2
import random
import time
import sys


def debug_camera():
    """Debug camera initialization and properties"""
    print("\n🎥 Initializing camera...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try DirectShow backend
        if not cap.isOpened():
            print("❌ Initial camera open failed with DirectShow")
            # Fallback to default backend
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Camera open failed with default backend")
                return None
    except Exception as e:
        print(f"❌ Error initializing camera: {e}")
        return None

    # Check camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    backend = cap.getBackendName()
    print(f"📊 Camera backend: {backend}")
    print(f"📊 Resolution: {width}x{height} @ {fps}fps")

    # Test frame capture
    print("📸 Testing frame capture...")
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Could not read test frame")
        cap.release()
        return None

    print(f"✅ Frame capture successful - Shape: {frame.shape}")
    return cap


# Initialize camera with debug info
cap = debug_camera()
if cap is None:
    print("Camera initialization failed. Check if:")
    print("1. Another program is using the camera")
    print("2. Camera drivers are properly installed")
    print("3. Camera is properly connected")
    exit(1)

print("📸 Press 'q' to quit and take screenshots.")

# Labels for fake predictions
hand_signs = ['A']
expressions = ['neutral']

# Add frame counter for debugging
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Frame capture failed at frame {frame_count}")
        break

    frame_count += 1
    if frame_count % 30 == 0:  # Log FPS every 30 frames
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        print(f"📊 FPS: {fps:.1f}")

    frame = cv2.flip(frame, 1)

    # Fake prediction results
    fake_hand_sign = random.choice(hand_signs)
    fake_expression = random.choice(expressions)

    # Draw hand ROI
    hx1, hy1, hx2, hy2 = 100, 100, 300, 300
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
    cv2.putText(frame, f"Sign: {fake_hand_sign}", (hx1, hy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw face ROI
    fx1, fy1, fx2, fy2 = 350, 100, 550, 300
    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
    cv2.putText(frame, f"Expression: {fake_expression}", (fx1, fy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("HandySense Demo", frame)

    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
