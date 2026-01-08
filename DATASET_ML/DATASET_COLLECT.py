import cv2
import os
import time

# ========= SETTINGS =========
SAVE_DIR = "dataset_images"
CAMERA_INDEX = 1
TARGET_FPS = 10                  
BLUR_THRESHOLD = 120.0            # <-- BLUR CONTROL (IMPORTANT)
# ============================

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Camera not detected")
    exit()

image_count = 0
recording = False
prev_key = None
last_save_time = 0
save_interval = 1.0 / TARGET_FPS

print("Dataset capture ready")
print("Press 's' to START / STOP saving")
print("Press 'q' to quit")

def is_image_sharp(frame, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_value, blur_value >= threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()

    # Toggle recording
    if key == ord('s') and prev_key != ord('s'):
        recording = not recording
        print(" Recording ON" if recording else "⏸ Recording OFF")

    # Save frames at 10 FPS if sharp
    if recording and (current_time - last_save_time) >= save_interval:
        blur_value, sharp = is_image_sharp(frame, BLUR_THRESHOLD)

        if sharp:
            filename = f"img_{image_count:06d}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR, filename), frame)
            image_count += 1
            last_save_time = current_time
            status = "SAVED"
            color = (0, 255, 0)
        else:
            status = "BLUR"
            color = (0, 0, 255)

        cv2.putText(frame,
                    f"{status} | Blur: {blur_value:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

    cv2.imshow("Dataset Capture", frame)

    if key == ord('q'):
        break

    prev_key = key

cap.release()
cv2.destroyAllWindows()
print("Camera closed")
