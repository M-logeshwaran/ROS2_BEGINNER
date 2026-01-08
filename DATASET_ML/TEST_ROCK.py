import cv2
import numpy as np
import tensorflow as tf
import threading
import time
import os
import logging
from skimage.feature import local_binary_pattern
from scipy.stats import entropy
from tensorflow.keras import models, applications

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Camera
CAMERA_INDEX = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Model
IMG_SIZE = 224
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS

# 🔴 ROCK CLASSES
CLASSES = ["red_brick", "gravel_stone", "pebble"]
NUM_CLASSES = len(CLASSES)

# Confidence & Filtering
CONFIDENCE_THRESHOLD = 0.60
NORM_GAP_MIN = 0.15
MAX_ENTROPY = np.log(NUM_CLASSES)
ENTROPY_THRESHOLD = 0.85 * MAX_ENTROPY
TEMPERATURE = 1.25

# --- VALIDATION GATEKEEPER ---
MIN_FOCUS_SCORE = 60.0
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 230

# Rock color ranges (OpenCV Hue: 0–180)
ROCK_HUE_RANGES = [
    (0, 30),     # Red / Brick
    (30, 90),    # Brown / Gravel
    (90, 150),   # Grey stones
    (150, 180)   # Red wrap
]

# Threading globals
current_frame = None
latest_result = {
    "status": "INITIALIZING",
    "prediction": "---",
    "confidence": 0.0,
    "color": (128, 128, 128),
    "detail": "Loading AI..."
}
is_running = True
lock = threading.Lock()

# Face detector (safety)
try:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
except:
    face_cascade = None

# ==========================================
# 2. IMAGE PROCESSING
# ==========================================
class ImageProcessor:

    @staticmethod
    def pad_resize(image, target=224):
        h, w = image.shape[:2]
        scale = min(target / h, target / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (nw, nh))
        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        y, x = (target - nh) // 2, (target - nw) // 2
        canvas[y:y+nh, x:x+nw] = resized
        return canvas

    @staticmethod
    def generate_lbp(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
        lbp = (lbp - (LBP_POINTS + 2)/2) / ((LBP_POINTS + 2)/2)
        return lbp

    @staticmethod
    def validate_sample(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face block
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                return False, "FACE DETECTED"

        # Brightness
        mean_bright = np.mean(gray)
        if mean_bright < MIN_BRIGHTNESS:
            return False, "TOO DARK"
        if mean_bright > MAX_BRIGHTNESS:
            return False, "GLARE"

        # Blur
        focus = cv2.Laplacian(gray, cv2.CV_64F).var()
        if focus < MIN_FOCUS_SCORE:
            return False, f"BLUR ({int(focus)})"

        # Color validation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = np.mean(hsv[:, :, 0])
        sat = np.mean(hsv[:, :, 1])

        if sat < 20:
            return False, "NO COLOR"

        valid_color = any(lo <= hue <= hi for lo, hi in ROCK_HUE_RANGES)
        if not valid_color:
            return False, "INVALID MATERIAL"

        return True, "VALID"

# ==========================================
# 3. MODEL ENSEMBLE
# ==========================================
class ModelEnsemble:
    def __init__(self):
        self.models = []
        self.load()

    def load(self):
        logger.info("Loading models...")
        files = sorted(f for f in os.listdir('.') if f.startswith("model_fold_") and f.endswith(".h5"))
        if not files:
            logger.critical("No model files found")
            return

        for f in files:
            self.models.append(models.load_model(f))

        # Warm-up
        dummy = {
            "rgb_input": np.zeros((1, IMG_SIZE, IMG_SIZE, 3)),
            "lbp_input": np.zeros((1, IMG_SIZE, IMG_SIZE, 1))
        }
        self.predict(dummy)
        logger.info(f"Ensemble ready ({len(self.models)} models)")

    def predict(self, inputs):
        logits = np.zeros((1, NUM_CLASSES))
        for m in self.models:
            p = m.predict(inputs, verbose=0) / TEMPERATURE
            logits += np.log(p + 1e-9)
        logits /= len(self.models)
        return tf.nn.softmax(logits, axis=1).numpy()[0]

# ==========================================
# 4. AI WORKER THREAD
# ==========================================
def worker():
    global latest_result
    ensemble = ModelEnsemble()

    if not ensemble.models:
        latest_result = {
            "status": "ERROR",
            "prediction": "NO MODEL",
            "confidence": 0,
            "color": (0, 0, 255),
            "detail": "MODEL LOAD FAILED"
        }
        return

    while is_running:
        time.sleep(0.05)

        with lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()

        h, w = frame.shape[:2]
        d = min(h, w)
        crop = frame[(h-d)//2:(h+d)//2, (w-d)//2:(w+d)//2]

        valid, msg = ImageProcessor.validate_sample(crop)
        if not valid:
            latest_result = {
                "status": "INVALID",
                "prediction": "---",
                "confidence": 0,
                "color": (128, 128, 128),
                "detail": msg
            }
            continue

        img = ImageProcessor.pad_resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        rgb = np.expand_dims(applications.efficientnet_v2.preprocess_input(img.copy()), 0)
        lbp = np.expand_dims(ImageProcessor.generate_lbp(img), (0, -1))

        probs = ensemble.predict({"rgb_input": rgb, "lbp_input": lbp})

        top1, top2 = probs.argsort()[::-1][:2]
        p1, p2 = probs[top1], probs[top2]
        gap = (p1 - p2) / (p1 + 1e-9)
        ent = entropy(probs)

        status, detail, color = "SUCCESS", "CONFIDENT", (0, 255, 0)
        if p1 < CONFIDENCE_THRESHOLD:
            status, detail, color = "UNCERTAIN", "LOW CONFIDENCE", (0, 255, 255)
        elif gap < NORM_GAP_MIN:
            status, detail, color = "UNCERTAIN", f"AMBIGUOUS vs {CLASSES[top2]}", (0, 255, 255)
        elif ent > ENTROPY_THRESHOLD:
            status, detail, color = "UNCERTAIN", "HIGH ENTROPY", (0, 255, 255)

        latest_result = {
            "status": status,
            "prediction": CLASSES[top1].replace("_", " ").upper(),
            "confidence": float(p1),
            "color": color,
            "detail": detail
        }

# ==========================================
# 5. UI LOOP
# ==========================================
def main():
    global current_frame, is_running

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    threading.Thread(target=worker, daemon=True).start()
    print("🪨 ROCK CLASSIFICATION LIVE — Press Q to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with lock:
            current_frame = frame.copy()
            res = latest_result.copy()

        h, w = frame.shape[:2]
        d = min(h, w)
        x1, y1 = (w-d)//2, (h-d)//2
        x2, y2 = x1+d, y1+d

        cv2.rectangle(frame, (x1,y1), (x2,y2), res["color"], 2)
        cv2.rectangle(frame, (0,0), (w,100), (0,0,0), -1)

        cv2.putText(frame, res["prediction"], (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, res["color"], 3)
        cv2.putText(frame,
                    f"{res['detail']} | {res['confidence']*100:.1f}%",
                    (20,85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("Rock Analysis - Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    is_running = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
