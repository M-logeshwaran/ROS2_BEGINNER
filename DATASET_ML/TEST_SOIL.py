import cv2
import numpy as np
import tensorflow as tf
import threading
import time
import os
import sys
import logging
from collections import OrderedDict
from skimage.feature import local_binary_pattern
from scipy.stats import entropy
from tensorflow.keras import models, applications

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardware
CAMERA_INDEX = 1        # 0 for Webcam, 1 for USB Microscope (Change if black screen)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Model Constraints
IMG_SIZE = 224
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
CLASSES = ["alluvial_soil", "black_soil", "laterite_soil", "red_soil"]
NUM_CLASSES = len(CLASSES)

# Confidence & Thresholds
CONFIDENCE_THRESHOLD = 0.60
NORM_GAP_MIN = 0.15 
MAX_ENTROPY = np.log(NUM_CLASSES)
ENTROPY_THRESHOLD = 0.85 * MAX_ENTROPY 
TEMPERATURE = 1.25

# --- VALIDATION GATEKEEPER CONFIG ---
MIN_FOCUS_SCORE = 60.0      # Reject blur
MIN_BRIGHTNESS = 40         # Reject dark
MAX_BRIGHTNESS = 230        # Reject glare
SOIL_HUE_RANGES = [         # Earth tones only (OpenCV Hue 0-180)
    (0, 30),    # Red/Orange/Yellow
    (150, 180)  # Red wrap-around
]

# Globals for Threading
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

# Load Face Detector (Safety Nuke)
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except:
    logger.warning("Face detector XML not found. Face check will be disabled.")
    face_cascade = None

# ==========================================
# 2. IMAGE PROCESSING ENGINE
# ==========================================
class ImageProcessor:
    @staticmethod
    def pad_resize(image, target_size=224):
        h, w = image.shape[:2]
        scale = min(target_size/h, target_size/w)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(image, (nw, nh))
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        x_off = (target_size - nw) // 2
        y_off = (target_size - nh) // 2
        canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
        return canvas

    @staticmethod
    def generate_lbp(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
        lbp = (lbp - (LBP_POINTS + 2)/2) / ((LBP_POINTS + 2)/2)
        return lbp

    @staticmethod
    def validate_sample(img):
        """
        The 'Gatekeeper': Rejects blur, faces, and non-soil colors.
        Returns: (IsValid, StatusMessage)
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # A. Face Check
        if face_cascade:
            small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(small_gray, 1.3, 5)
            if len(faces) > 0: return False, "FACE DETECTED"

        # B. Brightness
        avg_bright = np.mean(gray)
        if avg_bright < MIN_BRIGHTNESS: return False, "TOO DARK"
        if avg_bright > MAX_BRIGHTNESS: return False, "GLARE DETECTED"

        # C. Focus (Blur)
        focus = cv2.Laplacian(gray, cv2.CV_64F).var()
        if focus < MIN_FOCUS_SCORE: return False, f"FOCUS: {int(focus)}"

        # D. Color Check (Earth Tones)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = np.mean(hsv[:,:,0])
        sat = np.mean(hsv[:,:,1])
        if sat < 20: return False, "GRAYSCALE / NO COLOR"
        
        is_earth = False
        for (lower, upper) in SOIL_HUE_RANGES:
            if lower <= hue <= upper:
                is_earth = True
                break
        if not is_earth: return False, "INVALID COLOR"

        return True, "VALID"

# ==========================================
# 3. MODEL MANAGER (Titanium Engine)
# ==========================================
class ModelEnsemble:
    def __init__(self):
        self.models = []
        self.load_models()

    def load_models(self):
        logger.info("⚡ Loading AI Models...")
        model_files = sorted([f for f in os.listdir('.') if f.startswith('model_fold_') and f.endswith('.h5')])
        
        if not model_files:
            logger.critical("❌ No models found! (model_fold_*.h5)")
            return

        for f in model_files:
            try:
                m = models.load_model(f)
                self.models.append(m)
            except Exception as e:
                logger.warning(f"Skipped {f}: {e}")
        
        # Warmup
        if self.models:
            dummy_rgb = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
            dummy_lbp = np.zeros((1, IMG_SIZE, IMG_SIZE, 1))
            self.predict({"rgb_input": dummy_rgb, "lbp_input": dummy_lbp})
            logger.info(f"✅ Ensemble Ready ({len(self.models)} models)")

    def predict(self, inputs):
        if not self.models: return np.zeros(NUM_CLASSES)
        
        accumulated_log = np.zeros((1, NUM_CLASSES))
        for m in self.models:
            preds = m.predict(inputs, verbose=0)
            preds = preds / TEMPERATURE
            accumulated_log += np.log(preds + 1e-9)
            
        avg_log = accumulated_log / len(self.models)
        return tf.nn.softmax(avg_log, axis=1).numpy()[0]

# ==========================================
# 4. BACKGROUND WORKER (The "Brain")
# ==========================================
def worker():
    global latest_result
    
    # Initialize Engine inside thread to keep UI fast
    ensemble = ModelEnsemble()
    if not ensemble.models:
        with lock:
            latest_result = {"status": "ERROR", "detail": "NO MODELS FOUND", "color": (0,0,255), "prediction": "ERROR", "confidence": 0}
        return

    while is_running:
        time.sleep(0.05) # Limit AI loop speed (approx 20 FPS max)
        
        # Snapshot frame safely
        with lock:
            if current_frame is None: continue
            frame_copy = current_frame.copy()

        # 1. Center Crop
        h, w = frame_copy.shape[:2]
        dim = min(h, w)
        crop = frame_copy[(h-dim)//2:(h+dim)//2, (w-dim)//2:(w+dim)//2]

        # 2. Gatekeeper Validation
        is_valid, msg = ImageProcessor.validate_sample(crop)

        if not is_valid:
            # Update Status (Skip heavy AI)
            with lock:
                latest_result = {
                    "status": "INVALID",
                    "prediction": "---",
                    "confidence": 0.0,
                    "color": (0, 165, 255), # Orange/Yellow
                    "detail": msg
                }
            continue

        # 3. AI Inference
        try:
            # Preprocess
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            processed_img = ImageProcessor.pad_resize(img_rgb, IMG_SIZE)
            
            rgb_batch = np.expand_dims(applications.efficientnet_v2.preprocess_input(processed_img.copy()), axis=0)
            lbp = ImageProcessor.generate_lbp(processed_img)
            lbp_batch = np.expand_dims(np.expand_dims(lbp, axis=-1), axis=0)

            # Predict
            probs = ensemble.predict({"rgb_input": rgb_batch, "lbp_input": lbp_batch})
            
            # Post-Process
            top_idx = probs.argsort()[::-1][0]
            top2_idx = probs.argsort()[::-1][1]
            
            p1 = probs[top_idx]
            p2 = probs[top2_idx]
            norm_gap = (p1 - p2) / (p1 + 1e-9)
            entr = entropy(probs)
            
            pred_class = CLASSES[top_idx]
            
            # Decision Logic
            status = "SUCCESS"
            detail = "CONFIDENT"
            color = (0, 255, 0) # Green

            if p1 < CONFIDENCE_THRESHOLD:
                status = "UNCERTAIN"
                detail = "LOW CONFIDENCE"
                color = (0, 255, 255) # Yellow
            elif norm_gap < NORM_GAP_MIN:
                status = "UNCERTAIN"
                detail = f"AMBIGUOUS (vs {CLASSES[top2_idx]})"
                color = (0, 255, 255)
            elif entr > ENTROPY_THRESHOLD:
                status = "UNCERTAIN"
                detail = "CONFUSED (High Entropy)"
                color = (0, 255, 255)

            # Update Global Result
            with lock:
                latest_result = {
                    "status": status,
                    "prediction": pred_class.replace("_", " ").upper(),
                    "confidence": float(p1),
                    "color": color,
                    "detail": detail
                }

        except Exception as e:
            logger.error(f"Inference Error: {e}")

# ==========================================
# 5. MAIN UI LOOP
# ==========================================
def main():
    global current_frame, is_running
    
    print("\n🎥 STARTING CAMERA...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ Error: Camera {CAMERA_INDEX} not found. Try changing CAMERA_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Start Background AI Worker
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()

    print("✅ SYSTEM LIVE. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Send to worker
        with lock:
            current_frame = frame.copy()
            res = latest_result # Copy safely

        display = frame.copy()
        h, w = display.shape[:2]
        dim = min(h, w)
        
        # --- UI DRAWING ---
        
        # 1. Scanning Zone Box
        x1, y1 = (w - dim) // 2, (h - dim) // 2
        x2, y2 = x1 + dim, y1 + dim
        
        # Box color depends on status
        box_color = res["color"]
        if res["status"] == "INVALID": box_color = (128, 128, 128) # Grey for invalid
        
        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
        
        # 2. Top Info Banner
        cv2.rectangle(display, (0, 0), (w, 100), (0, 0, 0), -1)
        
        if res["status"] == "INVALID":
            # Show "Invalid" Warning
            cv2.putText(display, f"⚠️ {res['detail']}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        elif res["status"] == "INITIALIZING":
             cv2.putText(display, "⏳ LOADING MODELS...", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            # Show Prediction
            cv2.putText(display, f"{res['prediction']}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, box_color, 3)
            
            conf_text = f"Confidence: {res['confidence']*100:.1f}% | {res['detail']}"
            cv2.putText(display, conf_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 3. Debug Overlay (Optional)
        cv2.putText(display, "Press 'q' to Exit", (w - 150, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Soil Analysis - Live Microscope", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    is_running = False
    cap.release()
    cv2.destroyAllWindows()
    print("👋 System Shutdown.")

if __name__ == "__main__":
    main()
