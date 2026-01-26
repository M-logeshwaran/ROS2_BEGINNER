import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import Counter
from skimage.feature import local_binary_pattern

# ================= PATHS =================
YOLO_MODEL = r"C:\Users\Sanjay\Downloads\ROCK_FINAL\best.pt"
H5_MODEL   = r"C:\Users\Sanjay\Downloads\ROCK_FINAL\rock_classifier_best_final_incu.h5"
INPUT_IMG  = r"C:\Users\Sanjay\Downloads\ROCK_FINAL\dataset_test\dataset_origin\red_brick\img_000172.jpg"
OUTPUT_IMG = r"C:\Users\Sanjay\Downloads\ROCK_FINAL\output.jpg"

# ================= CONFIG =================
GRID_SIZE = 4
IMG_SIZE = 224
rock_counter = Counter()

CLASS_NAMES = ["gravel", "pebble", "red_brick", "unknown"]

# ================= LOAD MODELS =================
yolo = YOLO(YOLO_MODEL)
clf = load_model(H5_MODEL)

# ================= IMAGE =================
img = cv2.imread(INPUT_IMG)
if img is None:
    raise RuntimeError("❌ Image load failed")

H, W, _ = img.shape
tile_h = H // GRID_SIZE
tile_w = W // GRID_SIZE

# ================= FEATURE EXTRACTION =================
def extract_color(img):
    img = img.astype("float32") / 255.0
    mean_rgb = np.mean(img, axis=(0,1))
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV) / 255.0
    mean_hsv = np.mean(hsv, axis=(0,1))
    return np.concatenate([mean_rgb, mean_hsv])

def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0,10), density=True)
    return hist.astype("float32")

def preprocess_all(crop):
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    img = crop.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    color = np.expand_dims(extract_color(crop), axis=0)
    lbp   = np.expand_dims(extract_lbp(crop), axis=0)

    return [img, color, lbp]

# ================= 5 JUDGES + SMART LOGIC =================
def five_judges(crop):
    h, w, _ = crop.shape

    judges = [
        crop,
        crop[int(0.1*h):int(0.9*h), int(0.1*w):int(0.9*w)],
        cv2.flip(crop, 1),
        cv2.convertScaleAbs(crop, alpha=1.15, beta=15),
        cv2.convertScaleAbs(crop, alpha=0.9, beta=-15)
    ]

    votes, confs = [], []

    for j in judges:
        j = cv2.resize(j, (w, h))
        pred = clf.predict(preprocess_all(j), verbose=0)
        idx = int(np.argmax(pred))
        votes.append(idx)
        confs.append(float(pred[0][idx]))

    final_idx = Counter(votes).most_common(1)[0][0]
    avg_conf = float(np.mean(confs))
    label = CLASS_NAMES[final_idx]

    # ===== SMART OVERRIDE (SAFE) =====
    if label == "red_brick" and avg_conf < 0.85:

        mean_bgr = np.mean(crop, axis=(0,1))
        red_ratio = mean_bgr[2] / (mean_bgr[1] + 1e-6)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if red_ratio < 1.05 and lap_var > 260:
            label = "gravel"

    if avg_conf < 0.45:
        label = "unknown"

    return label, avg_conf

# ================= MAIN PIPELINE =================
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):

        y1, y2 = r * tile_h, (r + 1) * tile_h
        x1, x2 = c * tile_w, (c + 1) * tile_w
        tile = img[y1:y2, x1:x2]

        results = yolo(tile, conf=0.15, iou=0.4)[0]
        if results.boxes is None:
            continue

        for box in results.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            bx1 += x1; bx2 += x1
            by1 += y1; by2 += y1

            crop = img[by1:by2, bx1:bx2]
            if crop.size == 0:
                continue

            label, conf = five_judges(crop)
            rock_counter[label] += 1

            # ALWAYS draw box
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (0,255,0), 2)

            # 🔥 DRAW TEXT ONLY IF NOT UNKNOWN
            if label != "unknown":
                cv2.putText(
                    img,
                    f"{label} {conf:.2f}",
                    (bx1, by1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

# ================= OUTPUT =================
cv2.imwrite(OUTPUT_IMG, img)

print("\n========== ROCK COUNT SUMMARY ==========")
total = 0
for cls in CLASS_NAMES:
    cnt = rock_counter.get(cls, 0)
    print(f"{cls:12s}: {cnt}")
    total += cnt

print("----------------------------------------")
print(f"TOTAL ROCKS  : {total}")
print("========================================")
print("DONE – Production pipeline success")

