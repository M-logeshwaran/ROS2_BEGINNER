import os
import cv2
import random
import math
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from skimage.feature import local_binary_pattern
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. FIXED CONFIGURATION & SEEDING
# ==========================================
SEED = 42

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

seed_everything(SEED)

IMG_SIZE = 224
BATCH_SIZE = 16
FOLDS = 5

LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS

# 🔴 ROCK CLASSES
CLASSES = ["red_brick", "gravel_stone", "pebble"]
CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}

DATASET_DIR = "dataset"

# ==========================================
# 2. LBP GENERATION
# ==========================================
def generate_lbp_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    return lbp

# ==========================================
# 3. DATA GENERATOR
# ==========================================
class RobustGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, shuffle_data=True, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.augment = augment
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle_data:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_rgb, batch_lbp, batch_labels = [], [], []

        for idx in batch_indices:
            img = cv2.imread(self.image_paths[idx])
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            if self.augment:
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)
                if random.random() > 0.5:
                    img = cv2.flip(img, 0)
                k = random.randint(0, 3)
                if k > 0:
                    img = np.rot90(img, k)

            rgb = applications.efficientnet_v2.preprocess_input(img.copy())

            lbp = generate_lbp_image(img)
            lbp = (lbp - (LBP_POINTS + 2) / 2) / ((LBP_POINTS + 2) / 2)
            lbp = np.expand_dims(lbp, axis=-1)

            batch_rgb.append(rgb)
            batch_lbp.append(lbp)
            batch_labels.append(self.labels[idx])

        return {
            "rgb_input": np.array(batch_rgb),
            "lbp_input": np.array(batch_lbp)
        }, np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle_data:
            np.random.shuffle(self.indices)

# ==========================================
# 4. DUAL-BRANCH MODEL
# ==========================================
def build_advanced_model():
    # RGB BRANCH
    input_rgb = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="rgb_input")
    base_eff = applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=input_rgb
    )
    base_eff.trainable = False

    x1 = layers.GlobalAveragePooling2D()(base_eff.output)
    x1 = layers.Dropout(0.3)(x1)

    # LBP BRANCH
    input_lbp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="lbp_input")
    x2 = layers.Conv2D(32, 3, activation="relu", padding="same")(input_lbp)
    x2 = layers.MaxPooling2D(2)(x2)
    x2 = layers.Conv2D(64, 3, activation="relu", padding="same")(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)

    # FUSION
    combined = layers.Concatenate()([x1, x2])
    z = layers.Dense(256, activation="relu")(combined)
    z = layers.Dropout(0.5)(z)

    output = layers.Dense(3, activation="softmax", name="class_output")(z)

    return models.Model(inputs=[input_rgb, input_lbp], outputs=output)

# ==========================================
# 5. TRAINING (STRATIFIED K-FOLD)
# ==========================================
def run_training():
    print("Indexing dataset...")
    all_paths, all_labels = [], []

    for cls, idx in CLASS_MAP.items():
        cls_dir = os.path.join(DATASET_DIR, cls)
        for f in os.listdir(cls_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                all_paths.append(os.path.join(cls_dir, f))
                all_labels.append(idx)

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(all_labels), y=all_labels
    )
    class_weights = dict(enumerate(weights))

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr, va) in enumerate(skf.split(all_paths, all_labels)):
        print(f"\n=== Fold {fold+1}/{FOLDS} ===")

        train_gen = RobustGenerator(all_paths[tr], all_labels[tr], BATCH_SIZE, augment=True)
        val_gen = RobustGenerator(all_paths[va], all_labels[va], BATCH_SIZE)

        model = build_advanced_model()
        model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(train_gen, validation_data=val_gen, epochs=4, class_weight=class_weights)

        eff = [l for l in model.layers if "efficientnet" in l.name][0]
        eff.trainable = True
        for l in eff.layers:
            if isinstance(l, layers.BatchNormalization):
                l.trainable = False

        model.compile(
            optimizer=optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=25,
            callbacks=[
                callbacks.ModelCheckpoint(f"model_fold_{fold+1}.h5", save_best_only=True),
                callbacks.EarlyStopping(patience=6, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(patience=3)
            ],
            class_weight=class_weights
        )

        preds = model.predict(val_gen)
        y_pred = np.argmax(preds, axis=1)
        y_true = all_labels[va][:len(y_pred)]

        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=CLASSES))

        tf.keras.backend.clear_session()

# ==========================================
# 6. ENSEMBLE INFERENCE
# ==========================================
def predict_rock(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    rgb = applications.efficientnet_v2.preprocess_input(img.copy())
    lbp = generate_lbp_image(img)
    lbp = (lbp - (LBP_POINTS + 2) / 2) / ((LBP_POINTS + 2) / 2)
    lbp = np.expand_dims(lbp, axis=-1)

    rgb = np.expand_dims(rgb, 0)
    lbp = np.expand_dims(lbp, 0)

    preds = np.zeros((1, 3))
    for i in range(1, 6):
        model = models.load_model(f"model_fold_{i}.h5")
        preds += model.predict({"rgb_input": rgb, "lbp_input": lbp}, verbose=0)

    preds /= 5
    idx = np.argmax(preds)
    print(f"\nPrediction: {CLASSES[idx]} ({preds[0][idx]*100:.2f}%)")
    return CLASSES[idx]

# ==========================================
# 7. RUN
# ==========================================
if __name__ == "__main__":
    run_training()
    # predict_rock("test.jpg")
