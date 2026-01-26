import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern

# ================= CONFIG =================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 45
DATASET_DIR = r"C:\Users\Sanjay\Downloads\ROCK_FINAL\dataset _used\dataset"
NUM_CLASSES = 4

# ================= FEATURE EXTRACTORS =================
def extract_color(img):
    img = img.astype("float32") / 255.0
    mean_rgb = np.mean(img, axis=(0,1))
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV) / 255.0
    mean_hsv = np.mean(hsv, axis=(0,1))
    return np.concatenate([mean_rgb, mean_hsv])

def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0,10), density=True)
    return hist.astype("float32")

# ================= DATA GENERATOR =================
class FusionGenerator(tf.keras.utils.Sequence):
    def __init__(self, root, batch, subset, aug, cls_idx):
        self.data = []
        self.batch = batch
        self.aug = aug
        self.cls_idx = cls_idx

        for c in os.listdir(root):
            p = os.path.join(root, c)
            if not os.path.isdir(p): continue
            for f in os.listdir(p):
                self.data.append((os.path.join(p, f), cls_idx[c]))

        np.random.shuffle(self.data)
        split = int(0.8 * len(self.data))
        self.data = self.data[:split] if subset=="train" else self.data[split:]

    def __len__(self):
        return int(np.ceil(len(self.data)/self.batch))

    def __getitem__(self, i):
        batch = self.data[i*self.batch:(i+1)*self.batch]
        imgs, colors, lbps, labels = [], [], [], []

        for path, lab in batch:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = self.aug.random_transform(img)

            imgs.append(img/255.0)
            colors.append(extract_color(img))
            lbps.append(extract_lbp(img))

            y = np.zeros(NUM_CLASSES)
            y[lab] = 1
            labels.append(y)

        return (np.array(imgs), np.array(colors), np.array(lbps)), np.array(labels)

# ================= AUGMENTATION (BALANCED) =================
datagen = ImageDataGenerator(
    rotation_range=60,
    zoom_range=0.25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True
)

# ================= CLASSES =================
class_names = sorted(os.listdir(DATASET_DIR))
class_indices = {n:i for i,n in enumerate(class_names)}

train_gen = FusionGenerator(DATASET_DIR, BATCH_SIZE, "train", datagen, class_indices)
val_gen   = FusionGenerator(DATASET_DIR, BATCH_SIZE, "val", datagen, class_indices)

# ================= MODEL =================
base = MobileNetV2(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

img_in   = Input((IMG_SIZE, IMG_SIZE, 3))
color_in = Input((6,))
lbp_in   = Input((10,))

x = base(img_in)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

# 🔑 Fusion of both logics
x = Concatenate()([x, color_in, lbp_in])

x = Dense(512, activation="relu")(x)
x = Dropout(0.45)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.35)(x)

out = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model([img_in, color_in, lbp_in], out)

# ================= COMPILE =================
model.compile(
    optimizer=Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.04),
    metrics=["accuracy"]
)

# ================= CALLBACKS =================
callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.3),
    ModelCheckpoint("rock_classifier_best.h5", save_best_only=True)
]

# ================= TRAIN =================
model.fit(train_gen, validation_data=val_gen,
          epochs=EPOCHS, callbacks=callbacks)

# ================= FINE TUNE =================
base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_gen, validation_data=val_gen,
          epochs=15, callbacks=callbacks)

model.save("rock_classifier_final.h5")
print("✅ Fusion training complete – gravel + red_brick optimized")
