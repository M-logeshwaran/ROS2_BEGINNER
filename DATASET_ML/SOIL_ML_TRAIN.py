import os
import cv2
import random
import math
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models, applications, optimizers, callbacks, losses
from skimage.feature import local_binary_pattern
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. FIXED CONFIGURATION & SEEDING
# ==========================================
# Ensuring full reproducibility across runs
SEED = 42
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

seed_everything(SEED)

IMG_SIZE = 224
BATCH_SIZE = 16  # Smaller batch = better generalization
FOLDS = 5        # Standard 5-Fold Cross Validation
LBP_RADIUS = 3   # Radius for texture detection
LBP_POINTS = 8 * LBP_RADIUS
CLASSES = ["alluvial_soil", "black_soil", "laterite_soil", "red_soil"]
CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}
DATASET_DIR = "dataset" # Folder name containing subfolders of soil types

# ==========================================
# 2. ROBUST DATA GENERATOR
# ==========================================
def generate_lbp_image(image):
    """
    Generates Local Binary Pattern (LBP) for texture analysis.
    Uses 'uniform' method to detect edges/corners vs flat areas.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    return lbp

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
        # Uses ceil() to ensure NO data is dropped during validation/training
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, index):
        # Handle the last batch which might be smaller
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_rgb = []
        batch_lbp = []
        batch_labels = []

        for idx in batch_indices:
            path = self.image_paths[idx]
            label = self.labels[idx]
            
            img = cv2.imread(path)
            if img is None: continue 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # --- AUGMENTATION (RGB FIRST) ---
            # We augment the RGB image first, so the texture (LBP)
            # calculated afterwards matches the rotated/flipped visual.
            if self.augment:
                if random.random() > 0.5: img = cv2.flip(img, 1) # Horizontal Flip
                if random.random() > 0.5: img = cv2.flip(img, 0) # Vertical Flip
                k = random.randint(0, 3) 
                if k > 0: img = np.rot90(img, k)

            # --- PREPROCESSING ---
            # 1. RGB Branch: EfficientNetV2 standard preprocessing
            rgb_proc = applications.efficientnet_v2.preprocess_input(img.copy())
            
            # 2. Texture Branch: Compute LBP on the AUGMENTED image
            lbp = generate_lbp_image(img)
            
            # Zero-mean Normalization (Crucial for fast convergence)
            # Centers the data around 0 instead of 0.5
            lbp = (lbp - (LBP_POINTS + 2)/2) / ((LBP_POINTS + 2)/2) 
            lbp = np.expand_dims(lbp, axis=-1)

            batch_rgb.append(rgb_proc)
            batch_lbp.append(lbp)
            batch_labels.append(label)

        return {"rgb_input": np.array(batch_rgb), "lbp_input": np.array(batch_lbp)}, np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle_data:
            np.random.shuffle(self.indices)

# ==========================================
# 3. ADVANCED DUAL-BRANCH MODEL
# ==========================================
def build_advanced_model():
    # --- Branch 1: RGB Backbone (Color & Shape) ---
    input_rgb = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="rgb_input")
    
    # EfficientNetV2B0 is newer and faster than B0.
    # Start FROZEN. We will unfreeze specific layers in Stage 2.
    base_eff = applications.EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=input_rgb)
    base_eff.trainable = False 
    
    x1 = layers.GlobalAveragePooling2D()(base_eff.output)
    x1 = layers.Dropout(0.3)(x1)

    # --- Branch 2: Texture CNN (Micro-features) ---
    # Specifically targets Laterite Pores vs Red Soil Smoothness
    input_lbp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="lbp_input")
    x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_lbp)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)
    
    # --- Fusion & Head ---
    combined = layers.Concatenate()([x1, x2])
    z = layers.Dense(256, activation='relu')(combined)
    z = layers.Dropout(0.5)(z) # High dropout for ensemble diversity
    
    output = layers.Dense(4, activation='softmax', name="class_output")(z)

    model = models.Model(inputs=[input_rgb, input_lbp], outputs=output)
    return model

# ==========================================
# 4. MAIN TRAINING LOOP (Stratified K-Fold)
# ==========================================

def run_training():
    print("--- 1. Indexing Data ---")
    all_paths = []
    all_labels = []

    for class_name, label_id in CLASS_MAP.items():
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Error: Missing directory {class_dir}")
            continue
            
        files = os.listdir(class_dir)
        count = 0
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                all_paths.append(os.path.join(class_dir, f))
                all_labels.append(label_id)
                count += 1
        print(f"Found {count} images for {class_name}")

    if len(all_paths) == 0:
        print("No images found! Check dataset structure.")
        return

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    # Calculate Class Weights to handle imbalance automatically
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(all_labels), y=all_labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Computed Class Weights: {class_weights_dict}")

    # --- Stratified K-Fold Loop ---
    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_paths, all_labels)):
        print(f"\n=== Training Fold {fold+1}/{FOLDS} ===")
        
        X_train, X_val = all_paths[train_idx], all_paths[val_idx]
        y_train, y_val = all_labels[train_idx], all_labels[val_idx]
        
        train_gen = RobustGenerator(X_train, y_train, BATCH_SIZE, augment=True)
        val_gen = RobustGenerator(X_val, y_val, BATCH_SIZE, augment=False)
        
        # --- STAGE 1: Warmup (Head Only) ---
        print(">> Stage 1: Warming up head (Backbone Frozen)...")
        model = build_advanced_model()
        
        model.compile(optimizer=optimizers.Adam(1e-3), 
                      loss="sparse_categorical_crossentropy", 
                      metrics=['accuracy'])
        
        model.fit(train_gen, validation_data=val_gen, epochs=4, verbose=1, class_weight=class_weights_dict)
        
        # --- STAGE 2: Fine-Tuning (Unfreeze Top Layers) ---
        print(">> Stage 2: Fine-Tuning (Unfreezing Backbone)...")
        
        # Unfreeze the EfficientNet block
        eff_layer = model.get_layer(index=1) # The second layer is usually the backbone
        if "efficientnet" not in eff_layer.name:
             # Safety check: find layer by name if index shifted
             eff_layer = [l for l in model.layers if "efficientnet" in l.name][0]

        eff_layer.trainable = True
        
        # CRITICAL: Freeze BatchNormalization layers inside EfficientNet
        # If we don't do this, the small batch size will ruin the pretrained stats.
        for layer in eff_layer.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
                
        # Lower learning rate significantly (10x lower)
        model.compile(
            optimizer=optimizers.Adam(1e-4), 
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy']
        )
        
        checkpoint = callbacks.ModelCheckpoint(f"model_fold_{fold+1}.h5", save_best_only=True, monitor="val_accuracy")
        early_stop = callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=25,
            callbacks=[checkpoint, early_stop, reduce_lr],
            class_weight=class_weights_dict
        )
        
        # --- EVALUATION ---
        print(f"--- Evaluating Fold {fold+1} Performance ---")
        # Re-create Val Generator without shuffle for Confusion Matrix alignment
        val_gen_eval = RobustGenerator(X_val, y_val, BATCH_SIZE, shuffle_data=False, augment=False)
        val_preds = model.predict(val_gen_eval)
        val_pred_classes = np.argmax(val_preds, axis=1)
        
        # Extract true labels from generator
        val_true_labels = []
        for i in range(len(val_gen_eval)):
            _, labs = val_gen_eval[i]
            val_true_labels.extend(labs)
        val_true_labels = np.array(val_true_labels)

        # Print Matrices
        print("Confusion Matrix:")
        print(confusion_matrix(val_true_labels, val_pred_classes))
        print("\nClassification Report:")
        print(classification_report(val_true_labels, val_pred_classes, target_names=CLASSES))
        
        tf.keras.backend.clear_session()

# ==========================================
# 5. INFERENCE FUNCTION (Ensemble)
# ==========================================
def predict_soil(image_path):
    """
    Loads all 5 trained models and averages their predictions.
    This reduces variance and boosts accuracy significantly.
    """
    if not os.path.exists("model_fold_1.h5"):
        print("Error: Models not found. Run training first.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    
    # Preprocess
    rgb = applications.efficientnet_v2.preprocess_input(img_resized.copy())
    lbp = generate_lbp_image(img_resized)
    lbp = (lbp - (LBP_POINTS + 2)/2) / ((LBP_POINTS + 2)/2)
    lbp = np.expand_dims(lbp, axis=-1)
    
    # Prepare Batches
    rgb_batch = np.expand_dims(rgb, axis=0)
    lbp_batch = np.expand_dims(lbp, axis=0)
    inputs = {"rgb_input": rgb_batch, "lbp_input": lbp_batch}
    
    # Ensemble Prediction
    total_preds = np.zeros((1, 4))
    
    print("\nRunning Ensemble Prediction...")
    valid_models = 0
    for i in range(1, 6):
        model_name = f"model_fold_{i}.h5"
        if os.path.exists(model_name):
            print(f"Loading {model_name}...")
            model = models.load_model(model_name)
            pred = model.predict(inputs, verbose=0)
            total_preds += pred
            valid_models += 1
            
    if valid_models == 0: return
    
    avg_pred = total_preds / valid_models
    class_idx = np.argmax(avg_pred)
    confidence = avg_pred[0][class_idx] * 100
    
    print(f"\nResult: {CLASSES[class_idx]} ({confidence:.2f}%)")
    return CLASSES[class_idx]

# ==========================================
# 6. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # 1. Train the models
    run_training()
    
    # 2. Example Inference (Uncomment to test a single image after training)
    # predict_soil("test_image.jpg")
