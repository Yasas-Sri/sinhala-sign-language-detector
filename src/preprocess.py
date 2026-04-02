# """
# preprocess_paper.py -- SSL400 Replicated Pipeline
# Optimized for Hybrid CNN-BiLSTM + Attention as per Abhishek et al. (2025)
# """

# import os
# import ast
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# # ─────────────────────────────────────────────────────────
# #  CONFIGURATION (Paper Specs)
# # ─────────────────────────────────────────────────────────
# DATASET_ROOT   = "./Dataset - MP - CSV"
# OUTPUT_DIR     = "ssl400_processed_paper"

# # The paper evaluates 10, 100, and 300 classes [cite: 108]
# # To match your previous run, we'll keep the filter at 10 samples
# MIN_SAMPLES_PER_CLASS = 2 

# TARGET_FRAMES  = 30    # Key Spec: Paper uses K=30 
# N_LANDMARKS    = 33    # MediaPipe Holistic landmarks [cite: 42, 113]
# FEATURE_DIM    = 99    # 33 landmarks * (x, y, z) 

# TRAIN_RATIO    = 0.70
# VAL_RATIO      = 0.10
# TEST_RATIO     = 0.20
# RANDOM_SEED    = 42

# # ══════════════════════════════════════════════════════════
# #  STEP 1 -- SELECTIVE TEMPORAL SAMPLING
# # ══════════════════════════════════════════════════════════

# def selective_temporal_sampling(seq, k=TARGET_FRAMES):
#     """
#     Computes total frames N and selects K evenly spaced frames.
#     This preserves gesture dynamics while reducing frame redundancy[cite: 46, 187].
#     """
#     n = len(seq)
#     if n == 0: 
#         return np.zeros((k, FEATURE_DIM), dtype=np.float32)
    
#     # Generate K indices evenly spaced from 0 to N-1 
#     indices = np.linspace(0, n - 1, k).astype(int)
#     return seq[indices]

# # ══════════════════════════════════════════════════════════
# #  STEP 2 -- DATA DISCOVERY & CSV PARSING
# # ══════════════════════════════════════════════════════════

# def collect_manifest(root, min_samples):
#     records = []
#     for category in sorted(os.listdir(root)):
#         cat_path = os.path.join(root, category)
#         if not os.path.isdir(cat_path): continue
#         for word in sorted(os.listdir(cat_path)):
#             word_path = os.path.join(cat_path, word)
#             if not os.path.isdir(word_path): continue
            
#             files = [os.path.join(word_path, f) for f in os.listdir(word_path) if f.endswith(".csv")]
#             if len(files) >= min_samples:
#                 for fp in files:
#                     records.append({"filepath": fp, "label": word})
#     return pd.DataFrame(records)

# def parse_csv_to_array(filepath):
#     """Parses MediaPipe landmarks and returns sampled sequence."""
#     df = pd.read_csv(filepath)
#     frames = []
#     # Paper focuses on skeletal keypoints [cite: 42, 48]
#     for _, row in df.iterrows():
#         coords = []
#         for col in df.columns[:N_LANDMARKS]:
#             try:
#                 # Extract x, y, z from string format '[x, y, z, v]'
#                 vals = ast.literal_eval(str(row[col]).strip())
#                 coords.extend(vals[:3])
#             except:
#                 coords.extend([0.0, 0.0, 0.0])
#         frames.append(coords)
    
#     seq = np.array(frames, dtype=np.float32)
#     return selective_temporal_sampling(seq)

# # ══════════════════════════════════════════════════════════
# #  STEP 3 -- NORMALIZATION & SPLITTING
# # ══════════════════════════════════════════════════════════

# def process_and_save():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     df = collect_manifest(DATASET_ROOT, MIN_SAMPLES_PER_CLASS)
    
#     # Stratified Split
#     le = LabelEncoder()
#     df['label_id'] = le.fit_transform(df['label'])
#     np.save(os.path.join(OUTPUT_DIR, "label_classes.npy"), le.classes_)

#     # Group by label to ensure even distribution
#     train_list, val_list, test_list = [], [], []
#     for _, group in df.groupby('label_id'):
#         group = group.sample(frac=1, random_state=RANDOM_SEED)
#         n = len(group)
#         n_train = int(n * TRAIN_RATIO)
#         n_val = int(n * VAL_RATIO)
        
#         train_list.append(group.iloc[:n_train])
#         val_list.append(group.iloc[n_train : n_train + n_val])
#         test_list.append(group.iloc[n_train + n_val :])

#     train_df = pd.concat(train_list)
#     val_df = pd.concat(val_list)
#     test_df = pd.concat(test_list)

#     # Load and Normalize
#     def load_set(manifest, name):
#         X, y = [], []
#         for i, row in manifest.iterrows():
#             arr = parse_csv_to_array(row['filepath'])
#             X.append(arr)
#             y.append(row['label_id'])
#             if i % 100 == 0: print(f"Processing {name}... {i}/{len(manifest)}", end="\r")
#         return np.stack(X), np.array(y)

#     X_train, y_train = load_set(train_df, "Train")
#     X_val, y_val = load_set(val_df, "Val")
#     X_test, y_test = load_set(test_df, "Test")

#     # Paper uses normalization [cite: 132]
#     mean = X_train.mean(axis=(0, 1))
#     std = X_train.std(axis=(0, 1)) + 1e-8
    
#     X_train = (X_train - mean) / std
#     X_val = (X_val - mean) / std
#     X_test = (X_test - mean) / std

#     # Save finalized data 
#     np.savez_compressed(os.path.join(OUTPUT_DIR, "train_data.npz"), X=X_train, y=y_train)
#     np.savez_compressed(os.path.join(OUTPUT_DIR, "val_data.npz"), X=X_val, y=y_val)
#     np.savez_compressed(os.path.join(OUTPUT_DIR, "test_data.npz"), X=X_test, y=y_test)
#     np.savez(os.path.join(OUTPUT_DIR, "norm_stats.npz"), mean=mean, std=std)

#     print(f"\nPreprocess Complete! Classes: {len(le.classes_)}")
#     print(f"Input Shape: ({TARGET_FRAMES}, {FEATURE_DIM})")

# if __name__ == "__main__":
#     process_and_save()







"""
preprocess_paper.py -- Full Replicated Pipeline for SSL400
Optimized for Hybrid CNN-BiLSTM + Attention as per Abhishek et al. (2025)
"""

import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


DATASET_ROOT   = "./Dataset - MP - CSV"
OUTPUT_DIR     = "ssl400_processed_paper"

# Set to 2 to match the "Comprehensive 300-class" scenario [cite: 108]
MIN_SAMPLES_PER_CLASS = 2 

TARGET_FRAMES  = 30    # Key Spec: K=30 [cite: 113, 132]
N_LANDMARKS    = 33    # MediaPipe Holistic landmarks [cite: 42, 113]
FEATURE_DIM    = 99    # 33 landmarks * (x, y, z) 

TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.10
TEST_RATIO     = 0.20
RANDOM_SEED    = 42


def selective_temporal_sampling(seq, k=TARGET_FRAMES):
    """
    Selects K=30 evenly spaced frames to reduce redundancy 
    while preserving gesture dynamics.
    """
    n = len(seq)
    if n == 0: 
        return np.zeros((k, FEATURE_DIM), dtype=np.float32)
    
    # Generate K indices evenly spaced from 0 to N-1 
    indices = np.linspace(0, n - 1, k).astype(int)
    return seq[indices]


def apply_augmentations(seq):
    """
    Applies Gaussian noise, scaling, and rotation as 
    described in the paper's preprocessing summary.
    """
    # 1. Gaussian Noise 
    noise = np.random.normal(0, 0.005, seq.shape)
    seq = seq + noise
    
    # 2. Scaling 
    scale = np.random.uniform(0.9, 1.1)
    seq = seq * scale
    
    # 3. Simple Rotation (approximate) 
    # Small random rotation on the X-Y plane
    angle = np.random.uniform(-0.1, 0.1)
    c, s = np.cos(angle), np.sin(angle)
    for i in range(0, FEATURE_DIM, 3):
        x, y = seq[:, i], seq[:, i+1]
        seq[:, i] = c*x - s*y
        seq[:, i+1] = s*x + c*y
        
    return seq.astype(np.float32)


# def parse_csv_to_array(filepath, augment=False):
#     """
#     Extracts 3D skeletal landmarks from shoulders, elbows, 
#     wrists, and hands.
#     """
#     df = pd.read_csv(filepath)
#     frames = []
    
#     for _, row in df.iterrows():
#         coords = []
#         for col in df.columns[:N_LANDMARKS]:
#             try:
#                 # Extract x, y, z from string format '[x, y, z, v]' 
#                 vals = ast.literal_eval(str(row[col]).strip())
#                 coords.extend(vals[:3])
#             except:
#                 coords.extend([0.0, 0.0, 0.0])
#         frames.append(coords)
    
#     seq = np.array(frames, dtype=np.float32)
#     seq = selective_temporal_sampling(seq)
    
#     if augment:
#         seq = apply_augmentations(seq)
        
#     return seq



def center_landmarks(seq):
    """
    Centers all landmarks relative to the mid-shoulder.
    Landmark 11 = Left Shoulder, 12 = Right Shoulder.
    """
    for t in range(seq.shape[0]):
        # Calculate mid-shoulder point (x, y, z)
        # Each landmark has 3 coords, so index 33 and 36 are x-coords for 11 and 12
        ls_idx, rs_idx = 11 * 3, 12 * 3
        mid_shoulder = (seq[t, ls_idx:ls_idx+3] + seq[t, rs_idx:rs_idx+3]) / 2
        
        # Subtract mid-shoulder from all 33 landmarks
        for i in range(0, seq.shape[1], 3):
            seq[t, i:i+3] -= mid_shoulder
            
    return seq

def parse_csv_to_array(filepath, augment=False):
    df = pd.read_csv(filepath)
    frames = []
    for _, row in df.iterrows():
        coords = []
        for col in df.columns[:N_LANDMARKS]:
            try:
                vals = ast.literal_eval(str(row[col]).strip())
                coords.extend(vals[:3])
            except:
                coords.extend([0.0, 0.0, 0.0])
        frames.append(coords)
    
    seq = np.array(frames, dtype=np.float32)
    seq = selective_temporal_sampling(seq)
    
    
    seq = center_landmarks(seq)
    
    if augment:
        seq = apply_augmentations(seq)
    return seq













def collect_manifest(root, min_samples):
    records = []
    for category in sorted(os.listdir(root)):
        cat_path = os.path.join(root, category)
        if not os.path.isdir(cat_path): continue
        for word in sorted(os.listdir(cat_path)):
            word_path = os.path.join(cat_path, word)
            if not os.path.isdir(word_path): continue
            
            files = [os.path.join(word_path, f) for f in os.listdir(word_path) if f.endswith(".csv")]
            if len(files) >= min_samples:
                for fp in files:
                    records.append({"filepath": fp, "label": word})
    return pd.DataFrame(records)

# ══════════════════════════════════════════════════════════
#  STEP 4 -- CORE PROCESSING LOOP
# ══════════════════════════════════════════════════════════

def process_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = collect_manifest(DATASET_ROOT, MIN_SAMPLES_PER_CLASS)
    
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])
    np.save(os.path.join(OUTPUT_DIR, "label_classes.npy"), le.classes_)

    # Stratified Split 
    train_list, val_list, test_list = [], [], []
    for _, group in df.groupby('label_id'):
        group = group.sample(frac=1, random_state=RANDOM_SEED)
        n = len(group)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        
        train_list.append(group.iloc[:n_train])
        val_list.append(group.iloc[n_train : n_train + n_val])
        test_list.append(group.iloc[n_train + n_val :])

    train_df = pd.concat(train_list)
    val_df = pd.concat(val_list)
    test_df = pd.concat(test_list)

    def load_set(manifest, name, augment=False):
        X, y = [], []
        for i, row in manifest.iterrows():
            # Load original
            arr = parse_csv_to_array(row['filepath'], augment=False)
            X.append(arr)
            y.append(row['label_id'])
            
            # Add augmented copy if training 
            if augment:
                X.append(parse_csv_to_array(row['filepath'], augment=True))
                y.append(row['label_id'])
                
            if i % 100 == 0: print(f"Processing {name}... {i}/{len(manifest)}", end="\r")
        return np.stack(X), np.array(y)

    X_train, y_train = load_set(train_df, "Train", augment=True)
    X_val, y_val = load_set(val_df, "Val", augment=False)
    X_test, y_test = load_set(test_df, "Test", augment=False)

    # Normalize using training statistics 
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1)) + 1e-8
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Serialize with NumPy's binary format for rapid access 
    np.savez_compressed(os.path.join(OUTPUT_DIR, "train_data.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "val_data.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "test_data.npz"), X=X_test, y=y_test)
    np.savez(os.path.join(OUTPUT_DIR, "norm_stats.npz"), mean=mean, std=std)

    print(f"\nPreprocess Complete! Final Classes: {len(le.classes_)}")
    print(f"Sample Count (Train + Aug): {len(X_train)}")

if __name__ == "__main__":
    process_and_save()

















