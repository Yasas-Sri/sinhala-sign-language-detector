import os
import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(SCRIPT_DIR)

DATASET_DIR = os.path.join(SCRIPT_DIR, "../Dataset")   
MAX_SEQUENCE_LENGTH = 100     
OUTPUT_DIR = "preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


sequences = []
labels = []


for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            
            
            label_name = os.path.basename(root)
            
           
            df = pd.read_csv(file_path, header=None)
            
            sequence_data = []
            
            
            for _, row in df.iterrows():
                frame_features = []
                for cell in row:
                    try:
                        
                        values = ast.literal_eval(str(cell))
                        frame_features.extend(values)
                    except:
                        
                        frame_features.extend([0.0] * 4) 
                
                sequence_data.append(frame_features)
            
            sequences.append(np.array(sequence_data, dtype=np.float32))
            labels.append(label_name)

print(f"Loaded {len(sequences)} samples across {len(set(labels))} classes.")


print("Normalizing data...")
all_data = np.vstack(sequences)
scaler = StandardScaler()
scaler.fit(all_data)


sequences_scaled = [scaler.transform(s) for s in sequences]

print("Padding sequences...")
X = pad_sequences(
    sequences_scaled,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post",
    dtype="float32"
)

print("Encoding labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)


np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

print(f"\n Final Data Shape: {X.shape}")
print(f"Features per frame: {X.shape[2]}")
print(f"Saved to '{OUTPUT_DIR}' folder.")