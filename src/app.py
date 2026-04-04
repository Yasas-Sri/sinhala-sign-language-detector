# import streamlit as st
# import cv2
# import numpy as np
# import torch
# import mediapipe as mp
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# from collections import deque
# import os
# import sys

# # Add the root directory to path so python can see the "notebooks" folder
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# from notebooks.fc110562_Yasas.model import SLRHybridModel

# st.set_page_config(page_title="Real-Time Sign Language Detector", layout="wide")
# st.title("Real-Time Sign Language Detection")

# # ─────────────────────────────────────────────────────────
# #  SIDEBAR: MODEL SELECTION
# # ─────────────────────────────────────────────────────────
# st.sidebar.title("Settings")

# MODELS_DIR = "models"
# if not os.path.exists(MODELS_DIR):
#     os.makedirs(MODELS_DIR)

# # Get all model files in the models directory
# available_models = [m for m in os.listdir(MODELS_DIR) if m.endswith(('.pt', '.pth'))]
# if not available_models:
#     st.sidebar.warning("No models found in the 'models' folder. Please add your .pt files there.")
#     selected_model = None
# else:
#     selected_model = st.sidebar.selectbox("Choose a Model:", available_models)

# # ─────────────────────────────────────────────────────────
# #  CONFIGURATION & SETUP
# # ─────────────────────────────────────────────────────────
# LABELS_PATH = "ssl400_processed_paper/label_classes.npy"
# STATS_PATH = "ssl400_processed_paper/norm_stats.npz"

# NUM_CLASSES = 353
# INPUT_DIM = 99
# HIDDEN_DIM = 128
# SEQ_LEN = 30

# @st.cache_resource
# def load_assets(model_filename):
#     if not model_filename:
#         return None, None, None, None, None
        
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_path = os.path.join(MODELS_DIR, model_filename)
    
#     # Load Model structure
#     model = SLRHybridModel(NUM_CLASSES, INPUT_DIM, HIDDEN_DIM).to(device)
#     try:
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.eval()
#     except Exception as e:
#         st.error(f"Error loading model weights from {model_path}: {e}")
#         return None, None, None, None, None

#     # Load Labels & Stats
#     try:
#         labels = np.load(LABELS_PATH, allow_pickle=True)
#         stats = np.load(STATS_PATH)
#         mean, std = stats["mean"], stats["std"]
#     except Exception as e:
#         st.error(f"Error loading data files: {e}")
#         return model, None, None, None, device

#     return model, labels, mean, std, device

# model, class_names, mean_stat, std_stat, device = load_assets(selected_model)

# # ─────────────────────────────────────────────────────────
# #  MEDIAPIPE & FEATURE EXTRACTION
# # ─────────────────────────────────────────────────────────

# def extract_landmarks(results):
#     """Extract 33 Pose landmarks (33 * 3 = 99 features)"""
#     # The training uses N_LANDMARKS=33, likely pose or a specific subset. 
#     # Adjust this if you specifically trained on different MediaPipe landmarks.
#     if results.pose_landmarks:
#         res = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
#         # Fallback if fewer than 33 landmarks for some reason
#         if len(res) < INPUT_DIM:
#             res = np.pad(res, (0, INPUT_DIM - len(res)), 'constant')
#         else:
#             res = res[:INPUT_DIM]
#         return res
#     else:
#         return np.zeros(INPUT_DIM)

# # ─────────────────────────────────────────────────────────
# #  UI AND INFERENCE LOOP
# # ─────────────────────────────────────────────────────────
# if model is None or class_names is None:
#     st.warning("Please ensure the model weights, labels, and stats exist.")
#     st.stop()

# run_app = st.checkbox("Start Webcam")
# FRAME_WINDOW = st.image([])
# text_output = st.empty()

# if run_app:
#     cap = cv2.VideoCapture(0)
#     frames_queue = deque(maxlen=SEQ_LEN)
    
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while cap.isOpened() and run_app:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to capture video.")
#                 break

#             # Convert colors for MediaPipe and UI
#             image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image_rgb.flags.writeable = False
            
#             # Process with MediaPipe
#             results = holistic.process(image_rgb)
            
#             image_rgb.flags.writeable = True
            
#             # Draw landmarks
#             mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
#             # Extract and queue features
#             features = extract_landmarks(results)
#             frames_queue.append(features)
            
#             # Inference if queue is full
#             prediction_text = "Waiting for frames..."
#             if len(frames_queue) == SEQ_LEN:
#                 seq_array = np.array(frames_queue) # Shape (30, 99)
                
#                 # Normalize exactly like preprocess.py
#                 seq_array = (seq_array - mean_stat) / (std_stat + 1e-8)
                
#                 # To Tensor (1, 30, 99)
#                 input_tensor = torch.from_numpy(seq_array).float().unsqueeze(0).to(device)
                
#                 with torch.no_grad():
#                     output = model(input_tensor)
#                     pred_idx = output.argmax(dim=1).item()
#                     confidence = torch.softmax(output, dim=1)[0, pred_idx].item()
                    
#                     if confidence > 0.4: # Lowered confidence threshold slightly for live video
#                         prediction_text = f"Sign: {class_names[pred_idx]} ({confidence:.2%})"
#                     else:
#                         prediction_text = "Sign: None (Low Confidence)"
            
#             # Update UI
#             # Mirror the view for a more natural webcam experience
#             display_frame = cv2.flip(image_rgb, 1) 
            
#             # Draw the prediction text directly on the video stream!
#             cv2.putText(
#                 display_frame, 
#                 prediction_text, 
#                 (20, 50), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 1, 
#                 (0, 255, 0), 
#                 2, 
#                 cv2.LINE_AA
#             )
            
#             FRAME_WINDOW.image(display_frame)
            
#     cap.release()
# else:
#     st.write("Click 'Start Webcam' to begin sign language detection.")








"""
SSL400 Real-Time Sign Language Recognition — Streamlit App
Camera loop is driven entirely by session_state, never by widget return values.
"""

import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from notebooks.fc110562_Yasas.model import SLRHybridModel

st.set_page_config(page_title="SLR Pipeline", layout="wide")


@st.cache_resource
def load_all_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SLRHybridModel(num_classes=353, input_dim=99, hidden_dim=128).to(device)
    model.load_state_dict(torch.load("models/best_hybrid_model.pt", map_location=device))
    model.eval()
    stats  = np.load("ssl400_processed_paper/norm_stats.npz")
    labels = np.load("ssl400_processed_paper/label_classes.npy", allow_pickle=True)
    return model, labels, stats["mean"], stats["std"], device

model, class_names, mean_stat, std_stat, device = load_all_assets()


def init_state():
    defaults = {
        "camera_on":    False,
        "capture_flag": False,
        "recording":    False,
        "buffer":       [],
        "frame_count":  0,
        "sentence":     [],
        "last_word":    "",
        "last_conf":    0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def cb_start_camera():
    st.session_state.camera_on = True

def cb_stop_camera():
    st.session_state.camera_on    = False
    st.session_state.recording    = False
    st.session_state.capture_flag = False
    st.session_state.buffer       = []
    st.session_state.frame_count  = 0
    if "cap" in st.session_state and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

def cb_capture():
    if not st.session_state.recording:
        st.session_state.capture_flag = True

def cb_clear():
    st.session_state.sentence  = []
    st.session_state.last_word = ""
    st.session_state.last_conf = 0.0

st.title(" Real-Time Sign Language Recognition")

col_video, col_ctrl = st.columns([3, 1])

with col_ctrl:
    st.subheader("Controls")

    if not st.session_state.camera_on:
        st.button(" Start Camera",  on_click=cb_start_camera, use_container_width=True)
    else:
        st.button(" Stop Camera",   on_click=cb_stop_camera,  use_container_width=True)
        st.button(" Capture Sign",  on_click=cb_capture,      use_container_width=True,
                  disabled=st.session_state.recording)

    st.divider()
    status_placeholder   = st.empty()
    progress_placeholder = st.empty()
    result_placeholder   = st.empty()

    st.divider()
    st.subheader("Sentence")
    sentence_placeholder = st.empty()
    st.button(" Clear", on_click=cb_clear, use_container_width=True)

with col_video:
    video_placeholder = st.empty()


sentence_placeholder.markdown(
    " ".join(st.session_state.sentence) if st.session_state.sentence else "_No words yet_"
)


if st.session_state.camera_on:

    if "cap" not in st.session_state or st.session_state.cap is None:

        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        st.session_state.cap.set(cv2.CAP_PROP_FPS,          30)
        st.session_state.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   

    cap = st.session_state.cap

    if not cap.isOpened():
        cap.open(0) # try reopening

    if not cap.isOpened():
        st.error(" Could not open camera. Make sure no other app is using it.")
        st.session_state.camera_on = False
        st.stop()

    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while st.session_state.camera_on: 

            ret, frame = cap.read()
            if not ret:
                status_placeholder.warning(" Dropped frame, retrying…")
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)

            
            if st.session_state.capture_flag and not st.session_state.recording:
                st.session_state.capture_flag = False
                st.session_state.recording    = True
                st.session_state.buffer       = []
                st.session_state.frame_count  = 0

            
            overlay_text  = "READY — press Capture"
            overlay_color = (0, 220, 0)

            if st.session_state.recording:
                n = st.session_state.frame_count

                if n < 30:
                    if results.pose_landmarks:
                        lm = np.array([
                            [lk.x, lk.y, lk.z]
                            for lk in results.pose_landmarks.landmark
                        ]).flatten()[:99]

                        
                        mid = (lm[11*3:11*3+3] + lm[12*3:12*3+3]) / 2
                        for i in range(0, 99, 3):
                            lm[i:i+3] -= mid

                        st.session_state.buffer.append(lm)
                        st.session_state.frame_count += 1
                        n = st.session_state.frame_count

                    overlay_text  = f"RECORDING  {n}/30"
                    overlay_color = (255, 100, 0)
                    progress_placeholder.progress(n / 30)
                    status_placeholder.info(f"Recording… {n}/30 frames")

                else:
                    
                    st.session_state.recording = False
                    progress_placeholder.empty()

                    buf = np.array(st.session_state.buffer, dtype=np.float32)
                    if len(buf) == 30:
                        seq     = (buf - mean_stat) / (std_stat + 1e-8)
                        input_t = (torch.from_numpy(seq)
                                        .float()
                                        .unsqueeze(0)
                                        .to(device))

                        with torch.no_grad():
                            logits   = model(input_t)
                            probs    = torch.softmax(logits, dim=1)[0]
                            pred_idx = probs.argmax().item()
                            conf     = probs[pred_idx].item()
                            word     = str(class_names[pred_idx])

                        st.session_state.last_word = word
                        st.session_state.last_conf = conf
                        st.session_state.sentence.append(word)

                        result_placeholder.success(f"**{word}** ({conf*100:.1f}%)")
                        status_placeholder.success(" Done! Press Capture for next sign.")
                        sentence_placeholder.markdown(" ".join(st.session_state.sentence))
                    else:
                        status_placeholder.warning(" Pose not detected in enough frames.")

                    st.session_state.buffer      = []
                    st.session_state.frame_count = 0

            
            display = cv2.flip(img_rgb, 1)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1),
                )

            cv2.rectangle(display, (0, 0), (430, 50), (0, 0, 0), -1)
            cv2.putText(display, overlay_text, (8, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, overlay_color, 2)

            if st.session_state.last_word:
                wm = f"{st.session_state.last_word}  {st.session_state.last_conf*100:.0f}%"
                cv2.putText(display, wm, (8, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            video_placeholder.image(display, channels="RGB", use_container_width=True)

   

else:
    video_placeholder.markdown(
        """
        <div style="text-align:center;padding:100px 40px;
                    border:2px dashed #555;border-radius:14px;color:#888;">
            <h2></h2><p>Press <b>Start Camera</b> to begin</p>
        </div>
        """,
        unsafe_allow_html=True,
    )