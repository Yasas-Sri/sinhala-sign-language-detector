# Real-Time Sign Language Detector 

A real-time, interactive Sign Language Recognition (SLR) application built with **Streamlit**, **MediaPipe**, and **PyTorch**. The system captures video from your webcam, extracts anatomical landmarks in real-time, and uses a trained PyTorch model to classify the sign out of **353 distinct vocabulary words**.

##  Features

- **Real-Time Landmark Extraction**: Uses MediaPipe Holistic to track body and hand landmarks instantly.
- **Deep Learning Model**: Utilizes a PyTorch hybrid architecture to process temporal sequences of skeletal landmarks.
- **Experiment Tracking**: Integrated with **MLflow** to track training runs, log hyperparameters, and manage model versions.
- **Production Monitoring**: Exposes live metrics (inference latency, prediction counts, CPU/memory usage) via **Prometheus** on port 8000, visualized through **Grafana** dashboards on port 3000.
- **Data Drift Detection**: Uses **Evidently AI** to compare the production feature distribution against the training set, flagging when real-world pose data has shifted away from what the model was trained on.

## Dataset

This project is trained and evaluated using the [SSL400 Dynamic Sri Lankan Sign Language Dataset](https://www.kaggle.com/datasets/yohanabhishek/ssl400-dynamic-sri-lankan-sign-language-dataset) from Kaggle, which provides a comprehensive collection of dynamic signs for 400 vocabulary words.

##  Project Structure

- `src/app.py`: The main Streamlit web application.
- `models/`: Directory where the PyTorch model weights (e.g., `.pt`, `.pth`) are stored.
- `src/preprocess.py`: dataset preprocessing code

##  Getting Started

### Prerequisites
Make sure you have Python 3.9+ installed.

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Add Model Weights**: Ensure your trained model weights (e.g., `best_hybrid_model.pt`) are placed securely in the `/models/` directory.

### Running the App

Start the Streamlit interface using the command:

```bash
streamlit run src/app.py
```

##  How to Use

1. **Start the Camera**: Once the interface loads, click ` Start Camera` to initialize your webcam.
2. **Capture a Sign**: Click ` Capture Sign` to begin recording.
3. **Perform the Sign**: You have a 30-frame window . Ensure your upper body and hands are visible.
4. **View Result**: The model will parse the frames, present the predicted sign along with a confidence score, and add the recognized word to your sentence builder!

