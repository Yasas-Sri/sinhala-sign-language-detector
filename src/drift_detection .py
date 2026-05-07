import pandas as pd
import numpy as np
import os
import json

from evidently.core.report import Report
from evidently.presets.drift import DataDriftPreset

def generate_drift_report():
    print("Loading reference training data...")
    train_path = os.path.join(os.path.dirname(__file__), "..", "ssl400_processed_paper", "train_data.npz")
    if not os.path.exists(train_path):
        print(f"Could not find training data at {train_path}")
        return
        
    train_data = np.load(train_path)
    X_train_mean = train_data["X"].mean(axis=1) 
    
    feature_columns = [f"feat_{i}" for i in range(99)]
    df_reference = pd.DataFrame(X_train_mean, columns=feature_columns)
    
    if len(df_reference) > 2000:
        df_reference = df_reference.sample(n=2000, random_state=42)
    
    log_path = os.path.join(os.path.dirname(__file__), "..", "production_logs.csv")
    print(f"Loading current production logs from {log_path}...")
    if not os.path.exists(log_path):
        print("No production logs found! Make some predictions in the Streamlit app first.")
        return
        
    df_current = pd.read_csv(log_path)
    



    print("Generating Data Drift report...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    

    drift_snapshot = data_drift_report.run(
        reference_data=df_reference[feature_columns], 
        current_data=df_current[feature_columns]
    )
    
    json_path = os.path.join(os.path.dirname(__file__), "..", "data_drift_report.json")
    

    drift_snapshot.save_json(json_path)
    

    print(" Drift report generated successfully!")
    print(f" Open '{json_path}' to view the drift results.")

if __name__ == "__main__":
    generate_drift_report()