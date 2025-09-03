import joblib
import numpy as np
from feature_extraction import extract_features_file # same one as before
import os

# Load saved model and scaler
model = joblib.load("audio_detection_model.pkl")


# Predict function
def predict(file_path):

    if not file_path.lower().endswith(".wav"):
        print(f"Unsupported file type: {file_path}. Only .wav files are supported.")
        return
    
    features = extract_features_file(file_path)
    if features is None:
        print("Could not extract features.")
        return
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    print(f"Prediction for the file: {'STEGANOGRAPHIC' if prediction == 1 else 'CLEAN'}")

# Example usage
predict("test_set\positive_wav\Chinese130368.wav")
