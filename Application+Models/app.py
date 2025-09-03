import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import librosa
import numpy as np
import dpkt
import io
import os
import torch.nn as nn
import torchvision.models as models
import pickle
from audio_model.feature_extraction import extract_features_file
import joblib
import pandas as pd
class LogitScaler(nn.Module):
    def __init__(self, model, scale=0.5):
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, x):
        return self.model(x) * self.scale
    
# Define image transformations
val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Image model class
class ImageModel(nn.Module):
    def __init__(self, num_classes=4):  # Adjust num_classes to match your training
        super(ImageModel, self).__init__()
        base_model = models.resnet34(pretrained=False)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        self.model = LogitScaler(base_model, scale=0.5)  # Match training scale
    
    def forward(self, x):
        return self.model(x)

# Placeholder model classes
class AudioModel:
    def process(self, audio_data, sr):
        # Placeholder for audio model processing
        return "Audio processed"

class PacketModel(nn.Module):
    def __init__(self):
        super(PacketModel, self).__init__()
        # Example layers; replace with your actual model architecture
        self.fc = nn.Linear(10, 2)  # example

    def forward(self, x):
        return self.fc(x)

def load_image_model(model_path="image_detection_model.pth", state_dict_path="phase1_best_model_state_dict.pth"):
    """Load the image model from saved files."""
    try:
        model = ImageModel(num_classes=4)  # Replace 4 with actual class count

        if os.path.exists(model_path):
            try:
                # Try loading full model
                model = torch.load(model_path, map_location=torch.device('cpu'),weights_only=False)
                st.write("Loaded full image model from image_detection_model.pth")
                model.eval()
                return model
            except Exception as e:
                st.warning(f"Failed to load full model. Attempting state_dict... ({e})")

        if os.path.exists(state_dict_path):
            state = torch.load(state_dict_path, map_location=torch.device('cpu'))
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
                st.write("Loaded state dict from checkpoint (model_state_dict).")
            else:
                model.load_state_dict(state)
                st.write("Loaded plain state dict.")
            model.eval()
            return model

        st.error("Neither full model nor state dict found.")
        return None

    except Exception as e:
        st.error(f"Error loading image model: {str(e)}")
        return None


def load_audio_model(model_path="audio_model/audio_detection_model.pkl"):
    """Load the audio model from a .pkl file."""
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            st.write("Loaded audio model from audio_model.pkl")
            return model
        else:
            st.error("Audio model file not found.")
            return None
    except Exception as e:
        st.error(f"Error loading audio model: {str(e)}")
        return None


def load_packet_model(model_path="packet_detection_model.joblib"):
    try:
        abs_path = os.path.abspath(model_path)
        st.write(f"Attempting to load Scikit-learn model from: {abs_path}")

        if not os.path.exists(abs_path):
            st.error(f"Model file does not exist at: {abs_path}")
            return None

        model = joblib.load(abs_path)
        st.success("Packet model loaded successfully (Scikit-learn).")
        return model
        
    except Exception as e:
        st.error(f"Detailed error loading packet model: {type(e).__name__}")
        st.error(f"Error message: {str(e)}")
        st.error("Please ensure:")
        st.error("- PyTorch versions match between saving and loading")
        st.error("- Model class definition is identical to when saved")
        return None

def is_image_file(filename):
    """Check if the file is an image based on extension."""
    image_extensions = {'.pgm', '.png', '.jpg', '.jpeg', '.bmp'}
    return os.path.splitext(filename)[1].lower() in image_extensions

def is_audio_file(filename):
    """Check if the file is an audio based on extension."""
    audio_extensions = {'.wav'}
    return os.path.splitext(filename)[1].lower() in audio_extensions

def is_csv_file(filename):
    """Check if the file is a CSV based on extension."""
    return os.path.splitext(filename)[1].lower() == '.csv'

def process_image(file, model):
    """Process image file with transformations and model."""
    try:
        if model is None:
            return "Image model not loaded."
        
        # Define class labels in same order as used during training
        class_labels = ['cover', 'LSB', 'WOW', 'HILL']
        
        # Load and transform image
        image = Image.open(file).convert('L')  # Convert to grayscale for .pgm
        image_tensor = val_transforms(image).unsqueeze(0)  # Add batch dimension
        
        # Perform inference
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_index = torch.argmax(probabilities, dim=1).item()
            predicted_label = class_labels[predicted_index]
            confidence = probabilities[0, predicted_index].item()

        return f"Predicted: **{predicted_label}**"
    
    except Exception as e:
        return f"Error processing image: {str(e)}"


def process_audio(file, model):
    """Process audio file."""
    try:
        if model is None:
            return "Audio model not loaded."
        
        # Load audio file
        audio,sr = librosa.load(file)
        features = extract_features_file(audio,sr)
        if features is None:
            print("Could not extract features.")
            return
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        if prediction == 1 : 
            return "the file is STEGANOGRAPHIC. Be aware!"
        else: 
            return "The file is CLEAN"
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def process_csv(file, model):
    try:
        if model is None:
            return "Packet model not loaded."

        df = pd.read_csv(file)

        if df.empty:
            return "CSV is empty or improperly formatted."

        # Drop label if present
        if 'Label' in df.columns:
            df = df.drop(columns=['Label'])

        # Coerce to numeric and drop invalid rows
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # Match expected features
        expected_features = getattr(model, "feature_names_in_", None)
        if expected_features is not None:
            df = df.loc[:, df.columns.intersection(expected_features)]

            if len(df.columns) != len(expected_features):
                return f"Mismatch in expected features. Expected: {len(expected_features)}, Got: {len(df.columns)}"

        if df.empty:
            return "No valid data to analyze."

        preds = model.predict(df.values)

        # Indicate presence of stego
        if (preds != 0).any():
            return "Stego detected"
        else:
            return "No stego detected"

    except Exception as e:
        return f"Error processing CSV: {str(e)}"

def main():
    st.title("AI Model Testing Interface")
    st.write("Upload an image (.pgm), audio (.wav), or Network Traffic (.csv) file.")
    
    # Load models
    image_model = load_image_model(model_path="image_detection_model.pth", state_dict_path="phase1_best_model_state_dict.pth")
    audio_model = load_audio_model()
    packet_model = load_packet_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pgm', 'png', 'jpg', 'jpeg', 'bmp', 'wav', 'csv'])
    
    if uploaded_file is not None:
        # Get file extension and name
        file_name = uploaded_file.name
        
        # Detect file type and process accordingly
        if is_image_file(file_name):
            st.write("Detected: Image file")
            result = process_image(uploaded_file, image_model)
            st.write(result)
        elif is_audio_file(file_name):
            st.write("Detected: Audio file")
            result = process_audio(uploaded_file, audio_model)
            st.write(result)
        elif is_csv_file(file_name):
            st.write("Detected: CSV file")
            result = process_csv(uploaded_file, packet_model)
            st.write(result)
        else:
            st.error("Unsupported file type. Please upload an image (.pgm, .png, .jpg, .jpeg, .bmp), audio (.wav, .mp3), or CSV (.csv) file.")

if __name__ == "__main__":
    main()