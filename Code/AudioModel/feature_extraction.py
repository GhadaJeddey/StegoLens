import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal.windows import hamming
from librosa import lpc
import soundfile as sf
from tqdm import tqdm



def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=None)  # Load audio

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1) 

    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # Zero-crossing rate
    zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y))

    # Combine all features into one array
    features = np.hstack([mfccs_mean, spectral_centroid, spectral_bandwidth, spectral_flatness, zero_crossings])
    
    return features

def extract_lpc_features(audio, sr, frame_length=0.05, frame_shift=0.01, order=10, min_duration=0.2):

    try:

        # Duration check
        duration = len(audio) / sr
        if duration < min_duration:
            print(f"Skipped (too short): {audio}")
            return None

        frame_size = int(frame_length * sr)
        hop_size = int(frame_shift * sr)
        num_frames = 1 + (len(audio) - frame_size) // hop_size

        lpcs = []

        for i in range(num_frames):
            start = i * hop_size
            frame = audio[start:start + frame_size]
            if len(frame) < frame_size:
                break
            frame *= hamming(len(frame))
            frame = frame / (np.max(np.abs(frame)) + 1e-6)

            try:
                a = lpc(frame, order=order)
                #print("LPC coeffs:", a)
                if len(a) == order + 1:  # LPC includes a0=1
                    lpcs.append(a[1:])  # Skip the first coeff (it's always 1)
            except Exception as e:
                print(f"Skipped unstable frame: {e}")
                continue

        lpcs = np.array(lpcs)
        if len(lpcs) == 0:
            return None

        # Compute mean, variance, and delta of LPCs
        mean_lpc = np.mean(lpcs, axis=0)
        var_lpc = np.var(lpcs, axis=0)
        delta_lpc = np.mean(np.abs(np.diff(lpcs, axis=0)), axis=0)

        features = np.concatenate([mean_lpc, var_lpc, delta_lpc])
        return features

    except Exception as e:
        print(f"Error processing {audio}: {e}")
        return None
    
def process_folder(folder_path, output_csv, label=None):

    data = []
    for file in tqdm(os.listdir(folder_path)):
        if file.lower().endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            try:
                audio, sr = sf.read(file_path)
                if len(audio.shape) > 1:  # Convert stereo to mono
                    audio = audio[:, 0]

                lpc_features = extract_lpc_features(audio, sr)
                basic_features = extract_features(file_path)

                features = np.concatenate([basic_features, lpc_features] )

                if features is not None:
                    row = list(features)
                    if label is not None:
                        row.append(label)
                    data.append(row)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    if not data:
        print(f"No features extracted from folder: {folder_path}")
        return
    
    if data:
        columns =[f"mfcc_{i+1}" for i in range(13)] + \
              ["spectral_centroid", "spectral_bandwidth", "spectral_flatness", "zero_crossing_rate"] + \
              [f"mean_lpc_{i}" for i in range(10)] + \
              [f"var_lpc_{i}" for i in range(10)] + \
              [f"delta_lpc_{i}" for i in range(10)]+ ['label']
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"Saved features to {output_csv}")

    
def extract_features_file(file_path): 

    audio, sr = librosa.load(file_path, sr=None) 
    lpc_feature = extract_lpc_features(audio,sr)
    basic_features = extract_features(file_path)
    features = np.concatenate([basic_features, lpc_feature])
    return(features)

if __name__=="__main__":
    pass