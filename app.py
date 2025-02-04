import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
import streamlit as st
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
from models.GenreClassifier import GenreClassifier

SAMPLE_RATE = 22050 
n_mfcc = 13  
n_fft = 2048 
hop_length = 512
num_segments = 5  
SAMPLES_PER_TRACK = SAMPLE_RATE * 30

def load_model(model_path):
    model = GenreClassifier(num_classes=10) 
    model.load_state_dict(torch.load(model_path))  
    model.eval() 
    return model

# Define function to extract MFCC from each segment
def extract_mfcc_from_segment(signal, start_sample, finish_sample, n_mfcc, n_fft, hop_length):
    """Extract MFCC features from a segment of the audio signal."""
    segment_signal = signal[start_sample:finish_sample]
    mfcc_features = librosa.feature.mfcc(y=segment_signal, sr=SAMPLE_RATE, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc_features.T  # Shape: (num_frames, n_mfcc)

# Define function to predict genre from MFCC data
def predict_genre(mfcc_data, model):
    """Predict the genre based on MFCC features."""
    mfcc_data_tensor = torch.tensor(mfcc_data, dtype=torch.float32)
    mfcc_data_tensor = mfcc_data_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(mfcc_data_tensor)
        predicted = torch.argmax(F.softmax(output, dim=1), dim=1)
        return predicted.item() 

# Define function to process audio and predict genre for each segment
def process_audio_and_predict(file, model):
    """Process the uploaded audio file, extract MFCC features for each segment, and predict genre for each segment."""
    signal, sr = librosa.load(file, sr=SAMPLE_RATE)
    samples_ps = int(SAMPLES_PER_TRACK / num_segments) 
    expected_vects_ps = int(np.ceil(samples_ps / hop_length))  
    
    predictions = []  

    for s in range(num_segments):
        start_sample = samples_ps * s
        finish_sample = start_sample + samples_ps
        
        # Extract MFCC features for each segment
        mfcc_features = extract_mfcc_from_segment(signal, start_sample, finish_sample, n_mfcc, n_fft, hop_length)
        if len(mfcc_features) == expected_vects_ps:
            predicted_genre = predict_genre(mfcc_features, model)
            predictions.append(predicted_genre)
    
    return predictions  

def app():
    st.title("Audio Genre Prediction App")
    model = load_model('./models/cnn_model.pth')

    st.sidebar.title("Upload Audio File")
    audio_file = st.sidebar.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if audio_file is not None:
        predictions = process_audio_and_predict(audio_file, model)
        genre_list = [
        "disco", "metal", "reggae", "blues", "rock", 
        "classical", "jazz", "hiphop", "country", "pop"
        ]
        predicted_genre_names = [genre_list[idx] for idx in predictions]
        print(predictions)
        data = {
            "Segment": [f"Segment {i+1}" for i in range(len(predicted_genre_names))],
            "Predicted Genre": predicted_genre_names
        }

        st.table(data)
        genre_count = Counter(predicted_genre_names)
        majority_genre = genre_count.most_common(1)[0][0] 
        st.write(f"The classified genre for this song is: **{majority_genre}**")
        st.audio(audio_file, format='audio/wav')

if __name__ == "__main__":
    app()