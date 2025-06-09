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
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import streamlit.components.v1 as components


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
# Xác định hàm để trích xuất MFCC từ mỗi phân đoạn
def extract_mfcc_from_segment(signal, start_sample, finish_sample, n_mfcc, n_fft, hop_length):
    """Extract MFCC features from a segment of the audio signal."""
    segment_signal = signal[start_sample:finish_sample]
    mfcc_features = librosa.feature.mfcc(y=segment_signal, sr=SAMPLE_RATE, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc_features.T  # Shape: (num_frames, n_mfcc)

# Define function to predict genre from MFCC data
# Xác định hàm để dự đoán thể loại từ dữ liệu MFCC
def predict_genre(mfcc_data, model):
    """Predict the genre based on MFCC features."""
    mfcc_data_tensor = torch.tensor(mfcc_data, dtype=torch.float32)
    mfcc_data_tensor = mfcc_data_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(mfcc_data_tensor)
        predicted = torch.argmax(F.softmax(output, dim=1), dim=1)
        return predicted.item() 

# Define function to process audio and predict genre for each segment
# Xác định chức năng xử lý âm thanh và dự đoán thể loại cho từng phân đoạn
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
        # Trích xuất các tính năng MFCC cho từng phân đoạn
        mfcc_features = extract_mfcc_from_segment(signal, start_sample, finish_sample, n_mfcc, n_fft, hop_length)
        if len(mfcc_features) == expected_vects_ps:
            predicted_genre = predict_genre(mfcc_features, model)
            predictions.append(predicted_genre)
    
    return predictions, signal  

def app():

    # 🎊 Tích hợp Side Confetti từ tsParticles và particles.html
    with open("particles.html", "r", encoding="utf-8") as f:
        particles_html = f.read()
    components.html(particles_html, height=0, width=0)

    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>                
            @keyframes flicker {
                0%, 100% {
                    text-shadow: 0 0 10px #fff, 0 0 20px #f0f, 0 0 30px #0ff;
                }
                50% {
                    text-shadow: 0 0 5px #f0f, 0 0 15px #0ff, 0 0 25px #fff;
                }
            }

            @keyframes glow-border {
                0%, 100% {
                    box-shadow: 0 0 20px #fff, 0 0 30px #0ff, 0 0 40px #f0f;
                }
                50% {
                    box-shadow: 0 0 15px #0ff, 0 0 25px #f0f, 0 0 35px #fff;
                }
            }
                
            .glow-box {
                z-index: 1;
                position: relative;
                text-align: center;
                background-image: url('https://getwallpapers.com/wallpaper/full/6/d/f/447707.jpg'); 
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                width: 1000px;
                height: 250px;
                margin-left: -140px;
                overflow: hidden;
                animation: glow-border 2s infinite;
            }

            .glow-box::before {
                content: "";
                position: absolute;
                top: 0; left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                z-index: 0;
                border-radius: 12px;
            }

            .glow-text {
                position: relative;
                z-index: 2;
                font-size: 60px;
                font-family: 'Orbitron', sans-serif;
                animation: flicker 2s infinite;
                color: #ffffff;
                text-align: center;
                margin-bottom: 0;
            }
            
            .progress-bar {
                height: 20px;
                border-radius: 10px;
                background: linear-gradient(90deg, #6b48ff, #ff6b6b);
                position: relative;
                overflow: hidden;
                box-shadow: 0 0 5px rgba(107, 72, 255, 0.5);
                margin: 10px 0;
            }

            .progress-bar::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
                animation: sparkle 2s infinite;
            }

            @keyframes sparkle {
                0% { left: -100%; }
                20% { left: 0%; }
                100% { left: 100%; }
            }

            .progress {
                height: 100%;
                background: #fff;
                border-radius: 10px;
                transition: width 0.5s ease-in-out;
                box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.2);
            }
        </style>

        <div class='glow-box'>
            <h1><span class='glow-text'>🎵 Music Genre Prediction App</span></h1>
        </div>
    """, unsafe_allow_html=True)

    # Tùy chỉnh giao diện tiêu đề phụ và khối upload
    col1, col2, col3 = st.columns([1, 12, 1])
    with col2:
        st.markdown("""
            <style>
                .custom-upload-title {
                    font-family: Trebuchet MS;
                    font-size: 32px;
                    font-weight: 700;
                    color: #808080;
                    text-align: left;
                    margin-top: 15px;
                    margin-bottom: 10px;
                }

                .custom-caption {
                    font-family: Trebuchet MS;
                    text-align: left;
                    font-size: 16px;
                    color: #999999;
                    margin-bottom: 12px;
                    margin-left: 10px;
                }

                section[data-testid="stFileUploader"] > div {
                    background-color: #222;
                    border-radius: 15px;
                    padding: 20px;
                    border: 2px solid #555;
                    box-shadow: 0 0 20px rgba(0,255,255,0.2);
                    transition: 0.3s ease;
                    max-width: 100%; /* Không vượt khỏi cột */
                }

                section[data-testid="stFileUploader"] > div:hover {
                    box-shadow: 0 0 30px rgba(0,255,255,0.5);
                    border-color: #0ff;
                }
            </style>

            <div class="custom-upload-title">📤 Upload Audio File</div>
            <div class="custom-caption">🎶 Choose an audio file</div>
        """, unsafe_allow_html=True)

        # Upload file nằm ngay bên dưới caption (đúng block)
        audio_file = st.file_uploader(
            "Drag and drop file here", type=["wav", "mp3"], label_visibility="collapsed"
        )

    model = load_model('./models/cnn_model.pth')

    if audio_file is not None:
        # predictions = process_audio_and_predict(audio_file, model)
        predictions, signal = process_audio_and_predict(audio_file, model)
        genre_list = [
            "disco", "metal", "reggae", "blues", "rock", 
            "classical", "jazz", "hiphop", "country", "pop"
        ]
        predicted_genre_names = [genre_list[idx] for idx in predictions]

        # Dữ liệu bảng dự đoán
        data = {
            "Đoạn": [f"Đoạn {i+1}" for i in range(len(predicted_genre_names))],
            "Thể loại dự đoán": predicted_genre_names
        }

        df = pd.DataFrame(data)

        # Hiển thị bảng kết quả
        st.subheader("🎼 Dự đoán thể loại cho từng đoạn")
        st.dataframe(df, use_container_width=True)

        # Tính toán thể loại chiếm ưu thế
        genre_count = Counter(predicted_genre_names)
        majority_genre = genre_count.most_common(1)[0][0]

        # Hiển thị thể loại chính
        st.markdown(f"""
        ### 🏆 Dự đoán cuối cùng
        Bài hát này thuộc thể loại: **:green[{majority_genre.upper()}]**
        """)

        # Tính toán tần suất thể loại
        total_segments = len(predictions)
        genre_percentages = {genre: (count / total_segments) * 100 for genre, count in genre_count.items()}

        # Hiển thị kết quả với thanh tiến trình động
        st.subheader("🎼 Tần suất các thể loại")
        for genre, percentage in genre_percentages.items():
            st.markdown(
                f"""
                <div>
                    <span style="font-weight: bold; color: #fff;">{genre.capitalize()}</span>
                    <div class="progress-bar" style="width: 100%;">
                        <div class="progress" style="width: {percentage}%;"></div>
                    </div>
                    <span style="color: #ccc;">{percentage:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Phát lại bài hát
        st.subheader("▶️ Nghe lại bài hát đã tải lên")
        st.audio(audio_file, format='audio/wav')

        # Hiển thị sóng âm
        with st.expander("📊 Hiển thị dạng sóng (waveform)"):
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(signal, sr=SAMPLE_RATE, ax=ax)
            ax.set_title("Dạng sóng âm thanh")
            ax.set_xlabel("Thời gian")
            ax.set_ylabel("Biên độ")
            st.pyplot(fig)



if __name__ == "__main__":
    app()