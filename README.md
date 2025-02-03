# Music Genre Classification with Deep Learning

This project uses deep learning to classify music genres based on audio files. A Streamlit app has been developed to predict genres in real-time, where users can upload audio files and get predictions for each segment of the track.

The project is based on the **GTZAN music genre dataset** and includes models trained using **Convolutional Neural Networks (CNN)** and **Transformer models**.

## Features:
- **Real-time genre prediction** using a lightweight CNN model.
- **Audio feature extraction** via MFCC (Mel-Frequency Cepstral Coefficients).
- **Two different model architectures**: CNN (2MB) and Transformer (~2GB).
- **Streamlit app** for easy interaction and inference.

### Files:
├── **app.py**: A Streamlit web app that allows users to upload audio files and receive genre predictions based on the CNN model.              
├── **cnn_model.pth**: Trained CNN model, optimized for fast inference (2MB).         
├── **model.py**: Architecture definition of the CNN model used in the project.              
├── **music_classification.ipynb**: A notebook that processes the GTZAN dataset, extracts audio features, and trains both CNN and Transformer models.

└── **requirements.txt**: Lists all dependencies needed to run the project. 

### How the App Works:
- The audio file is broken down into 6-second segments.
- Each segment undergoes a separate classification.
- The final genre prediction is made by taking the majority vote of the classifications from all segments.

### Model Details:
- **CNN Model**: A lightweight, efficient model with 2MB size, ideal for real-time inference in the Streamlit app.
- **Transformer Model**: A more powerful model (2GB in size) used for offline processing and high-quality predictions.

### How to Run the App:
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run streamlit app:
   ```bash
   streamlit run app.py
   ```

Example of Genre Prediction:
Here’s an example of how the app predicts the genre of a song after analyzing a 6-second segment.

![Genre Prediction Example](./assets/demo_img.png)

Watch the demo of the app:
Click [here to watch the demo](./assetsdemo.mov).
