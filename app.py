import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import joblib
import tempfile
from utils.feature_extraction import extract_features
import sounddevice as sd
import soundfile as sf
import os

# Load trained model
model = joblib.load("models/emotion_model.pkl")

st.title("🎙️ Emotion Detection from Voice")
st.write("Click record and speak something to analyze your emotion.")

# Record voice
duration = 4  # seconds
sample_rate = 44100

if st.button("🔴 Record"):
    st.write("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    st.write("Recording finished!")

    # Save to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        wav.write(tmpfile.name, sample_rate, audio)
        filepath = tmpfile.name

    # Feature extraction
    features = extract_features(filepath)
    if features is not None:
        features = features.reshape(1, -1)  # Reshape for prediction
        prediction = model.predict(features)[0]
        st.success(f"🎯 Detected Emotion: **{prediction.upper()}**")
    else:
        st.error("Feature extraction failed.")

def record_voice(filename="recorded_audio.wav", duration=4, fs=44100):
    print("🎙️ Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    # Save in current directory (safe)
    path = os.path.join(os.getcwd(), filename)
    sf.write(path, recording, fs)
    print("✅ Saved at:", path)

    return path  # return full path

file_path = record_voice()
features = extract_features(file_path)
