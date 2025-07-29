import librosa
import numpy as np
import traceback

def extract_features(file_path):
    try:
        print(f"🔍 Loading audio from: {file_path}")
        X, sample_rate = librosa.load(file_path, sr=None)
        print("✅ Audio loaded, extracting MFCCs...")
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        print("✅ Feature extraction successful.")
        return mfccs_mean
    except Exception as e:
        print("❌ Error in extract_features:", e)
        traceback.print_exc()  # <== add this to see detailed error
        return None
