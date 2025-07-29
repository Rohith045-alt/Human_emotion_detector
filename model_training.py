import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import StratifiedShuffleSplit

from utils.feature_extraction import extract_features

DATA_DIR = "data/ravdess"

def load_data(directory):
    features = []
    labels = []

    for file in glob.glob(os.path.join(directory, "*.wav")):
        try:
            # Example file: happy_1.wav → label = happy
            basename = os.path.basename(file)
            label = basename.split("_")[0].lower()

            feature = extract_features(file)
            if feature is not None:
                features.append(feature)
                labels.append(label)
        except Exception as e:
            print("Error loading:", file, e)

    return np.array(features), np.array(labels)

X, y = load_data(DATA_DIR)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Labels:", y)
print("Feature shape:", X.shape)
from collections import Counter
print("Class count:", Counter(y))


model.fit(X, y)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Training Accuracy: {accuracy:.2f}")

joblib.dump(model, "models/emotion_model.pkl")
print("Model trained on full dataset")
