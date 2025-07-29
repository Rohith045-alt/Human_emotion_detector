#  Human Emotion Detection from Voice 🎭

This project is a **voice-based emotion recognition system** developed using **Python**, **Librosa**, **Scikit-learn**, and **Streamlit**. The application captures audio input from the user, extracts audio features, and predicts the **emotional state** (Happy, Sad, Angry, Fear) using a trained machine learning model.

---

##  Table of Contents

- [Introduction](#-introduction)
- [ Abstract](#-abstract)
- [Tools Used](#-tools-used)
- [Steps Involved](#️-steps-involved)
- [UI/Portfolio Design](#️-uiportfolio-design)
- [Future Enhancements](#-future-enhancements)
- [Conclusion](#-conclusion)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)

---

## Introduction

Emotion plays a crucial role in communication. In this project, I built a voice emotion detection system that can recognize human emotions based on their voice signals using **Machine Learning** and **Audio Signal Processing**.

---

## Abstract

The goal of this project is to detect emotions such as **Happy, Sad, Angry, and Fear** by analyzing short voice samples. By applying **MFCC-based feature extraction**, followed by training a **Random Forest Classifier**, the model can classify voice emotions effectively.

---

## Tools Used

| Tool | Purpose |
|------|---------|
| Python | Main programming language |
| Librosa | Audio analysis and MFCC extraction |
| NumPy | Data manipulation |
| Scikit-learn | Machine learning (Random Forest) |
| Streamlit | Frontend Web UI |
| SoundDevice | Recording audio via microphone |
| Joblib | Saving the trained model |

---

## Steps Involved

1. **Collect Audio Dataset**: WAV files labeled with emotions.
2. **Preprocessing & Feature Extraction**: 
   - Used **MFCC (Mel Frequency Cepstral Coefficients)** to extract relevant audio features.
3. **Train Model**:
   - Used `RandomForestClassifier` to train on extracted features.
4. **Build Streamlit UI**:
   - Added mic-recording, emotion prediction, and real-time response.
5. **Testing & Evaluation**:
   - Trained on 8+ WAV files.
   - Accuracy depends on diversity and quantity of training data.

---

## UI/Portfolio Design

I created the application to look like a **mini-portfolio project** with clean design and easy interactions. The interface allows users to:

- Record their voice directly
- Predict emotion instantly
- View feedback in real time

> The goal is to eventually host this as a live project in my portfolio.

---

## Future Enhancements

-  **Add more voice samples** with different accents, tones, and modulations.
-  Use **Deep Learning (e.g., CNN, LSTM)** for better accuracy.
-  Deploy to a **live web platform** for demo access.
-  Add analytics and graphs for prediction confidence.

---

## Conclusion

This project gave me practical experience with audio processing, machine learning, and frontend development using Streamlit. It highlights how voice features can reveal emotional states and how ML can classify them effectively.

> More data = better results!  
> This is a foundational project that I plan to improve and expand.

---

##  How to Run

### Prerequisites

Install required packages:
```bash
pip install streamlit librosa sounddevice scikit-learn joblib numpy
