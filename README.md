# 💤 Drowsiness Detection Web App

A real-time drowsiness detection prototype that uses your device camera (laptop/mobile/tablet) to capture an image, process it using deep learning models, and predict whether the driver is alert or drowsy.

---

## 🚀 Features

-  Uses your webcam to take a snapshot.
-  Detects eye status (open/closed) using a CNN-based model.
-  Detects yawning using a separate image classifier.
-  Combines both predictions with weighted logic to infer drowsiness.
-  Models are preloaded to avoid lag during prediction.

---

## 🧰 Tech Stack

- **Frontend:** Streamlit + HTML + JS Webcam Interface
- **Backend:** Python 
- **Models:**
  - `eye_classifier_model.keras`
  - `yawn_detection_model.keras`
---

## Multiple Implementations
- The `combinedModel.py` code has various different implementation that can be used for testing, real applications that best fits the use case of the user.
