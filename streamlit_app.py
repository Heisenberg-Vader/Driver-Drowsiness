# import cv2
# import numpy as np
# import tensorflow as tf
# import streamlit as st
# from tensorflow import keras
# import os
# import dlib
# from PIL import Image

# pred = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# det = dlib.get_frontal_face_detector()

# @st.cache_resource
# def load_models():
#     eye_model = keras.models.load_model('eye_classifier_model.keras')
#     yawn_model = keras.models.load_model('yawn_detection_model.keras')
#     return eye_model, yawn_model

# eye_model, yawn_model = load_models()

# eye_thresh = 0.5
# yawn_thresh = 0.9
# drowsy_thresh = 0.6

# def crop_feature(img, landmarks, indices, pad=10):
#     xs = [landmarks.part(i).x for i in indices]
#     ys = [landmarks.part(i).y for i in indices]
    
#     x_min, x_max = max(0, min(xs) - pad), min(img.shape[1], max(xs) + pad)
#     y_min, y_max = max(0, min(ys) - pad), min(img.shape[0], max(ys) + pad)
    
#     return img[y_min:y_max, x_min:x_max]

# def segment_eyes_and_mouth(img, pad=150):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#     faces = det(gray)

#     if len(faces) == 0:
#         return None, None, None

#     landmarks = pred(gray, faces[0])
#     left_eye = crop_feature(img, landmarks, [36, 37, 38, 39, 40, 41], max(pad-50, 0))
#     right_eye = crop_feature(img, landmarks, [42, 43, 44, 45, 46, 47], max(pad-50, 0))
#     mouth = crop_feature(img, landmarks, list(range(48, 60)), max(pad-40, 0))
#     return left_eye, right_eye, mouth

# def normalize_image(im, feature):
#     if im is None:
#         return None

#     if feature in ['left_eye', 'right_eye']:
#         im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         im = cv2.resize(im, (24, 24))
#         norm_im = (im / 255.0).astype(np.float32)
#         return norm_im.reshape((1, 24, 24, 1))
    
#     elif feature == 'mouth':
#         im = cv2.resize(im, (224, 224))
#         norm_im = (im / 255.0).astype(np.float32)
#         return norm_im.reshape((1, 224, 224, 3))
    
#     return None

# def predict_drowsiness(left_eye, right_eye, mouth):
#     res = []
    
#     left_eye_norm = normalize_image(left_eye, 'left_eye')
#     right_eye_norm = normalize_image(right_eye, 'right_eye')
#     mouth_norm = normalize_image(mouth, 'mouth')

#     if left_eye_norm is not None:
#         left_eye_pred = eye_model.predict(left_eye_norm)
#         res.append(left_eye_pred[0][0])

#     if right_eye_norm is not None:
#         right_eye_pred = eye_model.predict(right_eye_norm)
#         res.append(right_eye_pred[0][0])

#     if mouth_norm is not None:
#         mouth_pred = yawn_model.predict(mouth_norm)
#         res.append(mouth_pred[0][0])

#     if len(res) == 3:
#         print(res)
#         eye_pred = 0.5 * (res[0] + res[1])
#         yawn_pred = res[2]
#         comb_pred = 0.7 * eye_pred + 0.3 * yawn_pred

#         status = "Alert" if comb_pred > drowsy_thresh else "Drowsy"
#         return comb_pred, status
#     else:
#         return None, "Detection Failed"

# # --------------------------
# # Streamlit UI
# # --------------------------
# st.title("Drowsiness Detection Prototype")
# st.write("Upload a photo or use your webcam to test the model.")

# img_input = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
# webcam_img = st.camera_input("Or take a photo")

# input_source = None
# if webcam_img is not None:
#     input_source = webcam_img
# elif img_input is not None:
#     input_source = img_input

# if input_source is not None:
#     img = cv2.cvtColor(np.array(Image.open(input_source)), cv2.COLOR_RGB2BGR)

#     with st.spinner("Processing image..."):
#         left_eye, right_eye, mouth = segment_eyes_and_mouth(img, 50)
#         score, status = predict_drowsiness(left_eye, right_eye, mouth)

#         if score is not None:
#             st.success(f"Prediction: **{status}** (Confidence: {score:.2f})")
#         else:
#             st.error("Could not detect face or landmarks.")

#     if left_eye is not None:
#         st.image(left_eye, caption="Left Eye")
#     if right_eye is not None:
#         st.image(right_eye, caption="Right Eye")
#     if mouth is not None:
#         st.image(mouth, caption="Mouth")
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow import keras
import dlib
from PIL import Image

# Load Dlib face detector and landmark predictor
det = dlib.get_frontal_face_detector()
pred = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Cache model loading
@st.cache_resource
def load_models():
    eye_model = keras.models.load_model('eye_classifier_model.keras')
    yawn_model = keras.models.load_model('yawn_detection_model.keras')
    return eye_model, yawn_model

eye_model, yawn_model = load_models()

# Thresholds
eye_thresh = 0.5
yawn_thresh = 0.9
drowsy_thresh = 0.6

def crop_feature(img, landmarks, indices, pad=10):
    xs = [landmarks.part(i).x for i in indices]
    ys = [landmarks.part(i).y for i in indices]
    
    x_min, x_max = max(0, min(xs) - pad), min(img.shape[1], max(xs) + pad)
    y_min, y_max = max(0, min(ys) - pad), min(img.shape[0], max(ys) + pad)
    
    return img[y_min:y_max, x_min:x_max]

def segment_eyes_and_mouth(img, pad=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = det(gray)

    if len(faces) == 0:
        return None, None, None

    landmarks = pred(gray, faces[0])
    left_eye = crop_feature(img, landmarks, [36, 37, 38, 39, 40, 41], pad=pad-50)
    right_eye = crop_feature(img, landmarks, [42, 43, 44, 45, 46, 47], pad=pad-50)
    mouth = crop_feature(img, landmarks, list(range(48, 60)), pad=pad-40)

    return left_eye, right_eye, mouth

def normalize_image(im, feature):
    if im is None:
        return None

    if feature in ['left_eye', 'right_eye']:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (24, 24))
        norm_im = (im / 255.0).astype(np.float32)
        return norm_im.reshape((1, 24, 24, 1))
    
    elif feature == 'mouth':
        im = cv2.resize(im, (224, 224))
        norm_im = (im / 255.0).astype(np.float32)
        return norm_im.reshape((1, 224, 224, 3))

    return None

def predict_drowsiness(left_eye, right_eye, mouth):
    res = []

    left_eye_norm = normalize_image(left_eye, 'left_eye')
    right_eye_norm = normalize_image(right_eye, 'right_eye')
    mouth_norm = normalize_image(mouth, 'mouth')

    if left_eye_norm is not None:
        left_eye_pred = eye_model.predict(left_eye_norm, verbose=0)
        res.append(left_eye_pred[0][0])
    else:
        res.append(1.0)  # fallback

    if right_eye_norm is not None:
        right_eye_pred = eye_model.predict(right_eye_norm, verbose=0)
        res.append(right_eye_pred[0][0])
    else:
        res.append(1.0)

    if mouth_norm is not None:
        mouth_pred = yawn_model.predict(mouth_norm, verbose=0)
        res.append(mouth_pred[0][0])
    else:
        res.append(0.0)

    eye_pred = 0.5 * (res[0] + res[1])
    yawn_pred = res[2]
    comb_pred = 0.7 * eye_pred + 0.3 * yawn_pred

    status = "Alert" if comb_pred > drowsy_thresh else "Drowsy"
    return comb_pred, status, res

############################
####    Streamlit UI    ####
############################
st.title("🚨 Drowsiness Detection")
st.write("Upload a photo or use your webcam to check if you're alert or drowsy.")

img_input = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="upload_image")
webcam_img = st.camera_input("Or take a live photo", key="camera_image")

if "confirmed_image" not in st.session_state:
    st.session_state.confirmed_image = None

if webcam_img is not None:
    st.image(webcam_img, caption="Captured Photo (Webcam)", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Retake"):
            st.session_state.confirmed_image = None
            st.experimental_rerun()

    with col2:
        if st.button("Continue"):
            st.session_state.confirmed_image = webcam_img

elif img_input is not None:
    st.image(img_input, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Confirm Upload"):
        st.session_state.confirmed_image = img_input

if st.session_state.confirmed_image:
    img = cv2.cvtColor(np.array(Image.open(st.session_state.confirmed_image)), cv2.COLOR_RGB2BGR)
    st.image(img, caption="Final Image Used for Prediction", use_container_width=True)

    with st.spinner("Detecting drowsiness..."):
        left_eye, right_eye, mouth = segment_eyes_and_mouth(img, pad=70)

        if any(x is None for x in [left_eye, right_eye, mouth]):
            st.error("Face or facial landmarks not detected properly.")
        else:
            score, status, raw = predict_drowsiness(left_eye, right_eye, mouth)

            st.success(f"### {status.upper()} (Confidence: {score:.2f})")
            st.write(f"🔵 Left Eye: {raw[0]:.2f} | 🔵 Right Eye: {raw[1]:.2f} | 🔴 Yawn: {raw[2]:.2f}")
            st.image(left_eye, caption="Left Eye")
            st.image(right_eye, caption="Right Eye")
            st.image(mouth, caption="Mouth")
