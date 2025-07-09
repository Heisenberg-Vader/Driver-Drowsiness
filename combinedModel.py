# Assembled model of both yawn and eye detection models

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import os
import dlib
from matplotlib import pyplot as plt

pred = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
det = dlib.get_frontal_face_detector()

eye_thresh = 0.5
yawn_thresh = 0.9
drowsy_thresh = 0.6

def load_model(model_path):
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
    
yawn_model = load_model('yawn_detection_model.keras')
eye_model = load_model('eye_classifier_model.keras')
    
def crop_feature(img, landmarks, indices, pad=10, label="feature"):
    xs = [landmarks.part(i).x for i in indices]
    ys = [landmarks.part(i).y for i in indices]
    
    x_min, x_max = max(0, min(xs) - pad), min(img.shape[1], max(xs) + pad)
    y_min, y_max = max(0, min(ys) - pad), min(img.shape[0], max(ys) + pad)
    
    crop = img[y_min:y_max, x_min:x_max]
    cv2.imwrite(f"test_im/{label}.jpg", crop)
    return crop
    
def segment_eyes_and_mouth(img, pad=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    faces = det(gray)

    for face in faces:
        landmarks = pred(gray, face)
        
        crop_feature(img, landmarks, [36, 37, 38, 39, 40, 41], max(pad-50, 0), label="left_eye")
        crop_feature(img, landmarks, [42, 43, 44, 45, 46, 47], max(pad-50, 0), label="right_eye")
        crop_feature(img, landmarks, list(range(48, 60)), pad, label="mouth")
   
def load_and_normalize_img(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    filename = os.path.basename(path)
    if im is not None:
        if filename.startswith('left_eye') or filename.startswith('right_eye'):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (24, 24))
            norm_im = (im / 255.0).astype(np.float32)
            norm_im = norm_im.reshape((im.shape[0], im.shape[1], 1))
            
            return norm_im
        elif filename.startswith('mouth'):
            im = cv2.resize(im, (224, 224))
            norm_im = (im / 255.0).astype(np.float32)
            norm_im = norm_im.reshape((im.shape[0], im.shape[1], 3))

            return norm_im
        else:
            print(f"Unknown image type for path: {path}")
            return None
            
    else:
        print(f"Failed to load image: {path}")
        return None
        
def combine_model_predict(input_im):
    features = ['left_eye', 'right_eye', 'mouth']
    res = []
    for feature in features:
        norm_im = load_and_normalize_img(f'test_im/{feature}.jpg')
        if norm_im is not None:
            if feature in ['left_eye', 'right_eye']:
                norm_im = np.expand_dims(norm_im, axis=0)
                eye_pred = eye_model.predict(norm_im)
                
                if feature == 'left_eye':
                    if eye_pred[0][0] > eye_thresh:
                        print(f"Left Eye Prediction: Open with confidence {eye_pred[0][0]:.2f}")
                    else:
                        print(f"Left Eye Prediction: Closed with confidence {1 - eye_pred[0][0]:.2f}")
                
                elif feature == 'right_eye':
                    if eye_pred[0][0] > eye_thresh:
                        print(f"Right Eye Prediction: Open with confidence {eye_pred[0][0]:.2f}")
                    else:
                        print(f"Right Eye Prediction: Closed with confidence {1 - eye_pred[0][0]:.2f}")
                        
                res.append(eye_pred[0][0])
                    
            elif feature == 'mouth':
                norm_im = np.expand_dims(norm_im, axis=0)
                yawn_pred = yawn_model.predict(norm_im)
                
                if yawn_pred[0][0] > yawn_thresh:
                    print(f"Mouth Prediction: Yawning with confidence {yawn_pred[0][0]:.2f}")
                else:
                    print(f"Mouth Prediction: Not Yawning with confidence {1 - yawn_pred[0][0]:.2f}")
                    
                res.append(yawn_pred[0][0])
                
            else:
                print(f"Unknown feature: {feature}")
                continue
    
    print(res)
    
    if len(res) == 3:
        eye_pred_combined = 0.5 * (res[0] + res[1]) # 0.5 weight for left and right eye predictions
        yawn_pred_ = res[2]
        
        comb_pred = 0.7 * eye_pred_combined + 0.3 * yawn_pred_ # 0.7 weight for eye prediction, 0.3 for yawn prediction
        
        print(f"[SCAN RESULT] Status: {'Alert' if comb_pred > drowsy_thresh else 'Drowsy'} with confidence {comb_pred:.2f}")
    
im = cv2.imread('t1.jpg')    

if im is None:
    raise ValueError("Image not found or could not be read.")

segment_eyes_and_mouth(im)
combine_model_predict(im)
