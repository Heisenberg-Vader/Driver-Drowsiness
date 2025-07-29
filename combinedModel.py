# Assembled model of both yawn and eye detection models

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import os
import dlib
from matplotlib import pyplot as plt
import time

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

    if len(faces) == 0:
        print("No faces detected.")
        return False
        
    landmarks = pred(gray, faces[0])    
    crop_feature(img, landmarks, [36, 37, 38, 39, 40, 41], max(pad-50, 0), label="left_eye")
    crop_feature(img, landmarks, [42, 43, 44, 45, 46, 47], max(pad-50, 0), label="right_eye")
    crop_feature(img, landmarks, list(range(48, 60)), max(pad-40, 0), label="mouth")
    
    return True
   
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
                
                res.append(eye_pred[0][0])
                    
            elif feature == 'mouth':
                norm_im = np.expand_dims(norm_im, axis=0)
                yawn_pred = yawn_model.predict(norm_im)
                    
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
        return 'Alert' if comb_pred > drowsy_thresh else 'Drowsy', comb_pred
        
    else:
        print("Failed to get predictions for all features.")
        return "Detection Failed", 0.0
    
cam = cv2.VideoCapture('test.mp4')

if not cam.isOpened():
    raise ValueError("Could not open webcam. Please check your camera settings.")

last_cap_time = time.time()
cap_interval = 1
status_text = "Initializing..."
conf = 0.0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break
    
    display_frame = frame.copy()
    
    cur_time = time.time()
    elapsed = cur_time - last_cap_time
    rem_time = max(0, cap_interval - elapsed)   
    
    cv2.putText(display_frame, f"Status: {status_text} ({conf:.2f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display_frame, f"Next capture in: {int(rem_time)} seconds", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Drowsiness Detection', display_frame)
    
    key = cv2.waitKey(1)
    
    if key % 256 == 27 or key % 256 == ord('q'):
        print('Closing webcam feed.')
        break
    
    if key % 256 == 32 or (elapsed >= cap_interval):
        print("\n--- Capturing and processing frame ---")
        im = frame.copy()

        if im is None:
            raise ValueError("Image not found or could not be read.")

        status = segment_eyes_and_mouth(im, 70)
        if not status:
            print("No face detected or failed to segment features.")
            cv2.putText(display_frame, f"Face not found!", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            continue
        status_text, conf = combine_model_predict(im)

        print(f"Final Prediction: {status_text} with confidence {conf:.2f}\n")
        
        last_cap_time = time.time()

cam.release()
cv2.destroyAllWindows()

#############################################
# FOR TESTING PURPOSES ON INDIVIDUAL FRAMES #
#############################################

# pred_labels = []
# gt_labels = []
# conf_scores = []

# base_dir = 'sampled_frames'
# categories = ['drowsy', 'notdrowsy']

# for category in categories:
#     label = 'Drowsy' if category == 'drowsy' else 'Alert'
#     folder = os.path.join(base_dir, category)

#     for fname in os.listdir(folder):
#         path = os.path.join(folder, fname)
#         img = cv2.imread(path)

#         if img is None:
#             print(f"[WARNING] Could not load {path}")
#             continue

#         gt_labels.append(1 if label == 'Drowsy' else 0) 

#         if segment_eyes_and_mouth(img):
#             prd, conf = combine_model_predict(img)
#             pred_labels.append(1 if prd == 'Drowsy' else 0)
#             conf_scores.append(conf)
#         else:
#             prd = 'Unknown'
#             pred_labels.append(0)
#             conf_scores.append(0)

    
# gt_cln, pred_cln = [], []
# for gt, prd in zip(gt_labels, pred_labels):
#     if prd != 'Unknown':
#         gt_cln.append(gt)
#         pred_cln.append(prd)
        
# cm = confusion_matrix(gt_cln, pred_cln, labels=[0, 1])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Alert', 'Drowsy'])
# disp.plot()
# plt.show()

# TP = cm[0, 0] 
# TN = cm[1, 1]
# FN = cm[0, 1]
# FP = cm[1, 0]

# accuracy = (TP + TN) / cm.sum()
# precision = TP / (TP + FP) if (TP + FP) > 0 else 0
# recall = TP / (TP + FN) if (TP + FN) > 0 else 0
# f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
# specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1-Score: {f1:.2f}")
# print(f"Specificity: {specificity:.2f}")
