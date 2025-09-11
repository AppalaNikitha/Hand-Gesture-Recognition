import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque
import logging

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('mediapipe').setLevel(logging.ERROR)

class GesturePredictor:
    def __init__(self):
        # Initialize MediaPipe hands for up to 2 hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
            
        # Initialize variables
        self.model = None
        self.gesture_names = {}
        self.data_file = "gesture_data.pkl"
        self.model_file = "gesture_model.pkl"
        
        # Smoothing and confidence parameters (tuned for 20 gestures)
        self.prediction_history = deque(maxlen=7)  # Larger window for stability
        self.confidence_threshold = 0.85  # Stricter threshold for 20 classes

    def extract_features(self, hand_landmarks_list):
        """Extract normalized features from up to two hands"""
        features = []
        max_features = 84  # 21 landmarks * (x, y) * 2 hands
        
        if not hand_landmarks_list:
            print("No hands detected, returning zero features")
            return np.zeros(max_features)
        
        for hand_landmarks in hand_landmarks_list:
            ref_x, ref_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
            for landmark in hand_landmarks.landmark:
                rel_x = landmark.x - ref_x
                rel_y = landmark.y - ref_y
                features.extend([rel_x, rel_y])
        
        # Pad with zeros if only one hand
        while len(features) < max_features:
            features.append(0.0)
            
        features = np.array(features)
        if len(features) != max_features:
            print(f"Warning: Feature length {len(features)} != {max_features}")
        return features

    def load_data(self):
        """Load gesture names from file"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.gesture_names = saved_data['gesture_names']
                print(f"Loaded gesture names: {self.gesture_names}")
            return True
        print("No gesture data found")
        return False

    def load_model(self):
        """Load trained model from file"""
        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully")
            return True
        print("No model found")
        return False

    def get_stable_prediction(self, prediction, confidence):
        """Return stable prediction using majority voting"""
        self.prediction_history.append((prediction, confidence))
        
        if len(self.prediction_history) < self.prediction_history.maxlen:
            return prediction, confidence
        
        valid_preds = [pred for pred, conf in self.prediction_history if conf >= self.confidence_threshold]
        
        if not valid_preds:
            return "Not detected", 0.0
        
        pred_counts = {}
        for pred in valid_preds:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        stable_pred = max(pred_counts, key=pred_counts.get)
        avg_conf = np.mean([conf for pred, conf in self.prediction_history if pred == stable_pred])
        
        return stable_pred, avg_conf

    def predict_gesture(self):
        """Real-time gesture prediction with stabilization"""
        if not self.load_model() or not self.load_data():
            print("Model or data not found. Please train model first.")
            return

        cap = cv2.VideoCapture(0)
        print("Starting gesture prediction... Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            prediction = "Not detected"
            confidence = 0

            if results.multi_hand_landmarks and self.model:
                features = self.extract_features(results.multi_hand_landmarks)
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    print("Warning: Invalid features detected (NaN/Inf)")
                else:
                    try:
                        pred_id = self.model.predict([features])[0]
                        confidence = self.model.predict_proba([features])[0].max()
                        if confidence >= self.confidence_threshold:
                            prediction = self.gesture_names.get(pred_id, "Not detected")
                            prediction, confidence = self.get_stable_prediction(prediction, confidence)
                        else:
                            prediction = "Not detected"
                            confidence = 0.0
                    except Exception as e:
                        print(f"Prediction error: {str(e)}")

            cv2.putText(frame, f"Prediction: {prediction} ({confidence:.2f}) - {num_hands} hands", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            cv2.imshow('Gesture Prediction', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    predictor = GesturePredictor()
    predictor.predict_gesture()

if __name__ == "__main__":
    main()