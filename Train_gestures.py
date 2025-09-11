import cv2
import mediapipe as mp
import numpy as np
from sklearn import svm
import pickle
import os
import time

class GestureTrainer:
    def __init__(self):
        # Initialize MediaPipe hands for up to 2 hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize variables
        self.data = []
        self.labels = []
        self.gesture_names = {}
        self.data_file = "gesture_data.pkl"
        self.model_file = "gesture_model.pkl"

    def extract_features(self, hand_landmarks_list):
        """Extract normalized features from up to two hands"""
        features = []
        max_features = 84  # 21 landmarks * (x, y) * 2 hands
        
        if not hand_landmarks_list:
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

    def collect_data(self, gesture_id, gesture_name, num_samples=100):
        """Collect training data for a specific gesture continuously after pressing 's'"""
        self.gesture_names[gesture_id] = gesture_name
        cap = cv2.VideoCapture(0)
        collected = 0
        collecting = False
        
        print(f"Collecting {num_samples} samples for gesture: {gesture_name}")
        print("Press 's' to start collecting, 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            status = "Press 's' to start" if not collecting else f"Collecting... ({num_hands} hands)"
            cv2.putText(frame, f"Gesture: {gesture_name} ({collected}/{num_samples}) - {status}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not collecting:
                collecting = True
                print("Started continuous collection")

            if collecting and results.multi_hand_landmarks and collected < num_samples:
                features = self.extract_features(results.multi_hand_landmarks)
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    print("Warning: Invalid features detected (NaN/Inf)")
                    continue
                self.data.append(features)
                self.labels.append(gesture_id)
                collected += 1
                print(f"Collected sample {collected}/{num_samples}")
                time.sleep(0.05)  # Small delay to avoid overly fast collection
            
            if key == ord('q') or collected >= num_samples:
                if collected >= num_samples:
                    print(f"Completed collection of {num_samples} samples for {gesture_name}")
                break

        cap.release()
        cv2.destroyAllWindows()
        self.save_data()

    def save_data(self):
        """Save collected data to file"""
        print("Saving data to", self.data_file)
        with open(self.data_file, 'wb') as f:
            pickle.dump({
                'data': self.data,
                'labels': self.labels,
                'gesture_names': self.gesture_names
            }, f)

    def load_data(self):
        """Load collected data from file"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.data = saved_data['data']
                self.labels = saved_data['labels']
                self.gesture_names = saved_data['gesture_names']
                print(f"Loaded {len(self.data)} samples, {len(self.gesture_names)} gestures")
            return True
        print("No saved data found")
        return False

    def train_model(self):
        """Train SVM classifier with optimized parameters"""
        if not self.data or not self.labels:
            print("No data available for training")
            return False

        print(f"Training with {len(self.data)} samples, {len(set(self.labels))} classes")
        try:
            # Optimized SVM parameters for 20 gestures
            model = svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
            model.fit(self.data, self.labels)
            with open(self.model_file, 'wb') as f:
                pickle.dump(model, f)
            print("Model trained and saved successfully")
            return True
        except Exception as e:
            print("Training failed:", str(e))
            return False

def main():
    trainer = GestureTrainer()
    
    while True:
        print("\nHand Gesture Training System")
        print("1. Collect new gesture data")
        print("2. Train model")
        print("3. Load existing data")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ")

        if choice == '1':
            gesture_id = int(input("Enter gesture ID (number): "))
            gesture_name = input("Enter gesture name: ")
            num_samples = int(input("Enter number of samples to collect: "))
            trainer.collect_data(gesture_id, gesture_name, num_samples)
        
        elif choice == '2':
            if trainer.load_data() or trainer.data:
                trainer.train_model()
            else:
                print("No data available. Please collect data first.")
        
        elif choice == '3':
            trainer.load_data()
        
        elif choice == '4':
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()