import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time

class HandTracker:
    def __init__(self, model_path='hand_landmarker.task'):
        # Initialize the Hand Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, img, timestamp_ms):
        # Convert the image to MediaPipe format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Detect landmarks in the video frame
        self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        return self.results

    def get_landmarks(self, img_shape):
        lm_list = []
        if self.results and self.results.hand_landmarks:
            h, w = img_shape[:2]
            # Since we only track 1 hand, we take the first one
            for landmark in self.results.hand_landmarks[0]:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                lm_list.append([0, cx, cy]) # Matching old format [id, x, y] (id placeholder)
            # Add IDs correctly
            for i in range(len(lm_list)):
                lm_list[i][0] = i
        return lm_list

    def draw_landmarks(self, img, lm_list):
        if not lm_list:
            return img
            
        # Draw connections (simplified)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8), # Index
            (9, 10), (10, 11), (11, 12),    # Middle
            (13, 14), (14, 15), (15, 16),   # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)       # Palm
        ]
        
        for p1, p2 in connections:
            cv2.line(img, (lm_list[p1][1], lm_list[p1][2]), (lm_list[p2][1], lm_list[p2][2]), (255, 255, 255), 1)
            
        for id, cx, cy in lm_list:
            cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
            
        return img

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    start_time = time.time()
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1) # Mirror image
        
        # Current timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        # Detect hands
        results = tracker.find_hands(img, timestamp_ms)
        lm_list = tracker.get_landmarks(img.shape)
        
        if len(lm_list) != 0:
            # Draw the hand
            img = tracker.draw_landmarks(img, lm_list)
            
            # Index finger tip is 8, Thumb tip is 4
            x1, y1 = lm_list[8][1], lm_list[8][2]
            x2, y2 = lm_list[4][1], lm_list[4][2]
            
            # Draw circle at index finger tip
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            
            # Calculate distance to detect pinch
            length = math.hypot(x2 - x1, y2 - y1)
            
            # If pinched (distance is small), turn green to indicate drawing
            if length < 40:
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "State: DRAWING", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            else:
                cv2.putText(img, "State: HOVERING", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("MagicFlow - Core Tracker", img)
        if cv2.waitKey(1) & 0xFF == 27: # ESC to stop
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
