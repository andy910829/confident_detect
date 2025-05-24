import cv2
import numpy as np
import face_recognition
import time # Not explicitly used in calculation loops, but good for potential profiling
from collections import deque

class OptimizedConfidenceAnalyzer:
    """
    通過攝影機分析使用者說話時的信心程度
    分析依據:
    1. 眼神接觸的穩定性 (眼睛持續睜開)
    2. 面部表情的穩定性 (嘴部和眉毛的微小移動)
    3. 頭部姿勢的穩定性 (頭部位置的微小移動)
    """

    # --- Configuration Constants ---
    # Scoring Weights
    EYE_CONTACT_WEIGHT = 0.4
    HEAD_STABILITY_WEIGHT = 0.35 # Adjusted weight
    FACIAL_STABILITY_WEIGHT = 0.25 # Adjusted weight for "facial stability"

    # History Lengths
    CONFIDENCE_HISTORY_LEN = 30  # Approx 1 second at 30 FPS for smoothing
    FACIAL_LANDMARKS_HISTORY_LEN = 10
    HEAD_MOVEMENT_HISTORY_LEN = 15

    # Eye Contact Parameters
    EYE_AR_THRESHOLD = 0.20  # Threshold for eye aspect ratio to consider eye open
    MAX_EYE_CONTACT_FRAMES_FOR_SCORE = 30 # Frames of sustained contact for max eye score

    # Stability Scoring Parameters (0-10 scale for components)
    # For head movement, lower is better. movement_score = 10 - min(10, movement / SENSITIVITY)
    HEAD_MOVEMENT_SENSITIVITY_DIVISOR = 2.0
    # For facial features, lower change is better. stability_score = 10 - min(10, avg_change * SENSITIVITY)
    FACIAL_FEATURE_CHANGE_SENSITIVITY_MULTIPLIER = 2.0

    # Confidence Score Adjustment
    # Each component score (0-10) is compared to NEUTRAL_POINT.
    # The deviation is scaled and then weighted.
    COMPONENT_SCORE_NEUTRAL_POINT = 5.0
    COMPONENT_DEVIATION_SCALE_FACTOR = 0.2 # Scales (score - NEUTRAL_POINT) to a smaller range, e.g., -1 to +1 if score is 0-10
    OVERALL_SENSITIVITY_ADJUSTMENT = 1.0 # Final multiplier for the combined delta
    NO_FACE_PENALTY = 0.75 # Amount to decrease confidence score per frame if no face

    def __init__(self):
        self.cap = None
        self.confidence_score = 50.0  # Start at medium confidence
        self.confidence_history = deque(maxlen=self.CONFIDENCE_HISTORY_LEN)
        
        self.eye_contact_frames = 0 # Renamed from eye_contact_time for clarity (frames)
        self.last_frame_had_eye_contact = False # Renamed from last_eye_contact

        self.facial_landmarks_history = deque(maxlen=self.FACIAL_LANDMARKS_HISTORY_LEN)
        self.head_position_history = deque(maxlen=self.HEAD_MOVEMENT_HISTORY_LEN) # Renamed

        self.confidence_levels = {
            (0, 20): "Very Low - Nervous/Uncertain",
            (20, 40): "Low - Some Hesitation",
            (40, 60): "Medium - Moderately Confident",
            (60, 80): "High - Quite Confident",
            (80, 101): "Very High - Extremely Confident"
        }
        
    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("無法開啟攝影機 (Cannot open camera)")
            return
        
        print("正在分析使用者的信心程度... (Analyzing user confidence...)")
        print("按 'q' 鍵退出 (Press 'q' to quit)")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("無法接收影像 (Cannot receive frame)")
                break
                
            frame = cv2.flip(frame, 1)
            self.analyze_frame(frame)
            self._display_confidence_info(frame) # Renamed for clarity
            cv2.imshow('Confidence Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        
    def analyze_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert BGR (OpenCV default) to RGB (face_recognition default)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog") # "hog" is faster but less accurate than "cnn"
        
        if not face_locations:
            self._update_confidence_score(-self.NO_FACE_PENALTY) # Apply penalty
            # Reset histories that depend on face detection
            self.last_frame_had_eye_contact = False
            self.eye_contact_frames = 0
            # self.facial_landmarks_history.clear() # Optional: clear or let them phase out
            # self.head_position_history.clear()  # Optional
            return
            
        # Assuming one face for simplicity
        face_location = face_locations[0] 
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, [face_location])
        
        eye_contact_score = 0
        head_stability_score = self.COMPONENT_SCORE_NEUTRAL_POINT # Default to neutral if issues
        facial_stability_score = self.COMPONENT_SCORE_NEUTRAL_POINT

        if face_landmarks_list:
            landmarks = face_landmarks_list[0]
            self.facial_landmarks_history.append(landmarks)
            
            eye_contact_score = self._analyze_eye_contact(landmarks)
            facial_stability_score = self._analyze_facial_stability() # Renamed
        else:
            # If landmarks fail but face was detected, penalize less or differently
            self.last_frame_had_eye_contact = False
            self.eye_contact_frames = max(0, self.eye_contact_frames -2) # reduce eye contact faster
            eye_contact_score = self._get_current_eye_contact_score()


        head_stability_score = self._analyze_head_stability(face_location)
        
        # Calculate deltas for each component based on deviation from neutral
        eye_delta = (eye_contact_score - self.COMPONENT_SCORE_NEUTRAL_POINT) * self.COMPONENT_DEVIATION_SCALE_FACTOR
        head_delta = (head_stability_score - self.COMPONENT_SCORE_NEUTRAL_POINT) * self.COMPONENT_DEVIATION_SCALE_FACTOR
        facial_delta = (facial_stability_score - self.COMPONENT_SCORE_NEUTRAL_POINT) * self.COMPONENT_DEVIATION_SCALE_FACTOR
        
        # Combine weighted deltas
        combined_delta = (eye_delta * self.EYE_CONTACT_WEIGHT +
                          head_delta * self.HEAD_STABILITY_WEIGHT +
                          facial_delta * self.FACIAL_STABILITY_WEIGHT)
        
        # Apply overall sensitivity
        final_delta = combined_delta * self.OVERALL_SENSITIVITY_ADJUSTMENT
        
        self._update_confidence_score(final_delta)

    def _get_current_eye_contact_score(self):
        """Helper to calculate eye contact score based on current frame count."""
        # Score between 0 and 10. Max score at MAX_EYE_CONTACT_FRAMES_FOR_SCORE.
        return min(10.0, (self.eye_contact_frames / self.MAX_EYE_CONTACT_FRAMES_FOR_SCORE) * 10.0)

    def _analyze_eye_contact(self, landmarks):
        try:
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            
            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            
            avg_ear = (left_ear + right_ear) / 2.0
            
            is_eye_open_now = avg_ear > self.EYE_AR_THRESHOLD
            
            if is_eye_open_now:
                if self.last_frame_had_eye_contact:
                    self.eye_contact_frames += 1
                else: # Start or restart of eye contact
                    self.eye_contact_frames = 1 
                self.last_frame_had_eye_contact = True
            else:
                self.last_frame_had_eye_contact = False
                # Penalize losing eye contact more rapidly
                self.eye_contact_frames = max(0, self.eye_contact_frames - 2) 
            
            return self._get_current_eye_contact_score()
            
        except (KeyError, IndexError, TypeError):
            self.last_frame_had_eye_contact = False # Reset on error
            self.eye_contact_frames = max(0, self.eye_contact_frames -2)
            return self._get_current_eye_contact_score() # Return current score, likely decreasing
    
    def _eye_aspect_ratio(self, eye):
        if len(eye) != 6: return 0 # Should be 6 points for standard EAR
        # Vertical distances
        v1 = self._distance(eye[1], eye[5])
        v2 = self._distance(eye[2], eye[4])
        # Horizontal distance
        h = self._distance(eye[0], eye[3])
        
        if h == 0: return 0
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _analyze_head_stability(self, face_location_scaled):
        # face_location is (top, right, bottom, left) in the small_frame
        # Convert to center (x,y) for easier tracking
        top, right, bottom, left = face_location_scaled
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        current_head_pos = (center_x, center_y)

        self.head_position_history.append(current_head_pos)
        
        if len(self.head_position_history) < 5: # Need enough history
            return self.COMPONENT_SCORE_NEUTRAL_POINT 
        
        movements = []
        for i in range(1, len(self.head_position_history)):
            prev_pos = self.head_position_history[i-1]
            curr_pos = self.head_position_history[i]
            movement_dist = self._distance(prev_pos, curr_pos)
            movements.append(movement_dist)
        
        avg_movement = sum(movements) / len(movements)
        
        # Higher movement means lower stability score
        # Score from 0-10. If avg_movement is 0, score is 10.
        # If avg_movement is high (e.g., 20 pixels on small frame), score approaches 0.
        stability_score = 10.0 - min(10.0, avg_movement / self.HEAD_MOVEMENT_SENSITIVITY_DIVISOR)
        return stability_score
    
    def _analyze_facial_stability(self): # Renamed from _analyze_micro_expressions
        if len(self.facial_landmarks_history) < 5: # Need enough history
            return self.COMPONENT_SCORE_NEUTRAL_POINT
        
        try:
            # Analyze stability of mouth and eyebrows
            mouth_stability = (self._calculate_feature_stability('top_lip') +
                               self._calculate_feature_stability('bottom_lip')) / 2.0
            
            eyebrow_stability = (self._calculate_feature_stability('left_eyebrow') +
                                 self._calculate_feature_stability('right_eyebrow')) / 2.0
            
            # Combine: e.g., mouth 60%, eyebrows 40%
            # Higher stability means less movement, thus higher score
            facial_stability_score = mouth_stability * 0.6 + eyebrow_stability * 0.4
            return facial_stability_score
            
        except (KeyError, IndexError, TypeError):
            return self.COMPONENT_SCORE_NEUTRAL_POINT # Default on error
    
    def _calculate_feature_stability(self, feature_name):
        # Collect recent feature points
        recent_feature_frames = []
        for landmarks in list(self.facial_landmarks_history)[-5:]: # Get last 5, or fewer if not available
            if feature_name in landmarks:
                recent_feature_frames.append(landmarks[feature_name])
        
        if len(recent_feature_frames) < 2:
            return self.COMPONENT_SCORE_NEUTRAL_POINT # Not enough data
        
        # Calculate average change between consecutive frames for this feature
        total_avg_changes_for_feature = []
        for i in range(1, len(recent_feature_frames)):
            points_prev_frame = recent_feature_frames[i-1]
            points_curr_frame = recent_feature_frames[i]
            
            if len(points_prev_frame) != len(points_curr_frame) or not points_curr_frame:
                continue # Skip if landmark count mismatch or empty

            frame_change = 0
            num_points = len(points_curr_frame)
            for j in range(num_points):
                point_change = self._distance(points_curr_frame[j], points_prev_frame[j])
                frame_change += point_change
            
            avg_change_this_frame_pair = frame_change / num_points if num_points > 0 else 0
            total_avg_changes_for_feature.append(avg_change_this_frame_pair)
        
        if not total_avg_changes_for_feature:
             return self.COMPONENT_SCORE_NEUTRAL_POINT

        overall_avg_change = sum(total_avg_changes_for_feature) / len(total_avg_changes_for_feature)
        
        # Higher change means lower stability score
        # Score 0-10. If overall_avg_change is 0, score is 10.
        stability_score = 10.0 - min(10.0, overall_avg_change * self.FACIAL_FEATURE_CHANGE_SENSITIVITY_MULTIPLIER)
        return stability_score
    
    def _update_confidence_score(self, delta):
        # Update the "raw" confidence score
        self.confidence_score += delta
        self.confidence_score = max(0.0, min(100.0, self.confidence_score))
        
        # Add to history for smoothing
        self.confidence_history.append(self.confidence_score)
        
        # The score used for display will be the smoothed version
        # No, self.confidence_score itself becomes the smoothed score, which is fine.
        # The displayed score IS self.confidence_score after this block.
        if self.confidence_history: # Ensure not empty
             self.confidence_score = sum(self.confidence_history) / len(self.confidence_history)

    def _display_confidence_info(self, frame): # Renamed
        height, width = frame.shape[:2]
        
        # Use the (now smoothed) self.confidence_score for display
        current_display_score = self.confidence_score

        # Bottom overlay
        overlay_height = 120
        cv2.rectangle(frame, (0, height - overlay_height), (width, height), (20, 20, 20), -1)
        
        # Confidence bar
        bar_margin_x = int(width * 0.05)
        bar_width = width - 2 * bar_margin_x
        bar_height = 30
        bar_y = height - 60 # Adjusted position

        # Bar background
        cv2.rectangle(frame, (bar_margin_x, bar_y), (bar_margin_x + bar_width, bar_y + bar_height), (80, 80, 80), -1)
        
        # Confidence indicator part of the bar
        confidence_fill_width = int(bar_width * (current_display_score / 100.0))
        
        if current_display_score < 40: color = (0, 0, 200)       # Red
        elif current_display_score < 60: color = (0, 165, 255) # Orange/Yellow
        else: color = (0, 200, 0)                             # Green
            
        cv2.rectangle(frame, (bar_margin_x, bar_y), (bar_margin_x + confidence_fill_width, bar_y + bar_height), color, -1)
        cv2.rectangle(frame, (bar_margin_x, bar_y), (bar_margin_x + bar_width, bar_y + bar_height), (200,200,200), 1) # Border for bar

        # Percentage text
        cv2.putText(frame, f"{int(current_display_score)}%", 
                   (bar_margin_x + bar_width + 10, bar_y + bar_height - 5),  
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        
        # Confidence level description
        confidence_text = "分析中... (Analyzing...)"
        for (low, high), description in self.confidence_levels.items():
            if low <= current_display_score < high:
                confidence_text = description
                break
        
        # Position text above the bar
        text_size, _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = bar_margin_x 
        text_y = bar_y - 15
        cv2.putText(frame, confidence_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

if __name__ == "__main__":
    analyzer = OptimizedConfidenceAnalyzer()
    analyzer.start()