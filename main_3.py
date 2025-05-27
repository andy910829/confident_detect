import cv2
import numpy as np
import face_recognition
import time
from collections import deque, defaultdict
import mediapipe as mp
# from scipy.spatial.distance import euclidean # Removed: Unused
import warnings
warnings.filterwarnings("ignore")

class ConfidenceDetector:
    def __init__(self):
        # 初始化MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 人臉追蹤和信心分數管理
        self.face_trackers = {}  # 儲存每個人臉的追蹤資訊
        self.face_id_counter = 0
        self.face_tolerance = 0.6
        
        # 微表情偵測相關參數
        self.emotion_history_length = 30  # 保存30幀的情緒歷史
        self.baseline_frames = 60  # 建立基準線需要的幀數
        self.micro_expression_history_length = 15 # For rigidity calculation
        self.min_duration_for_rate_calc = 10 # seconds, for blink rate

        # 關鍵面部標記點（基於MediaPipe 468點模型） - Removed: This specific dict was not directly used as AUs had their own lists.
        # self.facial_landmarks = { ... }
        
        # 動作單位(AU)對應的標記點 - Updated AU1, AU2 landmarks
        self.action_units = {
            'AU1': {'landmarks': [63, 105, 66, 293, 334, 296], 'name': '眉毛內側上抬'}, # Inner brows
            'AU2': {'landmarks': [70, 52, 53, 300, 282, 283], 'name': '眉毛外側上抬'}, # Outer brows (70/300 are arch points)
            'AU5': {'landmarks': [159, 145, 386, 374], 'name': '上眼瞼抬高'}, # Using top/bottom eyelid points for openness
            'AU6': {'landmarks': [117, 118, 119, 346, 347, 348, 205, 206], 'name': '臉頬上提'}, # Example cheek landmarks
            'AU9': {'landmarks': [61, 291, 40, 39, 37, 0, 267, 269, 270, 80, 81, 82], 'name': '鼻翼皺起/上唇抬起'}, # Combined for simplicity
            'AU10': {'landmarks': [40, 39, 37, 0, 267, 269, 270, 13, 14], 'name': '上唇中部抬起'}, # Upper lip center
            'AU12': {'landmarks': [61, 291], 'name': '嘴角上拉'}, # Mouth corners
            'AU14': {'landmarks': [61, 291], 'name': '酒窩/單側嘴角'}, # Mouth corners for asymmetry check
            'AU15': {'landmarks': [61, 291], 'name': '嘴角下拉'}, # Mouth corners
            'AU20': {'landmarks': [61, 291], 'name': '嘴角水平拉伸'} # Mouth corners for width
        }

    def get_face_encoding(self, face_image):
        """獲取人臉編碼用於識別"""
        try:
            # Ensure face_image is BGR
            if face_image.shape[2] == 3: # If it has 3 channels
                 rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else: # Grayscale, convert to RGB by duplicating channels
                 rgb_face = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            encodings = face_recognition.face_encodings(rgb_face)
            return encodings[0] if encodings else None
        except Exception as e:
            # print(f"Error in get_face_encoding: {e}")
            return None

    def find_or_create_face_id(self, face_encoding, face_bbox):
        """找到現有人臉ID或創建新ID"""
        if face_encoding is None:
             # Try to find by bbox overlap if encoding failed but tracking is active
            for face_id, tracker_data in self.face_trackers.items():
                # Simple IoU or centroid distance could be used here
                # For now, let's assume if encoding fails, we can't reliably ID for new faces
                pass # Could add logic to re-associate based on bbox proximity for existing tracks
            return None # If encoding fails, don't create a new ID easily to avoid too many false IDs.
            
        # 與已知人臉比較
        for face_id, tracker in self.face_trackers.items():
            if tracker['encoding'] is not None:
                # Check if tracker['encoding'] is a valid encoding (list or ndarray)
                if isinstance(tracker['encoding'], (list, np.ndarray)) and len(tracker['encoding']) > 0:
                    known_encodings = [enc for enc in [tracker['encoding']] if enc is not None and len(enc) > 0]
                    if not known_encodings:
                        continue
                    try:
                        distance = face_recognition.face_distance(known_encodings, face_encoding)[0]
                        if distance < self.face_tolerance:
                            return face_id
                    except Exception as e:
                        # print(f"Error comparing face encodings: {e}")
                        continue # Skip if comparison fails
                
        
        # 創建新的人臉ID
        self.face_id_counter += 1
        new_face_id = f"face_{self.face_id_counter}"
        current_time = time.time()
        
        self.face_trackers[new_face_id] = {
            'encoding': face_encoding,
            'confidence_score': 50,  # 初始信心分數
            'emotion_history': deque(maxlen=self.emotion_history_length),
            'baseline_emotions': {},
            'baseline_established': False,
            'baseline_frame_count': 0,
            'last_seen': current_time,
            'first_seen_time': current_time, # For blink rate
            'bbox': face_bbox,
            'micro_expression_count': defaultdict(int),
            'micro_expression_history': deque(maxlen=self.micro_expression_history_length), # For rigidity
            'rigidity_score': 0.0, # Initialize rigidity score
            'eye_contact_ratio': 0.5, # Initial neutral
            'blink_rate': 0,
            'last_blink_time': current_time, # Initialize last_blink_time
            'blink_count': 0
        }
        
        return new_face_id

    def extract_facial_features(self, landmarks, image_shape):
        """提取面部特徵點座標"""
        h, w = image_shape[:2]
        features = {}
        
        for au_name, au_info in self.action_units.items():
            points = []
            for idx in au_info['landmarks']:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    points.append([point.x * w, point.y * h])
            features[au_name] = np.array(points)
        
        return features

    def detect_micro_expressions(self, features, image_h, face_id): # Added image_h
        """偵測微表情和動作單位. Higher value = more expression intensity."""
        tracker = self.face_trackers[face_id]
        micro_expressions = {}
        
        # For y-coordinates, smaller y means higher on screen.
        # We want larger value = more expression. So, for "up" movements, use image_h - y or -y.
        
        # AU1: 眉毛內側上抬 (恐懼指標)
        if 'AU1' in features and features['AU1'].size > 0:
            eyebrow_inner_height = np.mean(image_h - features['AU1'][:, 1]) 
            micro_expressions['fear_eyebrow_inner'] = eyebrow_inner_height
        
        # AU2: 眉毛外側上抬 (恐懼指標)
        if 'AU2' in features and features['AU2'].size > 0:
            eyebrow_outer_height = np.mean(image_h - features['AU2'][:, 1])
            micro_expressions['fear_eyebrow_outer'] = eyebrow_outer_height

        # AU5: 上眼瞼抬高 (恐懼指標) - Using specific landmarks for eye openness
        if 'AU5' in features and features['AU5'].size >= 4: # Ensure we have points for both eyes
            # landmarks: [159(L_top), 145(L_bottom), 386(R_top), 374(R_bottom)]
            left_eye_openness = abs(features['AU5'][0, 1] - features['AU5'][1, 1]) # y_top_L - y_bottom_L
            right_eye_openness = abs(features['AU5'][2, 1] - features['AU5'][3, 1]) # y_top_R - y_bottom_R
            avg_eye_openness = (left_eye_openness + right_eye_openness) / 2
            micro_expressions['fear_eyelid'] = avg_eye_openness # Higher value is more open

        # AU9 & AU10: 上唇抬起 (厭惡指標)
        if 'AU10' in features and features['AU10'].size > 0: # Using AU10 for upper lip center
            upper_lip_raise = np.mean(image_h - features['AU10'][:, 1])
            micro_expressions['disgust_lip'] = upper_lip_raise
        
        # AU12: 嘴角上拉 (快樂指標，可能是"欺騙的快樂") - Check asymmetry of mouth corners
        if 'AU12' in features and features['AU12'].size >= 2:
            mouth_corners_y = image_h - features['AU12'][:, 1] # Higher value means more pulled up
            # Asymmetry in how much they are pulled up (relative to neutral or just y-diff)
            # For deceptive happiness, it's often an asymmetric smile.
            # We could also measure the upward pull itself.
            # Let's use asymmetry of the Y positions of the corners.
            if len(mouth_corners_y) >= 2:
                 micro_expressions['happiness_asymmetry'] = abs(mouth_corners_y[0] - mouth_corners_y[1])
                 micro_expressions['smile_intensity'] = np.mean(mouth_corners_y) # Average upward pull

        # AU14: 單側嘴角緊繃 (輕蔑指標) - Asymmetry
        if 'AU14' in features and features['AU14'].size >= 2:
            # Using y-coordinates of mouth corners [61, 291]
            left_corner_y = features['AU14'][0, 1]
            right_corner_y = features['AU14'][1, 1]
            micro_expressions['contempt_asymmetry'] = abs(left_corner_y - right_corner_y)
        
        # AU15: 嘴角下拉 (悲傷指標)
        if 'AU15' in features and features['AU15'].size >=2:
            # features['AU15'] are mouth corners. Higher y means more droop.
            mouth_droop = np.mean(features['AU15'][:, 1]) 
            micro_expressions['sadness_droop'] = mouth_droop
        
        # AU20: 嘴角水平拉伸 (恐懼指標)
        if 'AU20' in features and features['AU20'].size >= 2:
            # features['AU20'] are mouth corners [61, 291]
            mouth_width = abs(features['AU20'][0, 0] - features['AU20'][1, 0])
            micro_expressions['fear_mouth_stretch'] = mouth_width
        
        if face_id and face_id in self.face_trackers:
             self.face_trackers[face_id]['micro_expression_history'].append(micro_expressions.copy())
        return micro_expressions

    def establish_baseline(self, micro_expressions, face_id):
        """建立個人行為基準線"""
        tracker = self.face_trackers[face_id]
        
        if not tracker['baseline_established']:
            for expr_name, value in micro_expressions.items():
                if expr_name not in tracker['baseline_emotions']:
                    tracker['baseline_emotions'][expr_name] = []
                tracker['baseline_emotions'][expr_name].append(value)
            
            tracker['baseline_frame_count'] += 1
            
            if tracker['baseline_frame_count'] >= self.baseline_frames:
                for expr_name, values in tracker['baseline_emotions'].items():
                    if not values: continue # Skip if no values recorded
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    # For "raise" or "increase" expressions, threshold is higher
                    # For "droop" or other types, adjust accordingly if needed, but consistent "higher value = more expression" helps.
                    threshold_val = mean_val + 2 * std_val 
                    # For expressions where a lower value means more (e.g. if not using image_h - y), it would be mean - 2 * std
                    
                    tracker['baseline_emotions'][expr_name] = {
                        'mean': mean_val,
                        'std': std_val,
                        'threshold': threshold_val,
                        'raw_values': values # Keep raw values if needed for other analysis
                    }
                tracker['baseline_established'] = True
                # print(f"Baseline established for {face_id}")

    def detect_deception_indicators(self, micro_expressions, face_id):
        """偵測欺騙指標"""
        tracker = self.face_trackers[face_id]
        deception_score = 0
        indicators = []
        
        if not tracker['baseline_established']:
            return deception_score, indicators
        
        # Fear indicators: eyebrow inner/outer raise, eyelid raise, mouth stretch
        fear_indicators_config = {
            'fear_eyebrow_inner': {'weight': 4, 'name': "Inner Eyebrow Raise (Fear)"},
            'fear_eyebrow_outer': {'weight': 4, 'name': "Outer Eyebrow Raise (Fear)"},
            'fear_eyelid': {'weight': 8, 'name': "Eyelid Raise (Fear)"}, # Usually more prominent
            'fear_mouth_stretch': {'weight': 6, 'name': "Mouth Stretch (Fear)"}
        }

        for indicator, config in fear_indicators_config.items():
            if indicator in micro_expressions and indicator in tracker['baseline_emotions']:
                current_value = micro_expressions[indicator]
                baseline = tracker['baseline_emotions'][indicator]
                if current_value > baseline['threshold']: # Higher value means more expression intensity
                    deception_score += config['weight']
                    indicators.append(config['name'])
                    tracker['micro_expression_count'][indicator] += 1
        
        # Disgust micro-expression
        if 'disgust_lip' in micro_expressions and 'disgust_lip' in tracker['baseline_emotions']:
            current_value = micro_expressions['disgust_lip']
            baseline = tracker['baseline_emotions']['disgust_lip']
            if current_value > baseline['threshold']:
                deception_score += 6
                indicators.append("Disgust Micro-expression (Lip)")
                tracker['micro_expression_count']['disgust'] += 1
        
        # Contempt micro-expression (asymmetry)
        if 'contempt_asymmetry' in micro_expressions and 'contempt_asymmetry' in tracker['baseline_emotions']:
            current_value = micro_expressions['contempt_asymmetry']
            baseline = tracker['baseline_emotions']['contempt_asymmetry']
            if current_value > baseline['threshold']: # Higher asymmetry
                deception_score += 7
                indicators.append("Contempt Micro-expression (Asymmetry)")
                tracker['micro_expression_count']['contempt'] += 1
        
        # Sadness micro-expression
        if 'sadness_droop' in micro_expressions and 'sadness_droop' in tracker['baseline_emotions']:
            current_value = micro_expressions['sadness_droop']
            baseline = tracker['baseline_emotions']['sadness_droop']
            if current_value > baseline['threshold']: # More droop
                deception_score += 5
                indicators.append("Sadness Micro-expression (Droop)")
                tracker['micro_expression_count']['sadness'] += 1
        
        # Deceptive happiness (asymmetric smile)
        if 'happiness_asymmetry' in micro_expressions and 'happiness_asymmetry' in tracker['baseline_emotions']:
            # Also check if smile_intensity is present and above a certain level to qualify as a smile
            smile_intensity_check = True
            if 'smile_intensity' in micro_expressions and 'smile_intensity' in tracker['baseline_emotions']:
                 smile_intensity_check = micro_expressions['smile_intensity'] > tracker['baseline_emotions']['smile_intensity']['mean'] # Basic check

            if smile_intensity_check:
                current_value = micro_expressions['happiness_asymmetry']
                baseline = tracker['baseline_emotions']['happiness_asymmetry']
                if current_value > baseline['threshold']: # More asymmetry
                    deception_score += 12
                    indicators.append("Deceptive Happiness (Asymmetric Smile)")
                    tracker['micro_expression_count']['fake_happiness'] += 1
        
        return deception_score, indicators

    def calculate_rigidity_score(self, face_id):
        tracker = self.face_trackers[face_id]
        if not tracker['baseline_established'] or len(tracker['micro_expression_history']) < self.micro_expression_history_length // 2:
            return 0.0 # Not enough data or baseline not ready

        rigidity_level = 0
        num_rigidity_metrics_checked = 0

        # Metrics that indicate movement. Low variance in these might mean rigidity.
        # We need their baseline std to compare against.
        rigidity_check_metrics = ['fear_eyebrow_inner', 'fear_eyebrow_outer', 'disgust_lip', 'smile_intensity']
        
        for metric_name in rigidity_check_metrics:
            if metric_name in tracker['baseline_emotions']:
                history_values = [hist[metric_name] for hist in tracker['micro_expression_history'] if metric_name in hist]
                if len(history_values) < self.micro_expression_history_length // 3:
                    continue

                current_std = np.std(history_values)
                baseline_std = tracker['baseline_emotions'][metric_name]['std']
                num_rigidity_metrics_checked +=1

                # If current movement (std) is significantly less than baseline movement, or very low absolutely
                # Adding a small epsilon to baseline_std to avoid division by zero or extreme ratios
                if current_std < 0.3 * (baseline_std + 1e-6) or current_std < 0.05 * np.abs(tracker['baseline_emotions'][metric_name]['mean'] + 1e-6): 
                    rigidity_level += 1
        
        if num_rigidity_metrics_checked == 0:
            return 0.0
            
        # Normalize rigidity score (0 to 1, where 1 is highly rigid)
        tracker['rigidity_score'] = rigidity_level / num_rigidity_metrics_checked 
        return tracker['rigidity_score']


    def calculate_confidence_score(self, deception_score, face_id):
        """計算信心分數"""
        tracker = self.face_trackers[face_id]
        
        # Calculate rigidity before using it
        current_rigidity = self.calculate_rigidity_score(face_id) # This updates tracker['rigidity_score']

        confidence_score = 50 # Start from a neutral base
        confidence_score -= deception_score
        
        if tracker['rigidity_score'] > 0.6: # If more than 60% of checked metrics show rigidity
            confidence_score -= 10 # Increased penalty for rigidity
        elif tracker['rigidity_score'] > 0.3:
            confidence_score -=5

        if tracker['eye_contact_ratio'] < 0.3:
            confidence_score -= 6
        elif tracker['eye_contact_ratio'] > 0.9: # Sustained, potentially unnatural staring
            confidence_score -= 4
        else: # Good eye contact
            confidence_score += 3
        
        # Blink rate assessment (per minute)
        if tracker['blink_rate'] > 35: # High blink rate
            confidence_score -= 7
        elif tracker['blink_rate'] < 8 and tracker['baseline_established']: # Low blink rate (if baseline established to avoid penalizing early on)
             confidence_score -= 4
        
        confidence_score = max(0, min(100, confidence_score))
        
        tracker['emotion_history'].append(confidence_score)
        # Smoothed confidence score
        if len(tracker['emotion_history']) > 0:
             # Use a weighted average giving more importance to recent scores
            weights = np.exp(np.linspace(-1., 0., len(tracker['emotion_history'])))
            weights /= weights.sum()
            weighted_avg_score = np.dot(list(tracker['emotion_history']), weights)
            tracker['confidence_score'] = weighted_avg_score
        else:
            tracker['confidence_score'] = confidence_score # Should not happen if emotion_history appends first
            
        return tracker['confidence_score']

    def get_confidence_level(self, score):
        """將信心分數轉換為等級"""
        if score >= 75: # Adjusted thresholds slightly
            return "Very Confident", (0, 255, 0)
        elif score >= 60:
            return "Confident", (0, 200, 100)
        elif score >= 40:
            return "Neutral", (0, 255, 255)
        elif score >= 25:
            return "Not Confident", (0, 165, 255)
        else:
            return "Very Not Confident", (0, 0, 255)

    def detect_eye_contact_and_blinks(self, landmarks_mp, face_id, image_h, image_w):
        tracker = self.face_trackers[face_id]
        
        # Eye landmarks (MediaPipe):
        # Left eye: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        # Right eye: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        # For eye openness:
        # Left: vertical distance between 159 (top) and 145 (bottom)
        # Right: vertical distance between 386 (top) and 374 (bottom)
        # Horizontal distances for normalization:
        # Left: 133 (corner) and 33 (corner)
        # Right: 362 (corner) and 263 (corner)

        try:
            left_eye_top_y = landmarks_mp.landmark[159].y * image_h
            left_eye_bottom_y = landmarks_mp.landmark[145].y * image_h
            left_eye_left_x = landmarks_mp.landmark[133].x * image_w
            left_eye_right_x = landmarks_mp.landmark[33].x * image_w

            right_eye_top_y = landmarks_mp.landmark[386].y * image_h
            right_eye_bottom_y = landmarks_mp.landmark[374].y * image_h
            right_eye_left_x = landmarks_mp.landmark[362].x * image_w
            right_eye_right_x = landmarks_mp.landmark[263].x * image_w

            left_v_dist = abs(left_eye_top_y - left_eye_bottom_y)
            left_h_dist = abs(left_eye_left_x - left_eye_right_x)
            right_v_dist = abs(right_eye_top_y - right_eye_bottom_y)
            right_h_dist = abs(right_eye_left_x - right_eye_right_x)
            
            # Avoid division by zero if eye is closed or not detected well
            left_ear = left_v_dist / (left_h_dist + 1e-6)
            right_ear = right_v_dist / (right_h_dist + 1e-6)
            
            avg_ear = (left_ear + right_ear) / 2.0 # Eye Aspect Ratio
            
            # Blink Detection using EAR
            EAR_THRESHOLD = 0.20 # Typical EAR threshold for blinks
            current_time = time.time()
            if avg_ear < EAR_THRESHOLD:
                if current_time - tracker['last_blink_time'] > 0.25: # Debounce blinks
                    tracker['blink_count'] += 1
                    tracker['last_blink_time'] = current_time
            
            # Update Blink Rate
            duration_seen = current_time - tracker['first_seen_time']
            if duration_seen > self.min_duration_for_rate_calc: # Calculate rate only after sufficient observation
                tracker['blink_rate'] = (tracker['blink_count'] / duration_seen) * 60
            
            # Eye Contact Proxy (Higher EAR generally means more open eyes)
            # Normalize EAR to a 0-1 range for contact_ratio (approximate)
            # Normal open EAR is ~0.25-0.35. Max can be ~0.4.
            if avg_ear > EAR_THRESHOLD + 0.05 : # Clearly open
                 tracker['eye_contact_ratio'] = min(1.0, (avg_ear - EAR_THRESHOLD) / (0.35 - EAR_THRESHOLD)) # Rough normalization
            else: # Closed or squinting
                 tracker['eye_contact_ratio'] = max(0.0, avg_ear / EAR_THRESHOLD * 0.2) # Low if near/below threshold

        except IndexError:
            # print("Error accessing landmark indices for eye contact/blink.")
            pass # Keep previous values or default


    def process_frame(self, image):
        """處理單幀影像"""
        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_mp = self.face_mesh.process(rgb_image)
        
        current_time = time.time()
        expired_faces = [face_id for face_id, tracker in self.face_trackers.items() 
                        if current_time - tracker['last_seen'] > 10.0] # Increased expiry time
        for face_id in expired_faces:
            # print(f"Removing expired face: {face_id}")
            del self.face_trackers[face_id]
        
        if results_mp.multi_face_landmarks:
            for face_landmarks_mp in results_mp.multi_face_landmarks:
                # 1. Derive BBox from MediaPipe landmarks
                x_coords = [lm.x * w for lm in face_landmarks_mp.landmark]
                y_coords = [lm.y * h for lm in face_landmarks_mp.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add padding to bbox for face encoding
                padding_w = (x_max - x_min) * 0.15
                padding_h = (y_max - y_min) * 0.15
                crop_x_min = max(0, int(x_min - padding_w))
                crop_y_min = max(0, int(y_min - padding_h))
                crop_x_max = min(w, int(x_max + padding_w))
                crop_y_max = min(h, int(y_max + padding_h))

                mp_bbox_for_id = (x_min, y_min, x_max, y_max) # Use precise bbox for ID association logic
                
                # 2. Crop face for encoding
                if crop_x_max > crop_x_min and crop_y_max > crop_y_min :
                    face_crop_bgr = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                    face_encoding = self.get_face_encoding(face_crop_bgr)
                else:
                    face_encoding = None

                # 3. Find or Create Face ID
                face_id = self.find_or_create_face_id(face_encoding, mp_bbox_for_id)
                
                if face_id is None:
                    # Could draw a generic box for unidentified faces if desired
                    # cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (128,128,128), 1)
                    continue # Skip if no ID could be assigned
                
                tracker = self.face_trackers[face_id]
                tracker['last_seen'] = current_time
                tracker['bbox'] = mp_bbox_for_id # Update bbox with the one from MediaPipe
                
                features = self.extract_facial_features(face_landmarks_mp, image.shape)
                micro_expressions = self.detect_micro_expressions(features, image.shape[0], face_id)
                
                if not tracker['baseline_established']:
                    self.establish_baseline(micro_expressions, face_id)
                
                self.detect_eye_contact_and_blinks(face_landmarks_mp, face_id, h, w)
                
                deception_score, indicators = self.detect_deception_indicators(micro_expressions, face_id)
                confidence_score = self.calculate_confidence_score(deception_score, face_id)
                
                self.draw_results(image, face_id, confidence_score, indicators)
                
                # Optional: Draw MediaPipe mesh
                # self.mp_drawing.draw_landmarks(
                #     image, face_landmarks_mp, self.mp_face_mesh.FACEMESH_TESSELATION,
                #     None, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
        
        return image

    def draw_results(self, image, face_id, confidence_score, indicators):
        """繪製分析結果"""
        if face_id not in self.face_trackers: return # Face might have expired
        tracker = self.face_trackers[face_id]
        
        # Use the bounding box stored in the tracker
        # It might be slightly different from a freshly computed one if processing is slow
        # but it's the one associated with this face_id's state
        x_min, y_min, x_max, y_max = tracker['bbox']
        bbox_tuple_int = (int(x_min), int(y_min), int(x_max), int(y_max))
        
        confidence_level, color = self.get_confidence_level(confidence_score)
        
        cv2.rectangle(image, (bbox_tuple_int[0], bbox_tuple_int[1]), 
                     (bbox_tuple_int[2], bbox_tuple_int[3]), color, 2)
        
        y_offset = bbox_tuple_int[1] - 7 # Adjusted for font size
        
        text = f"{face_id}: {confidence_level} ({confidence_score:.1f})"
        cv2.putText(image, text, (bbox_tuple_int[0], y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) # Slightly smaller font
        y_offset -= 20
        
        if not tracker['baseline_established']:
            progress = tracker['baseline_frame_count'] / self.baseline_frames * 100
            baseline_text = f"Baseline: {progress:.0f}%"
            cv2.putText(image, baseline_text, (bbox_tuple_int[0], y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset -= 15
        
        for i, indicator in enumerate(indicators[:2]): # Show max 2 indicators
            cv2.putText(image, indicator, (bbox_tuple_int[0], y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1) # Orange for indicators
            y_offset -= 12

        # Display Blinks and Rigidity below the box or to the side
        stats_y_start = bbox_tuple_int[3] + 15
        stats_text_blink = f"Blinks: {tracker['blink_rate']:.1f}/min"
        cv2.putText(image, stats_text_blink, (bbox_tuple_int[0], stats_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        
        if tracker['baseline_established']: # Only show rigidity if baseline is done
            stats_text_rigidity = f"Rigidity: {tracker['rigidity_score']:.2f}"
            cv2.putText(image, stats_text_rigidity, (bbox_tuple_int[0], stats_y_start + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)


def main():
    detector = ConfidenceDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 30) # Setting FPS might not always be respected

    print("Micro-expression Confidence Analysis System - Modified Version")
    print("Press 'q' to quit.")
    print("Establishing baseline for new faces...")

    frame_count = 0
    processing_time_sum = 0
    display_fps_interval = 30 # frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
        
        frame = cv2.flip(frame, 1)
        
        iter_start_time = time.time()
        processed_frame = detector.process_frame(frame.copy()) # Process a copy
        iter_end_time = time.time()

        processing_time_sum += (iter_end_time - iter_start_time)
        frame_count += 1

        if frame_count % display_fps_interval == 0:
            avg_processing_time = processing_time_sum / display_fps_interval
            current_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            print(f"FPS: {current_fps:.2f} (Avg processing time: {avg_processing_time*1000:.2f} ms)")
            processing_time_sum = 0 # Reset for next interval
        
        cv2.putText(processed_frame, "Micro-expression Confidence Analysis", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        cv2.putText(processed_frame, f"Tracking: {len(detector.face_trackers)} face(s)", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.putText(processed_frame, "Press 'q' to quit", 
                   (10, processed_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        
        cv2.imshow('Confidence Detection System', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("System closed.")

if __name__ == "__main__":
    try:
        # Re-check imports just in case for the main guard
        import cv2
        import numpy as np
        import face_recognition
        import mediapipe as mp
        print("All core dependencies seem to be available.")
        main()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required dependencies, e.g.:")
        print("pip install opencv-python numpy face-recognition mediapipe")
