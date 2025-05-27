import cv2
import numpy as np
import face_recognition
import time
from collections import deque, defaultdict
import mediapipe as mp
# from scipy.spatial.distance import euclidean # 已移除：未使用
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
        self.micro_expression_history_length = 15 # 用於計算僵硬程度
        self.min_duration_for_rate_calc = 10 # 秒數，用於計算眨眼率

        # 關鍵面部標記點（基於MediaPipe 468點模型） - 已移除：此特定字典未直接使用，因為動作單元有自己的列表
        # self.facial_landmarks = { ... }
        
        # 動作單位(AU)對應的標記點 - 更新了AU1, AU2的標記點
        self.action_units = {
            'AU1': {'landmarks': [63, 105, 66, 293, 334, 296], 'name': '眉毛內側上抬'}, # 內側眉毛
            'AU2': {'landmarks': [70, 52, 53, 300, 282, 283], 'name': '眉毛外側上抬'}, # 外側眉毛（70/300是拱形點）
            'AU5': {'landmarks': [159, 145, 386, 374], 'name': '上眼瞼抬高'}, # 使用上下眼瞼點來測量眼睛開合度
            'AU6': {'landmarks': [117, 118, 119, 346, 347, 348, 205, 206], 'name': '臉頬上提'}, # 臉頰標記點示例
            'AU9': {'landmarks': [61, 291, 40, 39, 37, 0, 267, 269, 270, 80, 81, 82], 'name': '鼻翼皺起/上唇抬起'}, # 簡化合併
            'AU10': {'landmarks': [40, 39, 37, 0, 267, 269, 270, 13, 14], 'name': '上唇中部抬起'}, # 上唇中心
            'AU12': {'landmarks': [61, 291], 'name': '嘴角上拉'}, # 嘴角
            'AU14': {'landmarks': [61, 291], 'name': '酒窩/單側嘴角'}, # 用於檢查不對稱性的嘴角
            'AU15': {'landmarks': [61, 291], 'name': '嘴角下拉'}, # 嘴角
            'AU20': {'landmarks': [61, 291], 'name': '嘴角水平拉伸'} # 用於測量寬度的嘴角
        }

    def get_face_encoding(self, face_image):
        """獲取人臉編碼用於識別"""
        try:
            # 確保face_image是BGR格式
            if face_image.shape[2] == 3: # 如果有3個通道
                 rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else: # 灰階，通過複製通道轉換為RGB
                 rgb_face = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            encodings = face_recognition.face_encodings(rgb_face)
            return encodings[0] if encodings else None
        except Exception as e:
            # print(f"Error in get_face_encoding: {e}")
            return None

    def find_or_create_face_id(self, face_encoding, face_bbox):
        """找到現有人臉ID或創建新ID"""
        if face_encoding is None:
             # 如果編碼失敗但追蹤仍在進行，嘗試通過邊界框重疊來尋找
            for face_id, tracker_data in self.face_trackers.items():
                # 這裡可以使用簡單的IoU或質心距離
                # 目前，假設如果編碼失敗，我們無法可靠地為新面孔識別
                pass # 可以添加基於邊界框接近度重新關聯的邏輯
            return None # 如果編碼失敗，不要輕易創建新ID以避免過多錯誤ID
            
        # 與已知人臉比較
        for face_id, tracker in self.face_trackers.items():
            if tracker['encoding'] is not None:
                # 檢查tracker['encoding']是否為有效的編碼（列表或ndarray）
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
                        continue # 如果比較失敗則跳過
                
        
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
            'first_seen_time': current_time, # 用於眨眼率
            'bbox': face_bbox,
            'micro_expression_count': defaultdict(int),
            'micro_expression_history': deque(maxlen=self.micro_expression_history_length), # 用於計算僵硬程度
            'rigidity_score': 0.0, # 初始化僵硬程度分數
            'eye_contact_ratio': 0.5, # 初始中性值
            'blink_rate': 0,
            'last_blink_time': current_time, # 初始化最後眨眼時間
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

    def detect_micro_expressions(self, features, image_h, face_id): # 添加了image_h參數
        """偵測微表情和動作單位。較高的值表示表情強度更大。"""
        tracker = self.face_trackers[face_id]
        micro_expressions = {}
        
        # 對於y座標，較小的y表示在螢幕上較高
        # 我們希望較大的值表示更多的表情。所以，對於"向上"的動作，使用image_h - y或-y
        
        # AU1: 眉毛內側上抬（恐懼指標）
        if 'AU1' in features and features['AU1'].size > 0:
            eyebrow_inner_height = np.mean(image_h - features['AU1'][:, 1]) 
            micro_expressions['fear_eyebrow_inner'] = eyebrow_inner_height
        
        # AU2: 眉毛外側上抬（恐懼指標）
        if 'AU2' in features and features['AU2'].size > 0:
            eyebrow_outer_height = np.mean(image_h - features['AU2'][:, 1])
            micro_expressions['fear_eyebrow_outer'] = eyebrow_outer_height

        # AU5: 上眼瞼抬高（恐懼指標）- 使用特定的眼睛開合度標記點
        if 'AU5' in features and features['AU5'].size >= 4: # 確保我們有兩個眼睛的點
            # 標記點：[159(左上), 145(左下), 386(右上), 374(右下)]
            left_eye_openness = abs(features['AU5'][0, 1] - features['AU5'][1, 1]) # y_top_L - y_bottom_L
            right_eye_openness = abs(features['AU5'][2, 1] - features['AU5'][3, 1]) # y_top_R - y_bottom_R
            avg_eye_openness = (left_eye_openness + right_eye_openness) / 2
            micro_expressions['fear_eyelid'] = avg_eye_openness # 較高的值表示眼睛更開

        # AU9 & AU10: 上唇抬起（厭惡指標）
        if 'AU10' in features and features['AU10'].size > 0: # 使用AU10作為上唇中心
            upper_lip_raise = np.mean(image_h - features['AU10'][:, 1])
            micro_expressions['disgust_lip'] = upper_lip_raise
        
        # AU12: 嘴角上拉（快樂指標，可能是"欺騙的快樂"）- 檢查嘴角的不對稱性
        if 'AU12' in features and features['AU12'].size >= 2:
            mouth_corners_y = image_h - features['AU12'][:, 1] # 較高的值表示更多向上拉
            # 嘴角上拉的不對稱性（相對於中性或僅y差異）
            # 對於欺騙性的快樂，通常是不對稱的微笑
            # 我們也可以測量向上拉的本身
            # 讓我們使用嘴角Y位置的不對稱性
            if len(mouth_corners_y) >= 2:
                 micro_expressions['happiness_asymmetry'] = abs(mouth_corners_y[0] - mouth_corners_y[1])
                 micro_expressions['smile_intensity'] = np.mean(mouth_corners_y) # 平均向上拉

        # AU14: 單側嘴角緊繃（輕蔑指標）- 不對稱性
        if 'AU14' in features and features['AU14'].size >= 2:
            # 使用嘴角的y座標[61, 291]
            left_corner_y = features['AU14'][0, 1]
            right_corner_y = features['AU14'][1, 1]
            micro_expressions['contempt_asymmetry'] = abs(left_corner_y - right_corner_y)
        
        # AU15: 嘴角下拉（悲傷指標）
        if 'AU15' in features and features['AU15'].size >=2:
            # features['AU15']是嘴角。較高的y表示更多下垂
            mouth_droop = np.mean(features['AU15'][:, 1]) 
            micro_expressions['sadness_droop'] = mouth_droop
        
        # AU20: 嘴角水平拉伸（恐懼指標）
        if 'AU20' in features and features['AU20'].size >= 2:
            # features['AU20']是嘴角[61, 291]
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
                    if not values: continue # 如果沒有記錄值則跳過
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    # 對於"抬起"或"增加"的表情，閾值較高
                    # 對於"下垂"或其他類型，如果需要可以相應調整，但保持"較高的值=更多的表情"的一致性有幫助
                    threshold_val = mean_val + 2 * std_val 
                    # 對於較低值表示更多的表情（例如，如果不使用image_h - y），它將是mean - 2 * std
                    
                    tracker['baseline_emotions'][expr_name] = {
                        'mean': mean_val,
                        'std': std_val,
                        'threshold': threshold_val,
                        'raw_values': values # 如果需要其他分析，保留原始值
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
        
        # 恐懼指標：眉毛內/外側抬起，眼瞼抬起，嘴巴拉伸
        fear_indicators_config = {
            'fear_eyebrow_inner': {'weight': 4, 'name': "內側眉毛抬起（恐懼）"},
            'fear_eyebrow_outer': {'weight': 4, 'name': "外側眉毛抬起（恐懼）"},
            'fear_eyelid': {'weight': 8, 'name': "眼瞼抬起（恐懼）"}, # 通常更明顯
            'fear_mouth_stretch': {'weight': 6, 'name': "嘴巴拉伸（恐懼）"}
        }

        for indicator, config in fear_indicators_config.items():
            if indicator in micro_expressions and indicator in tracker['baseline_emotions']:
                current_value = micro_expressions[indicator]
                baseline = tracker['baseline_emotions'][indicator]
                if current_value > baseline['threshold']: # 較高的值表示更多的表情強度
                    deception_score += config['weight']
                    indicators.append(config['name'])
                    tracker['micro_expression_count'][indicator] += 1
        
        # 厭惡微表情
        if 'disgust_lip' in micro_expressions and 'disgust_lip' in tracker['baseline_emotions']:
            current_value = micro_expressions['disgust_lip']
            baseline = tracker['baseline_emotions']['disgust_lip']
            if current_value > baseline['threshold']:
                deception_score += 6
                indicators.append("厭惡微表情（嘴唇）")
                tracker['micro_expression_count']['disgust'] += 1
        
        # 輕蔑微表情（不對稱性）
        if 'contempt_asymmetry' in micro_expressions and 'contempt_asymmetry' in tracker['baseline_emotions']:
            current_value = micro_expressions['contempt_asymmetry']
            baseline = tracker['baseline_emotions']['contempt_asymmetry']
            if current_value > baseline['threshold']: # 較高的不對稱性
                deception_score += 7
                indicators.append("輕蔑微表情（不對稱性）")
                tracker['micro_expression_count']['contempt'] += 1
        
        # 悲傷微表情
        if 'sadness_droop' in micro_expressions and 'sadness_droop' in tracker['baseline_emotions']:
            current_value = micro_expressions['sadness_droop']
            baseline = tracker['baseline_emotions']['sadness_droop']
            if current_value > baseline['threshold']: # 更多下垂
                deception_score += 5
                indicators.append("悲傷微表情（下垂）")
                tracker['micro_expression_count']['sadness'] += 1
        
        # 欺騙性快樂（不對稱微笑）
        if 'happiness_asymmetry' in micro_expressions and 'happiness_asymmetry' in tracker['baseline_emotions']:
            # 同時檢查smile_intensity是否存在且高於某個水平以符合微笑
            smile_intensity_check = True
            if 'smile_intensity' in micro_expressions and 'smile_intensity' in tracker['baseline_emotions']:
                 smile_intensity_check = micro_expressions['smile_intensity'] > tracker['baseline_emotions']['smile_intensity']['mean'] # 基本檢查

            if smile_intensity_check:
                current_value = micro_expressions['happiness_asymmetry']
                baseline = tracker['baseline_emotions']['happiness_asymmetry']
                if current_value > baseline['threshold']: # 更多不對稱性
                    deception_score += 12
                    indicators.append("欺騙性快樂（不對稱微笑）")
                    tracker['micro_expression_count']['fake_happiness'] += 1
        
        return deception_score, indicators

    def calculate_rigidity_score(self, face_id):
        tracker = self.face_trackers[face_id]
        if not tracker['baseline_established'] or len(tracker['micro_expression_history']) < self.micro_expression_history_length // 2:
            return 0.0 # 數據不足或基準線未準備好

        rigidity_level = 0
        num_rigidity_metrics_checked = 0

        # 表示動作的指標。這些指標的低變異性可能意味著僵硬
        # 我們需要它們的基準標準差來比較
        rigidity_check_metrics = ['fear_eyebrow_inner', 'fear_eyebrow_outer', 'disgust_lip', 'smile_intensity']
        
        for metric_name in rigidity_check_metrics:
            if metric_name in tracker['baseline_emotions']:
                history_values = [hist[metric_name] for hist in tracker['micro_expression_history'] if metric_name in hist]
                if len(history_values) < self.micro_expression_history_length // 3:
                    continue

                current_std = np.std(history_values)
                baseline_std = tracker['baseline_emotions'][metric_name]['std']
                num_rigidity_metrics_checked +=1

                # 如果當前動作（標準差）顯著小於基準動作，或絕對值非常低
                # 在baseline_std中添加一個小的epsilon以避免除以零或極端比率
                if current_std < 0.3 * (baseline_std + 1e-6) or current_std < 0.05 * np.abs(tracker['baseline_emotions'][metric_name]['mean'] + 1e-6): 
                    rigidity_level += 1
        
        if num_rigidity_metrics_checked == 0:
            return 0.0
            
        # 標準化僵硬程度分數（0到1，其中1表示高度僵硬）
        tracker['rigidity_score'] = rigidity_level / num_rigidity_metrics_checked 
        return tracker['rigidity_score']


    def calculate_confidence_score(self, deception_score, face_id):
        """計算信心分數"""
        tracker = self.face_trackers[face_id]
        
        # 在使用之前計算僵硬程度
        current_rigidity = self.calculate_rigidity_score(face_id) # 這會更新tracker['rigidity_score']

        confidence_score = 50 # 從中性基準開始
        confidence_score -= deception_score
        
        if tracker['rigidity_score'] > 0.6: # 如果超過60%的檢查指標顯示僵硬
            confidence_score -= 10 # 增加對僵硬的懲罰
        elif tracker['rigidity_score'] > 0.3:
            confidence_score -=5

        if tracker['eye_contact_ratio'] < 0.3:
            confidence_score -= 6
        elif tracker['eye_contact_ratio'] > 0.9: # 持續的，可能不自然的注視
            confidence_score -= 4
        else: # 良好的眼神接觸
            confidence_score += 3
        
        # 眨眼率評估（每分鐘）
        if tracker['blink_rate'] > 35: # 高眨眼率
            confidence_score -= 7
        elif tracker['blink_rate'] < 8 and tracker['baseline_established']: # 低眨眼率（如果基準線已建立以避免過早懲罰）
             confidence_score -= 4
        
        confidence_score = max(0, min(100, confidence_score))
        
        tracker['emotion_history'].append(confidence_score)
        # 平滑信心分數
        if len(tracker['emotion_history']) > 0:
             # 使用加權平均，給予最近的分數更多重要性
            weights = np.exp(np.linspace(-1., 0., len(tracker['emotion_history'])))
            weights /= weights.sum()
            weighted_avg_score = np.dot(list(tracker['emotion_history']), weights)
            tracker['confidence_score'] = weighted_avg_score
        else:
            tracker['confidence_score'] = confidence_score # 如果emotion_history先附加，這不應該發生
            
        return tracker['confidence_score']

    def get_confidence_level(self, score):
        """將信心分數轉換為等級"""
        if score >= 75: # 稍微調整了閾值
            return "非常自信", (0, 255, 0)
        elif score >= 60:
            return "自信", (0, 200, 100)
        elif score >= 40:
            return "中性", (0, 255, 255)
        elif score >= 25:
            return "不自信", (0, 165, 255)
        else:
            return "非常不自信", (0, 0, 255)

    def detect_eye_contact_and_blinks(self, landmarks_mp, face_id, image_h, image_w):
        tracker = self.face_trackers[face_id]
        
        # 眼睛標記點（MediaPipe）：
        # 左眼：33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        # 右眼：362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        # 用於眼睛開合度：
        # 左眼：159（頂部）和145（底部）之間的垂直距離
        # 右眼：386（頂部）和374（底部）之間的垂直距離
        # 用於標準化的水平距離：
        # 左眼：133（角落）和33（角落）
        # 右眼：362（角落）和263（角落）

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
            
            # 如果眼睛閉上或檢測不好，避免除以零
            left_ear = left_v_dist / (left_h_dist + 1e-6)
            right_ear = right_v_dist / (right_h_dist + 1e-6)
            
            avg_ear = (left_ear + right_ear) / 2.0 # 眼睛縱橫比
            
            # 使用EAR進行眨眼檢測
            EAR_THRESHOLD = 0.20 # 眨眼的典型EAR閾值
            current_time = time.time()
            if avg_ear < EAR_THRESHOLD:
                if current_time - tracker['last_blink_time'] > 0.25: # 消除眨眼抖動
                    tracker['blink_count'] += 1
                    tracker['last_blink_time'] = current_time
            
            # 更新眨眼率
            duration_seen = current_time - tracker['first_seen_time']
            if duration_seen > self.min_duration_for_rate_calc: # 只有在足夠的觀察後才計算率
                tracker['blink_rate'] = (tracker['blink_count'] / duration_seen) * 60
            
            # 眼神接觸代理（較高的EAR通常意味著眼睛更開）
            # 將EAR標準化為0-1範圍用於接觸率（近似）
            # 正常開眼EAR約為0.25-0.35。最大值約為0.4
            if avg_ear > EAR_THRESHOLD + 0.05 : # 明顯睜開
                 tracker['eye_contact_ratio'] = min(1.0, (avg_ear - EAR_THRESHOLD) / (0.35 - EAR_THRESHOLD)) # 粗略標準化
            else: # 閉眼或瞇眼
                 tracker['eye_contact_ratio'] = max(0.0, avg_ear / EAR_THRESHOLD * 0.2) # 如果接近/低於閾值則較低

        except IndexError:
            # print("Error accessing landmark indices for eye contact/blink.")
            pass # 保持先前的值或默認值


    def process_frame(self, image):
        """處理單幀影像"""
        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_mp = self.face_mesh.process(rgb_image)
        
        current_time = time.time()
        expired_faces = [face_id for face_id, tracker in self.face_trackers.items() 
                        if current_time - tracker['last_seen'] > 10.0] # 增加了過期時間
        for face_id in expired_faces:
            # print(f"Removing expired face: {face_id}")
            del self.face_trackers[face_id]
        
        if results_mp.multi_face_landmarks:
            for face_landmarks_mp in results_mp.multi_face_landmarks:
                # 1. 從MediaPipe標記點推導邊界框
                x_coords = [lm.x * w for lm in face_landmarks_mp.landmark]
                y_coords = [lm.y * h for lm in face_landmarks_mp.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # 為人臉編碼添加邊界框填充
                padding_w = (x_max - x_min) * 0.15
                padding_h = (y_max - y_min) * 0.15
                crop_x_min = max(0, int(x_min - padding_w))
                crop_y_min = max(0, int(y_min - padding_h))
                crop_x_max = min(w, int(x_max + padding_w))
                crop_y_max = min(h, int(y_max + padding_h))

                mp_bbox_for_id = (x_min, y_min, x_max, y_max) # 使用精確的邊界框進行ID關聯邏輯
                
                # 2. 裁剪人臉用於編碼
                if crop_x_max > crop_x_min and crop_y_max > crop_y_min :
                    face_crop_bgr = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                    face_encoding = self.get_face_encoding(face_crop_bgr)
                else:
                    face_encoding = None

                # 3. 尋找或創建人臉ID
                face_id = self.find_or_create_face_id(face_encoding, mp_bbox_for_id)
                
                if face_id is None:
                    # 如果需要，可以為未識別的人臉繪製通用框
                    # cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (128,128,128), 1)
                    continue # 如果無法分配ID則跳過
                
                tracker = self.face_trackers[face_id]
                tracker['last_seen'] = current_time
                tracker['bbox'] = mp_bbox_for_id # 使用MediaPipe的邊界框更新
                
                features = self.extract_facial_features(face_landmarks_mp, image.shape)
                micro_expressions = self.detect_micro_expressions(features, image.shape[0], face_id)
                
                if not tracker['baseline_established']:
                    self.establish_baseline(micro_expressions, face_id)
                
                self.detect_eye_contact_and_blinks(face_landmarks_mp, face_id, h, w)
                
                deception_score, indicators = self.detect_deception_indicators(micro_expressions, face_id)
                confidence_score = self.calculate_confidence_score(deception_score, face_id)
                
                self.draw_results(image, face_id, confidence_score, indicators)
                
                # 可選：繪製MediaPipe網格
                # self.mp_drawing.draw_landmarks(
                #     image, face_landmarks_mp, self.mp_face_mesh.FACEMESH_TESSELATION,
                #     None, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
        
        return image

    def draw_results(self, image, face_id, confidence_score, indicators):
        """繪製分析結果"""
        if face_id not in self.face_trackers: return # 人臉可能已過期
        tracker = self.face_trackers[face_id]
        
        # 使用存儲在追蹤器中的邊界框
        # 如果處理速度慢，它可能與新計算的略有不同
        # 但它是與這個face_id的狀態相關聯的
        x_min, y_min, x_max, y_max = tracker['bbox']
        bbox_tuple_int = (int(x_min), int(y_min), int(x_max), int(y_max))
        
        confidence_level, color = self.get_confidence_level(confidence_score)
        
        cv2.rectangle(image, (bbox_tuple_int[0], bbox_tuple_int[1]), 
                     (bbox_tuple_int[2], bbox_tuple_int[3]), color, 2)
        
        y_offset = bbox_tuple_int[1] - 7 # 調整字體大小
        
        text = f"{face_id}: {confidence_level} ({confidence_score:.1f})"
        cv2.putText(image, text, (bbox_tuple_int[0], y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) # 稍微小一點的字體
        y_offset -= 20
        
        if not tracker['baseline_established']:
            progress = tracker['baseline_frame_count'] / self.baseline_frames * 100
            baseline_text = f"基準線：{progress:.0f}%"
            cv2.putText(image, baseline_text, (bbox_tuple_int[0], y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset -= 15
        
        for i, indicator in enumerate(indicators[:2]): # 最多顯示2個指標
            cv2.putText(image, indicator, (bbox_tuple_int[0], y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1) # 指標用橙色
            y_offset -= 12

        # 在框下方或側面顯示眨眼和僵硬程度
        stats_y_start = bbox_tuple_int[3] + 15
        stats_text_blink = f"眨眼：{tracker['blink_rate']:.1f}/分鐘"
        cv2.putText(image, stats_text_blink, (bbox_tuple_int[0], stats_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        
        if tracker['baseline_established']: # 只有在基準線完成後才顯示僵硬程度
            stats_text_rigidity = f"僵硬程度：{tracker['rigidity_score']:.2f}"
            cv2.putText(image, stats_text_rigidity, (bbox_tuple_int[0], stats_y_start + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)


def main():
    detector = ConfidenceDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("錯誤：無法開啟網路攝影機。")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 30) # 設置FPS可能不會總是被尊重

    print("微表情信心分析系統 - 修改版")
    print("按'q'退出。")
    print("正在為新面孔建立基準線...")

    frame_count = 0
    processing_time_sum = 0
    display_fps_interval = 30 # 幀數

    while True:
        ret, frame = cap.read()
        if not ret:
            print("錯誤：無法接收幀（串流結束？）。正在退出...")
            break
        
        frame = cv2.flip(frame, 1)
        
        iter_start_time = time.time()
        processed_frame = detector.process_frame(frame.copy()) # 處理副本
        iter_end_time = time.time()

        processing_time_sum += (iter_end_time - iter_start_time)
        frame_count += 1

        if frame_count % display_fps_interval == 0:
            avg_processing_time = processing_time_sum / display_fps_interval
            current_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            print(f"FPS：{current_fps:.2f}（平均處理時間：{avg_processing_time*1000:.2f}毫秒）")
            processing_time_sum = 0 # 重置下一個間隔
        
        cv2.putText(processed_frame, "微表情信心分析", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        cv2.putText(processed_frame, f"追蹤：{len(detector.face_trackers)}張臉", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.putText(processed_frame, "按'q'退出", 
                   (10, processed_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        
        cv2.imshow('信心偵測系統', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("系統已關閉。")

if __name__ == "__main__":
    try:
        # 在主守衛中再次檢查導入
        import cv2
        import numpy as np
        import face_recognition
        import mediapipe as mp
        print("所有核心依賴似乎都可用。")
        main()
    except ImportError as e:
        print(f"缺少依賴：{e}")
        print("請安裝所需的依賴，例如：")
        print("pip install opencv-python numpy face-recognition mediapipe")
