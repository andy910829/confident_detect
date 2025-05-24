import cv2
import numpy as np
import face_recognition
import time
from collections import deque

class ConfidenceAnalyzer:
    """
    通過攝影機分析使用者說話時的信心程度
    分析依據:
    1. 眼神接觸的穩定性
    2. 面部表情的穩定性
    3. 頭部姿勢的穩定性
    4. 微表情分析
    """
    
    def __init__(self):
        # 初始化攝影機
        self.cap = None
        # 信心指標分數
        self.confidence_score = 50  # 起始中等信心水平
        # 保存歷史分數用於平滑處理
        self.confidence_history = deque(maxlen=30)  # 約1秒的歷史(假設30FPS)
        # 眼神接觸計時器
        self.eye_contact_time = 0
        self.last_eye_contact = False
        # 微表情檢測相關
        self.facial_landmarks_history = []
        self.head_movement_history = []
        # 信心等級描述
        self.confidence_levels = {
            (0, 20): "Very Low Confidence - Likely feeling nervous or uncertain",
            (20, 40): "Low Confidence - Some hesitation",
            (40, 60): "Medium Confidence - Basically confident",
            (60, 80): "High Confidence - Quite confident",
            (80, 101): "Very High Confidence - Extremely confident"
        }
        
    def start(self):
        """開始攝影機分析"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("無法開啟攝影機")
            return
        
        print("正在分析使用者的信心程度...")
        print("按 'q' 鍵退出")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("無法接收影像")
                break
                
            # 鏡像翻轉（更自然）
            frame = cv2.flip(frame, 1)
            
            # 分析這一幀
            self.analyze_frame(frame)
            
            # 顯示信心評分
            self._display_confidence_score(frame)
            
            # 顯示影像
            cv2.imshow('信心程度分析', frame)
            
            # 按q退出
            if cv2.waitKey(1) == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        
    def analyze_frame(self, frame):
        """分析單一影像幀"""
        # 縮小影格以加快處理速度
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # 檢測臉部位置
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        if not face_locations:
            # 沒有檢測到臉部，降低信心分數
            self._update_confidence_score(-1)
            return
            
        # 檢測臉部特徵點
        face_landmarks = face_recognition.face_landmarks(rgb_small_frame, face_locations)
        
        if face_landmarks:
            # 保存臉部特徵點歷史以分析穩定性
            self.facial_landmarks_history.append(face_landmarks[0])
            if len(self.facial_landmarks_history) > 10:
                self.facial_landmarks_history.pop(0)
            
            # 分析眼神接觸
            eye_contact = self._analyze_eye_contact(face_landmarks[0], rgb_small_frame)
            
            # 分析頭部姿勢穩定性
            head_stability = self._analyze_head_stability(face_locations[0])
            
            # 分析面部微表情
            micro_expression = self._analyze_micro_expressions()
            
            # 更新信心分數
            self._update_confidence_score(
                eye_contact * 0.4 +  # 眼神接觸佔40%的權重
                head_stability * 0.4 +  # 頭部穩定性佔40%的權重
                micro_expression * 0.2   # 微表情佔20%的權重
            )
            
    def _analyze_eye_contact(self, landmarks, face_image):
        """分析眼神接觸的穩定性"""
        # 簡單的眼睛開合檢測
        try:
            # 獲取左眼和右眼的點
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            
            # 計算左眼和右眼的高度和寬度比例
            left_eye_ratio = self._eye_aspect_ratio(left_eye)
            right_eye_ratio = self._eye_aspect_ratio(right_eye)
            
            # 取平均
            eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
            
            # 眼睛張開程度判斷（閾值可根據需要調整）
            eye_contact_now = eye_ratio > 0.2
            
            # 計算持續的眼神接觸時間
            if eye_contact_now:
                if self.last_eye_contact:
                    self.eye_contact_time += 1
                else:
                    self.eye_contact_time = 0
                    self.last_eye_contact = True
            else:
                self.last_eye_contact = False
                self.eye_contact_time = max(0, self.eye_contact_time - 2)  # 更快地降低
            
            # 長時間的眼神接觸表示更高的信心
            eye_contact_score = min(10, self.eye_contact_time / 30.0 * 10)  # 最高10分
            return eye_contact_score
            
        except (KeyError, IndexError):
            return 0  # 如果出現錯誤，返回0分
    
    def _eye_aspect_ratio(self, eye):
        """計算眼睛的縱橫比例"""
        # 假設eye是6個點的列表，代表眼睛輪廓
        if len(eye) < 6:
            return 0
            
        # 計算垂直距離
        v1 = self._distance(eye[1], eye[5])
        v2 = self._distance(eye[2], eye[4])
        
        # 計算水平距離
        h = self._distance(eye[0], eye[3])
        
        # 計算比例
        if h == 0:
            return 0
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _distance(self, p1, p2):
        """計算兩點之間的歐氏距離"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _analyze_head_stability(self, face_location):
        """分析頭部姿勢的穩定性"""
        # 添加到頭部運動歷史
        self.head_movement_history.append(face_location)
        if len(self.head_movement_history) > 15:
            self.head_movement_history.pop(0)
        
        # 如果歷史不夠長，給出中等分數
        if len(self.head_movement_history) < 5:
            return 5
        
        # 計算頭部移動的變化
        movement_scores = []
        for i in range(1, len(self.head_movement_history)):
            prev = self.head_movement_history[i-1]
            curr = self.head_movement_history[i]
            
            # 計算中心點的移動
            prev_center = ((prev[0] + prev[2]) // 2, (prev[1] + prev[3]) // 2)
            curr_center = ((curr[0] + curr[2]) // 2, (curr[1] + curr[3]) // 2)
            
            # 計算移動距離
            movement = np.sqrt((prev_center[0] - curr_center[0])**2 + 
                              (prev_center[1] - curr_center[1])**2)
            
            # 較小的移動表示更穩定的頭部姿勢
            stability = 10 - min(10, movement / 2)
            movement_scores.append(stability)
        
        # 返回平均穩定性分數
        return sum(movement_scores) / len(movement_scores)
    
    def _analyze_micro_expressions(self):
        """分析面部微表情"""
        # 需要更多的面部特徵歷史來做這個
        if len(self.facial_landmarks_history) < 5:
            return 5  # 默認中等分數
        
        # 分析嘴部和眉毛的穩定性
        try:
            # 計算最後幾幀嘴部形狀的變化
            mouth_stability = self._calculate_feature_stability('top_lip') * 0.5 + \
                             self._calculate_feature_stability('bottom_lip') * 0.5
            
            # 計算眉毛的穩定性
            eyebrow_stability = self._calculate_feature_stability('left_eyebrow') * 0.5 + \
                               self._calculate_feature_stability('right_eyebrow') * 0.5
            
            # 綜合分數 - 嘴部佔60%，眉毛佔40%
            # 較高的穩定性通常表示更高的信心
            micro_expression_score = mouth_stability * 0.6 + eyebrow_stability * 0.4
            
            return micro_expression_score
            
        except (KeyError, IndexError):
            return 5  # 如果出現錯誤，返回5分
    
    def _calculate_feature_stability(self, feature_name):
        """計算某個面部特徵的穩定性"""
        feature_points = []
        
        # 收集最後幾幀的特徵點
        for landmarks in self.facial_landmarks_history[-5:]:
            if feature_name in landmarks:
                feature_points.append(landmarks[feature_name])
        
        if len(feature_points) < 2:
            return 5  # 數據不足，返回中等分數
        
        # 計算連續幀之間特徵的平均變化
        changes = []
        for i in range(1, len(feature_points)):
            change = 0
            for j in range(len(feature_points[i])):
                # 計算對應點的移動距離
                point_change = self._distance(
                    feature_points[i][j], 
                    feature_points[i-1][j]
                )
                change += point_change
            
            # 計算平均變化
            avg_change = change / len(feature_points[i])
            changes.append(avg_change)
        
        # 計算穩定性分數（變化越小越穩定）
        avg_change = sum(changes) / len(changes)
        stability_score = 10 - min(10, avg_change * 2)
        
        return stability_score
    
    def _update_confidence_score(self, delta):
        """更新信心分數"""
        # 更新當前分數
        self.confidence_score += delta
        
        # 限制在0-100範圍內
        self.confidence_score = max(0, min(100, self.confidence_score))
        
        # 添加到歷史記錄以進行平滑處理
        self.confidence_history.append(self.confidence_score)
        
        # 計算平滑後的分數
        self.confidence_score = sum(self.confidence_history) / len(self.confidence_history)
    
    def _display_confidence_score(self, frame):
        """在影像上顯示信心分數和描述"""
        # 獲取畫面尺寸
        height, width = frame.shape[:2]
        
        # 底部區域背景
        cv2.rectangle(frame, (0, height - 120), (width, height), (0, 0, 0), -1)
        
        # 顯示信心分數條
        bar_width = int(width * 0.8)
        bar_height = 30
        bar_x = int(width * 0.05) 
        bar_y = height - 50
        
        # 背景條
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # 信心指示條
        confidence_width = int(bar_width * (self.confidence_score / 100.0))
        
        # 根據信心程度選擇顏色（紅色->黃色->綠色）
        if self.confidence_score < 40:
            color = (0, 0, 255)  # 紅色 - 低信心
        elif self.confidence_score < 60:
            color = (0, 255, 255)  # 黃色 - 中等信心
        else:
            color = (0, 255, 0)  # 綠色 - 高信心
            
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
        
        # 添加百分比文字
        cv2.putText(frame, f"{int(self.confidence_score)}%", 
                   (bar_x + bar_width + 5, bar_y + bar_height - 5),  
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 顯示信心等級描述
        confidence_text = "未知"
        for (low, high), description in self.confidence_levels.items():
            if low <= self.confidence_score < high:
                confidence_text = description
                break
                
        cv2.putText(frame, confidence_text, (bar_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

if __name__ == "__main__":
    analyzer = ConfidenceAnalyzer()
    analyzer.start()