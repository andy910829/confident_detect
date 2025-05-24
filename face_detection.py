import cv2
import numpy as np
import time
import argparse
import os
import pyaudio
import wave
import threading
import queue
import speech_recognition as sr
from scipy.io import wavfile
import librosa
# import librosa.display # Not used for display in this script
# import matplotlib.pyplot as plt # Not used for display in this script
from transformers import pipeline

# 全局變數用於語音分析結果
speech_confidence = 0.0
speech_text = ""
is_speaking = False
emotion_score = 0.0
emotion_label = ""

class AudioProcessor:
    """處理音訊並分析語音信心度"""
    
    def __init__(self, rate=16000, chunk=1024, record_seconds=3):
        self.rate = rate  # 採樣率
        self.chunk = chunk  # 每次讀取的音訊塊大小
        self.record_seconds = record_seconds  # 每次處理的音訊長度（秒）
        self.audio_queue = queue.Queue()  # 音訊數據隊列
        self.recognizer = sr.Recognizer()  # 語音識別器
        # 嘗試初始化情感分析模型
        try:
            self.emotion_analyzer = pipeline("text-classification", 
                                            model="j-hartmann/emotion-english-distilroberta-base", 
                                            top_k=1)
            self.emotion_model_loaded = True
            print("情感分析模型加載成功")
        except Exception as e:
            print(f"無法加載情感分析模型：{e}")
            self.emotion_model_loaded = False
            
        # 啟動音訊處理線程
        self.is_running = True
        self.audio_thread = threading.Thread(target=self._process_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def start_recording(self):
        """開始錄音"""
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                     channels=1,
                                     rate=self.rate,
                                     input=True,
                                     frames_per_buffer=self.chunk)
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        print("開始錄音...")
        
    def _record_audio(self):
        """錄製音訊並將其放入隊列"""
        while self.is_running:
            frames = []
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                if not self.is_running:
                    break
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"錄音錯誤: {e}")
                    break
            
            if frames and self.is_running:
                # 將音訊幀放入隊列以供處理
                self.audio_queue.put(frames)
    
    def _process_audio(self):
        """處理音訊數據並分析信心度"""
        global speech_confidence, speech_text, is_speaking, emotion_score, emotion_label
        
        while self.is_running:
            try:
                # 從隊列中獲取音訊
                frames = self.audio_queue.get(timeout=1)
                
                # 將音訊數據保存為WAV文件以便處理
                temp_filename = "temp_audio.wav"
                wf = wave.open(temp_filename, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16 bits = 2 bytes
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # 分析音量大小來判斷是否有人在說話
                try:
                    audio_data, _ = librosa.load(temp_filename, sr=self.rate)
                    rms = librosa.feature.rms(y=audio_data)[0]
                    avg_rms = np.mean(rms)
                    is_speaking = avg_rms > 0.001  # 閾值可能需要調整
                    # 計算音訊能量變化作為情感強度的一個指標
                    energy_variance = np.var(rms) * 100
                    base_confidence = min(1.0, energy_variance)
                except Exception as e:
                    print(f"音訊分析錯誤: {e}")
                    is_speaking = False
                    base_confidence = 0.0
                
                # 使用語音識別
                if is_speaking:
                    try:
                        with sr.AudioFile(temp_filename) as source:
                            audio = self.recognizer.record(source)
                            result = self.recognizer.recognize_google(audio, language='zh-TW', show_all=True) # Language can be changed if needed
                            
                            # Google Speech API返回的結果包含替代文本和信心度
                            if result and isinstance(result, dict) and 'alternative' in result:
                                best_result = result['alternative'][0]
                                if 'confidence' in best_result:
                                    # 直接使用Google提供的信心度
                                    api_confidence = float(best_result['confidence'])
                                    speech_confidence = api_confidence
                                else:
                                    # 如果Google沒有提供信心度，使用我們的估算值
                                    speech_confidence = base_confidence
                                
                                speech_text = best_result['transcript']
                                print(speech_text)
                                # 分析情感（如果模型已加載）
                                if self.emotion_model_loaded and speech_text:
                                    try:
                                        # Assuming emotion model works with English text primarily
                                        # If speech_text is Chinese, emotion analysis might be less accurate
                                        # or require a model trained for Chinese.
                                        # For simplicity, we pass it as is.
                                        emotion_result = self.emotion_analyzer(speech_text)
                                        emotion_label = emotion_result[0][0]['label']
                                        emotion_score = emotion_result[0][0]['score']
                                        
                                        # 依據情感調整信心度
                                        if emotion_label in ['joy', 'anger']:
                                            # 積極情感或激動情感通常表達更有信心
                                            speech_confidence = min(1.0, speech_confidence * 1.2)
                                        elif emotion_label in ['fear', 'sadness']:
                                            # 消極情感通常表達較少信心
                                            speech_confidence = max(0.0, speech_confidence * 0.8)
                                    except Exception as e:
                                        print(f"情感分析錯誤: {e}")
                            else:
                                # 沒有識別結果，可能是背景噪音
                                is_speaking = False
                                speech_confidence = 0.0
                                speech_text = ""
                    except sr.UnknownValueError:
                        # 語音不清晰
                        speech_text = ""
                        speech_confidence = base_confidence * 0.5  # 降低信心度
                    except Exception as e:
                        print(f"語音識別錯誤: {e}")
                        speech_text = ""
                        speech_confidence = 0.0
                else:
                    # 沒有檢測到說話
                    speech_text = ""
                    speech_confidence = 0.0
                
                # 刪除臨時文件
                try:
                    os.remove(temp_filename)
                except:
                    pass
                    
            except queue.Empty:
                # 隊列為空，繼續等待
                pass
            except Exception as e:
                print(f"音訊處理錯誤: {e}")
    
    def stop(self):
        """停止所有音訊處理"""
        self.is_running = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()


def download_face_models():
    """下載人臉檢測模型文件"""
    try:
        # 檢查YuNet模型是否存在
        if not os.path.exists("face_detection_yunet_2023mar.onnx"):
            print("正在下載YuNet人臉檢測模型...")
            yunet_model_url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            import urllib.request
            urllib.request.urlretrieve(yunet_model_url, "face_detection_yunet_2023mar.onnx")
            print("YuNet模型下載成功")
            return True
        else:
            print("YuNet模型已存在")
            return True
    except Exception as e:
        print(f"模型下載失敗: {e}")
        print("您可以手動下載模型或嘗試使用Haar分類器方法")
        return False


def detect_faces(image, confidence_threshold=0.5, model_type='dnn'):
    """
    使用OpenCV的深度學習模型檢測人臉並返回信心度
    
    參數:
        image: 輸入圖片（可以是路徑或圖片數據）
        confidence_threshold: 最低接受的信心度閾值
        model_type: 使用的模型類型 ('dnn' 或 'haar')
    
    返回:
        帶有人臉標記的圖片以及人臉位置和信心度的列表
    """
    # 檢查image是路徑還是數據
    if isinstance(image, str):
        # 讀取圖片
        image_data = cv2.imread(image) # Renamed to avoid conflict with outer scope 'image'
        if image_data is None:
            print(f"無法讀取圖片: {image}")
            return None, []
        img_to_process = image_data
    else:
        img_to_process = image # image is already image data

    # 獲取圖片尺寸
    (h, w) = img_to_process.shape[:2]
    
    try:
        if model_type == "dnn":
            # 使用 DNN 模型 (OpenCV內置模型)
            # print("使用DNN人臉檢測模型...") # Commented out for less console noise
            
            # 使用OpenCV的FaceDetectorYN
            face_detector = cv2.FaceDetectorYN.create(
                model="face_detection_yunet_2023mar.onnx",
                config="",
                input_size=(w, h)
            )
            
            # 檢測人臉
            _, faces_data = face_detector.detect(img_to_process)
            
            # 如果沒有檢測到人臉
            if faces_data is None:
                return img_to_process, []
            
            # 處理檢測結果
            faces = []
            for face_data in faces_data:
                # 獲取邊界框座標和信心度
                box = face_data[0:4].astype(int)
                confidence = face_data[14]  # YuNet模型中的信心度索引
                
                # 如果信心度高於閾值
                if confidence > confidence_threshold:
                    startX, startY, width_face, height_face = box # Renamed to avoid conflict
                    endX = startX + width_face
                    endY = startY + height_face
                    
                    # 確保邊界框在圖片範圍內
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    # 計算調整後的尺寸
                    face_width = endX - startX
                    face_height = endY - startY
                    
                    # 保存人臉信息
                    face_info = {
                        "box": (startX, startY, endX, endY),
                        "confidence": float(confidence),
                        "width": face_width,
                        "height": face_height
                    }
                    faces.append(face_info)
                    
        else:
            # 使用Haar級聯分類器 (不需要額外下載)
            # print("使用Haar級聯分類器人臉檢測...") # Commented out for less console noise
            
            # 載入OpenCV內置的人臉檢測器
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # 轉換為灰度圖
            gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
            
            # 檢測人臉
            # start_time_haar = time.time() # For specific timing if needed
            faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # end_time_haar = time.time()
            
            # print(f"人臉檢測耗時: {end_time_haar - start_time_haar:.2f}秒")
            
            # 處理檢測結果
            faces = []
            for (x_haar, y_haar, w_haar, h_haar) in faces_rect:
                confidence_haar = min(0.9, 0.5 + (w_haar * h_haar) / (w_haar * h_haar + 5000))  # 基於大小的簡單估算
                
                # 保存人臉信息
                face_info = {
                    "box": (x_haar, y_haar, x_haar + w_haar, y_haar + h_haar),
                    "confidence": float(confidence_haar),
                    "width": w_haar,
                    "height": h_haar
                }
                faces.append(face_info)
                
        return img_to_process, faces
    
    except Exception as e:
        print(f"檢測過程中發生錯誤: {e}")
        if model_type == "dnn" and "face_detection_yunet" in str(e):
            print("嘗試下載YuNet模型...")
            try:
                yunet_model_url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                import urllib.request
                urllib.request.urlretrieve(yunet_model_url, "face_detection_yunet_2023mar.onnx")
                print("模型下載成功，請重新運行程式")
            except Exception as download_error:
                print(f"模型下載失敗: {download_error}")
                print("請手動下載模型或嘗試使用Haar分類器方法")
        return img_to_process, []


def process_webcam_with_speech(model_type='dnn', confidence_threshold=0.5, camera_index=0):
    """
    使用網絡攝像頭實時檢測人臉並分析說話者的信心度
    
    參數:
        model_type: 使用的模型類型 ('dnn' 或 'haar')
        confidence_threshold: 最低接受的人臉檢測信心度閾值
        camera_index: 相機索引，通常默認相機為0
    """
    global speech_confidence, speech_text, is_speaking, emotion_score, emotion_label
    
    # 確保模型可用
    if model_type == 'dnn' and not download_face_models():
        print("DNN模型無法使用，切換到Haar分類器")
        model_type = 'haar'
    
    # 初始化視訊捕獲
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"無法開啟相機 (索引: {camera_index})")
        return
    
    # 獲取視頻流的寬度和高度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"相機解析度: {frame_width}x{frame_height}")
    
    # 創建一個不受干擾的窗口
    cv2.namedWindow('Real-time Speech Confidence Detection', cv2.WINDOW_NORMAL) # MODIFIED
    
    # 初始化音訊處理器
    try:
        audio_processor = AudioProcessor()
        audio_processor.start_recording()
        audio_enabled = True
        print("音訊處理器初始化成功")
    except Exception as e:
        print(f"無法初始化音訊處理器: {e}")
        print("將只進行人臉檢測，沒有語音信心度分析")
        audio_enabled = False
    
    # 初始化時間統計變量
    process_times = []
    last_fps_update = time.time()
    fps = 0
    
    # 用於處理相機輸入的循環
    try:
        while True:
            # 讀取一幀
            ret, frame = cap.read()
            if not ret:
                print("無法獲取視頻幀")
                break
            
            # 記錄開始處理時間
            start_time = time.time()
            
            # 檢測人臉
            result_frame, faces = detect_faces(frame, confidence_threshold, model_type)
            
            # 記錄結束處理時間
            end_time = time.time()
            process_time = end_time - start_time
            process_times.append(process_time)
            
            # 限制process_times的長度，只保留最近30個
            if len(process_times) > 30:
                process_times.pop(0)
            
            # 每秒更新一次FPS
            if time.time() - last_fps_update >= 1.0:
                if process_times:
                    avg_process_time = sum(process_times) / len(process_times)
                    fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
                last_fps_update = time.time()
            
            # 添加FPS和處理時間信息
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 添加檢測到的人臉數量
            cv2.putText(result_frame, f"Faces Detected: {len(faces)}", (10, 60),  # MODIFIED
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 繪製檢測到的人臉和語音信心度
            for i, face in enumerate(faces):
                box = face["box"]
                x1, y1, x2, y2 = box
                
                # 繪製人臉框
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 如果啟用了音訊分析並且檢測到說話
                if audio_enabled and is_speaking:
                    # 獲取信心度顏色（紅色到綠色的漸變）
                    confidence_color = (
                        0,  # B
                        int(255 * speech_confidence),  # G
                        int(255 * (1 - speech_confidence))  # R
                    )
                    
                    # 將信心度顯示在人臉框下方
                    conf_text = f"Speech Confidence: {speech_confidence:.2f}" # MODIFIED
                    cv2.putText(result_frame, conf_text, (x1, y2 + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, confidence_color, 2)
                    
                    # 顯示識別的文字
                    if speech_text:
                        # 限制文字長度以適應螢幕
                        display_text = speech_text[:30] + "..." if len(speech_text) > 30 else speech_text
                        cv2.putText(result_frame, f"Text: {display_text}", (x1, y2 + 50), # MODIFIED
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 顯示情感分析結果
                    if emotion_label:
                        emotion_text = f"Emotion: {emotion_label} ({emotion_score:.2f})" # MODIFIED
                        cv2.putText(result_frame, emotion_text, (x1, y2 + 75), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    # 只顯示人臉檢測信心度
                    face_conf_text = f"Face Detection Confidence: {face['confidence']:.2f}" # MODIFIED
                    cv2.putText(result_frame, face_conf_text, (x1, y2 + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 在畫面上顯示語音狀態
            if audio_enabled:
                status_text = "Speaking" if is_speaking else "No Speech Detected" # MODIFIED
                status_color = (0, 255, 0) if is_speaking else (0, 0, 255)
                cv2.putText(result_frame, status_text, (frame_width - 200, 30), # Adjusted x position for English text
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # 顯示結果
            cv2.imshow('Real-time Speech Confidence Detection', result_frame) # MODIFIED (window title)
            
            # 按 'q' 或 ESC 鍵退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC
                break
    
    finally:
        # 釋放資源
        cap.release()
        cv2.destroyAllWindows()
        
        # 停止音訊處理
        if audio_enabled and 'audio_processor' in locals(): # Check if audio_processor was initialized
            audio_processor.stop()
        
        print("相機已關閉，程式結束")


def process_image(image_path, model_type='dnn', confidence_threshold=0.5, output_path=None):
    """
    處理單張圖片中的人臉檢測
    
    參數:
        image_path: 圖片路徑
        model_type: 使用的模型類型 ('dnn' 或 'haar')
        confidence_threshold: 最低接受的人臉檢測信心度閾值
        output_path: 輸出圖片路徑
    """
    # 確保模型可用
    if model_type == 'dnn' and not download_face_models():
        print("DNN模型無法使用，切換到Haar分類器")
        model_type = 'haar'
    
    # 讀取圖片
    image_data = cv2.imread(image_path) # Renamed
    if image_data is None:
        print(f"無法讀取圖片: {image_path}")
        return
    
    # 檢測人臉
    result_image, faces = detect_faces(image_data, confidence_threshold, model_type)
    
    # 繪製檢測結果
    for face in faces:
        box = face["box"]
        x1, y1, x2, y2 = box
        
        # 繪製人臉框
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 顯示人臉檢測信心度
        conf_text = f"Confidence: {face['confidence']:.2f}" # MODIFIED
        cv2.putText(result_image, conf_text, (x1, y2 + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 顯示檢測到的人臉數量
    cv2.putText(result_image, f"Faces Detected: {len(faces)}", (10, 30),  # MODIFIED
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 保存或顯示結果
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"結果已保存到: {output_path}")
    else:
        cv2.imshow('Face Detection Result', result_image) # MODIFIED
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_image, faces


def main():
    # 創建命令行參數解析器
    parser = argparse.ArgumentParser(description='人臉和語音信心度分析') # This description is for console, not cv2
    parser.add_argument('--image', help='輸入圖片的路徑')
    parser.add_argument('--threshold', type=float, default=0.5, help='人臉檢測信心度閾值 (0.0-1.0)')
    parser.add_argument('--output', default='output.jpg', help='輸出圖片的路徑')
    parser.add_argument('--model', default='dnn', choices=['dnn', 'haar'], help='選擇使用的模型類型: dnn (需下載) 或 haar (OpenCV內建)')
    parser.add_argument('--webcam', action='store_true', help='使用網絡攝像頭進行實時檢測')
    parser.add_argument('--camera', type=int, default=0, help='相機設備索引（默認為0）')
    
    # 解析命令行參數
    args = parser.parse_args()
    
    # 如果指定了圖片路徑，處理單張圖片
    if args.image:
        process_image(
            image_path=args.image,
            model_type=args.model,
            confidence_threshold=args.threshold,
            output_path=args.output
        )
    # 如果指定了使用網絡攝像頭，進行實時檢測
    elif args.webcam:
        process_webcam_with_speech(
            model_type=args.model,
            confidence_threshold=args.threshold,
            camera_index=args.camera
        )
    else:
        # 如果沒有提供任何參數，默認使用網絡攝像頭
        print("未指定操作模式，默認使用網絡攝像頭進行實時檢測")
        process_webcam_with_speech(
            model_type=args.model,
            confidence_threshold=args.threshold,
            camera_index=args.camera
        )


if __name__ == "__main__":
    main()