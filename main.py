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
from scipy.io import wavfile # librosa can handle wav
import librosa
from transformers import pipeline
import face_recognition # For visual analysis
from collections import deque

# --- Global states from AudioProcessor (can be refactored into a shared state object or class members) ---
# These will be updated by AudioProcessor and read by the main loop
audio_is_speaking = False
audio_speech_text = ""
audio_api_speech_confidence = 0.0 # Confidence from Google API (0-1)
audio_emotion_label = ""
audio_emotion_score = 0.0
# --- End Global states ---

class AudioProcessor:
    def __init__(self, rate=16000, chunk=1024, record_seconds=2): # Shorter record_seconds for faster response
        global audio_is_speaking, audio_speech_text, audio_api_speech_confidence, audio_emotion_label, audio_emotion_score
        self.rate = rate
        self.chunk = chunk
        self.record_seconds = record_seconds
        self.audio_queue = queue.Queue()
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5 # shorter pause for quicker recognition start
        self.recognizer.energy_threshold = 400 # Adjust this based on your mic

        try:
            self.emotion_analyzer = pipeline("text-classification",
                                            model="j-hartmann/emotion-english-distilroberta-base",
                                            top_k=1)
            self.emotion_model_loaded = True
            print("Emotion analysis model loaded.")
        except Exception as e:
            print(f"Could not load emotion model: {e}")
            self.emotion_model_loaded = False

        self.is_running = True
        self.audio_thread = threading.Thread(target=self._process_audio_loop)
        self.audio_thread.daemon = True
        self.recording_thread = None # Will be initialized in start_recording
        self.stream = None
        self.audio = None

    def start_recording(self):
        self.audio = pyaudio.PyAudio()
        try:
            self.stream = self.audio.open(format=pyaudio.paInt16,
                                         channels=1,
                                         rate=self.rate,
                                         input=True,
                                         frames_per_buffer=self.chunk)
            self.recording_thread = threading.Thread(target=self._record_audio_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            self.audio_thread.start() # Start processing thread after recording starts
            print("Audio recording started...")
            return True
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            if self.audio:
                self.audio.terminate()
            return False


    def _record_audio_loop(self):
        while self.is_running and self.stream:
            frames_segment = []
            for _ in range(int(self.rate / self.chunk * self.record_seconds)):
                if not self.is_running or not self.stream or self.stream.is_stopped():
                    break
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    frames_segment.append(data)
                except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        print("DEBUG: Input overflowed. Skipping frame.")
                        continue
                    print(f"Recording error: {e}")
                    self.is_running = False # Stop if major error
                    break
                except Exception as e:
                    print(f"Generic recording error: {e}")
                    self.is_running = False
                    break
            
            if frames_segment and self.is_running:
                self.audio_queue.put(b''.join(frames_segment))
        print("Audio recording loop finished.")


    def _process_audio_loop(self):
        global audio_is_speaking, audio_speech_text, audio_api_speech_confidence, audio_emotion_label, audio_emotion_score
        while self.is_running:
            try:
                audio_data_bytes = self.audio_queue.get(timeout=1)
                
                # Convert bytes to numpy array for librosa
                audio_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                # VAD using librosa RMS
                rms = librosa.feature.rms(y=audio_np)[0]
                avg_rms = np.mean(rms)
                # print(f"DEBUG: avg_rms = {avg_rms:.4f}")
                current_speaking_status = avg_rms > 0.001 # Adjusted VAD threshold

                if current_speaking_status:
                    audio_is_speaking = True # Set speaking true first
                    # Save to temporary WAV for SpeechRecognition
                    temp_filename = "temp_speech.wav"
                    wavfile.write(temp_filename, self.rate, (audio_np * 32767).astype(np.int16))

                    try:
                        with sr.AudioFile(temp_filename) as source:
                            audio_sr = self.recognizer.record(source)
                        
                        # Use a timeout for recognize_google to prevent indefinite blocking
                        try:
                            result = self.recognizer.recognize_google(audio_sr, language='zh-TW', show_all=True)
                        except sr.WaitTimeoutError:
                            print("DEBUG: Google Speech Recognition timed out.")
                            result = None # Treat as no result
                        
                        # print(f"DEBUG: Google API result = {result}")
                        if result and isinstance(result, dict) and result.get('alternative'):
                            best_result = result['alternative'][0]
                            audio_speech_text = best_result.get('transcript', "")
                            print(audio_speech_text)
                            audio_api_speech_confidence = float(best_result.get('confidence', 0.0))

                            if self.emotion_model_loaded and audio_speech_text:
                                try:
                                    emotion_result = self.emotion_analyzer(audio_speech_text)
                                    if emotion_result and emotion_result[0]:
                                        audio_emotion_label = emotion_result[0][0]['label']
                                        audio_emotion_score = emotion_result[0][0]['score']
                                    else:
                                        audio_emotion_label = ""
                                        audio_emotion_score = 0.0
                                except Exception as e_emo:
                                    print(f"Emotion analysis error: {e_emo}")
                                    audio_emotion_label = ""
                                    audio_emotion_score = 0.0
                        else:
                            audio_speech_text = ""
                            audio_api_speech_confidence = 0.0 # No recognition
                            audio_is_speaking = False # If recognition fails, assume not valid speech for confidence
                            audio_emotion_label = ""
                            audio_emotion_score = 0.0

                    except sr.UnknownValueError:
                        # print("DEBUG: Speech not understood by Google.")
                        audio_speech_text = ""
                        audio_api_speech_confidence = 0.0
                        audio_is_speaking = False # If not understood, might not be confident speech
                        audio_emotion_label = ""
                        audio_emotion_score = 0.0
                    except Exception as e_sr:
                        print(f"Speech recognition error: {e_sr}")
                        audio_speech_text = ""
                        audio_api_speech_confidence = 0.0
                        audio_is_speaking = False
                        audio_emotion_label = ""
                        audio_emotion_score = 0.0
                    finally:
                        if os.path.exists(temp_filename):
                            try:
                                os.remove(temp_filename)
                            except Exception:
                                pass # Ignore if deletion fails
                else:
                    audio_is_speaking = False
                    audio_speech_text = "" # Clear text when not speaking
                    # audio_api_speech_confidence = 0.0 # Keep last confidence or reset? For now, reset.
                    # audio_emotion_label = ""
                    # audio_emotion_score = 0.0


            except queue.Empty:
                # If queue is empty, it means no new audio segment, we can assume not speaking
                # or rely on the VAD of the last segment.
                # For simplicity, if no new audio, assume not speaking.
                # However, this might make 'is_speaking' flicker if processing is slow.
                # A better VAD would be continuous.
                # For now, this is okay. If audio_is_speaking was true, it will remain true until VAD says false.
                pass
            except Exception as e_proc:
                print(f"Audio processing error: {e_proc}")
                audio_is_speaking = False # Reset on error
        print("Audio processing loop finished.")

    def stop(self):
        self.is_running = False
        print("Stopping audio processor...")
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2)
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)
        
        if self.stream:
            if not self.stream.is_stopped():
                self.stream.stop_stream()
            self.stream.close()
            print("Audio stream closed.")
        if self.audio:
            self.audio.terminate()
            print("PyAudio terminated.")


class VisualAnalyzer:
    """Analyzes visual cues for confidence from video frames."""
    # --- Constants from OptimizedConfidenceAnalyzer ---
    EYE_CONTACT_WEIGHT = 0.4
    HEAD_STABILITY_WEIGHT = 0.35
    FACIAL_STABILITY_WEIGHT = 0.25
    COMPONENT_SCORE_NEUTRAL_POINT = 5.0
    # ... (other constants like history lengths, thresholds, sensitivities)

    def __init__(self):
        self.confidence_score_visual = 50.0  # 0-100 scale for visual
        self.confidence_history_visual = deque(maxlen=15) # Shorter history for responsiveness

        self.eye_contact_frames = 0
        self.last_frame_had_eye_contact = False
        self.facial_landmarks_history = deque(maxlen=10) # face_recognition landmarks
        self.head_position_history = deque(maxlen=15) # (center_x, center_y) of face bbox

        # Parameters for scoring components (0-10)
        self.MAX_EYE_CONTACT_FRAMES_FOR_SCORE = 30
        self.HEAD_MOVEMENT_SENSITIVITY_DIVISOR = 2.0 # For small_frame movement
        self.FACIAL_FEATURE_CHANGE_SENSITIVITY_MULTIPLIER = 1.0 # For small_frame movement

    def _distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _eye_aspect_ratio(self, eye_landmarks):
        if len(eye_landmarks) != 6: return 0
        v1 = self._distance(eye_landmarks[1], eye_landmarks[5])
        v2 = self._distance(eye_landmarks[2], eye_landmarks[4])
        h = self._distance(eye_landmarks[0], eye_landmarks[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0

    def _analyze_eye_contact_from_landmarks(self, landmarks):
        # This is a simplified version based on eye openness, not true gaze.
        try:
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            is_eye_open_now = avg_ear > 0.20 # EAR_THRESHOLD

            if is_eye_open_now:
                self.eye_contact_frames = min(self.MAX_EYE_CONTACT_FRAMES_FOR_SCORE, self.eye_contact_frames + 1)
            else:
                self.eye_contact_frames = max(0, self.eye_contact_frames - 2)
            
            # Score from 0-10
            return min(10.0, (self.eye_contact_frames / self.MAX_EYE_CONTACT_FRAMES_FOR_SCORE) * 10.0)
        except (KeyError, IndexError):
            self.eye_contact_frames = max(0, self.eye_contact_frames - 2)
            return min(10.0, (self.eye_contact_frames / self.MAX_EYE_CONTACT_FRAMES_FOR_SCORE) * 10.0)


    def _analyze_head_stability_from_bbox(self, face_bbox_scaled): # (top, right, bottom, left)
        top, right, bottom, left = face_bbox_scaled
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        current_head_pos = (center_x, center_y)
        self.head_position_history.append(current_head_pos)

        if len(self.head_position_history) < 5:
            return self.COMPONENT_SCORE_NEUTRAL_POINT
        
        movements = [self._distance(self.head_position_history[i-1], self.head_position_history[i])
                     for i in range(1, len(self.head_position_history))]
        avg_movement = np.mean(movements) if movements else 0
        # Score 0-10, higher stability = less movement
        return max(0.0, 10.0 - (avg_movement / self.HEAD_MOVEMENT_SENSITIVITY_DIVISOR))

    def _calculate_feature_stability(self, feature_name):
        # ... (Copied and adapted from OptimizedConfidenceAnalyzer) ...
        # Ensure it uses self.facial_landmarks_history
        # Returns a score from 0-10 (higher is more stable)
        recent_feature_frames = []
        if not self.facial_landmarks_history: return self.COMPONENT_SCORE_NEUTRAL_POINT

        for landmarks_set in list(self.facial_landmarks_history)[-5:]: # Get last 5
            if feature_name in landmarks_set:
                recent_feature_frames.append(landmarks_set[feature_name])
        
        if len(recent_feature_frames) < 2:
            return self.COMPONENT_SCORE_NEUTRAL_POINT

        total_avg_changes_for_feature = []
        for i in range(1, len(recent_feature_frames)):
            points_prev_frame = recent_feature_frames[i-1]
            points_curr_frame = recent_feature_frames[i]
            
            if len(points_prev_frame) != len(points_curr_frame) or not points_curr_frame:
                continue

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
        stability_score = max(0.0, 10.0 - (overall_avg_change * self.FACIAL_FEATURE_CHANGE_SENSITIVITY_MULTIPLIER))
        return stability_score


    def _analyze_facial_stability_from_landmarks(self):
        if len(self.facial_landmarks_history) < 5:
            return self.COMPONENT_SCORE_NEUTRAL_POINT
        try:
            mouth_stability = (self._calculate_feature_stability('top_lip') +
                               self._calculate_feature_stability('bottom_lip')) / 2.0
            eyebrow_stability = (self._calculate_feature_stability('left_eyebrow') +
                                 self._calculate_feature_stability('right_eyebrow')) / 2.0
            return mouth_stability * 0.6 + eyebrow_stability * 0.4 # Score 0-10
        except (KeyError, IndexError):
            return self.COMPONENT_SCORE_NEUTRAL_POINT

    def analyze_frame(self, frame):
        """Analyzes a single frame for visual confidence cues."""
        # Resize frame for faster processing by face_recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face landmarks in the current frame
        face_locations_scaled = face_recognition.face_locations(rgb_small_frame, model="hog") # "hog" is faster
        
        # Get face landmarks for all faces in the image
        # Pass face_locations to avoid re-detection if needed, though face_landmarks does it.
        all_face_landmarks_scaled = face_recognition.face_landmarks(rgb_small_frame, face_locations_scaled)

        current_visual_score_delta = 0
        processed_face_locations = [] # For drawing bounding boxes later, scaled back to original

        if face_locations_scaled and all_face_landmarks_scaled:
            # For simplicity, analyze the first detected face
            primary_face_location_scaled = face_locations_scaled[0] # (top, right, bottom, left)
            primary_landmarks_scaled = all_face_landmarks_scaled[0]

            self.facial_landmarks_history.append(primary_landmarks_scaled)

            eye_score = self._analyze_eye_contact_from_landmarks(primary_landmarks_scaled)
            head_score = self._analyze_head_stability_from_bbox(primary_face_location_scaled)
            face_expr_score = self._analyze_facial_stability_from_landmarks()

            # Convert component scores (0-10) to deltas from neutral (5.0)
            eye_delta = (eye_score - self.COMPONENT_SCORE_NEUTRAL_POINT) * 0.1 # Smaller multiplier
            head_delta = (head_score - self.COMPONENT_SCORE_NEUTRAL_POINT) * 0.1
            face_expr_delta = (face_expr_score - self.COMPONENT_SCORE_NEUTRAL_POINT) * 0.05

            current_visual_score_delta = (eye_delta * self.EYE_CONTACT_WEIGHT +
                                         head_delta * self.HEAD_STABILITY_WEIGHT +
                                         face_expr_delta * self.FACIAL_STABILITY_WEIGHT)
            
            # Scale back face locations for drawing on the original frame
            for (top, right, bottom, left) in face_locations_scaled:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                processed_face_locations.append((top, right, bottom, left))
        else:
            # No face detected, decrease visual confidence
            current_visual_score_delta = -0.5 # Penalty if no face

        # Update and smooth the visual confidence score (0-100)
        self.confidence_score_visual += current_visual_score_delta
        self.confidence_score_visual = max(0.0, min(100.0, self.confidence_score_visual))
        self.confidence_history_visual.append(self.confidence_score_visual)
        if self.confidence_history_visual:
            self.confidence_score_visual = np.mean(self.confidence_history_visual)

        return self.confidence_score_visual, processed_face_locations


def process_webcam_combined(camera_index=0):
    global audio_is_speaking, audio_speech_text, audio_api_speech_confidence, audio_emotion_label, audio_emotion_score
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera (index: {camera_index})")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow('Combined Confidence Analysis', cv2.WINDOW_NORMAL)

    audio_processor = AudioProcessor()
    audio_started = audio_processor.start_recording()
    if not audio_started:
        print("Audio processing disabled.")
    
    visual_analyzer = VisualAnalyzer()

    # --- Combined Confidence Score ---
    combined_confidence_score = 50.0 # Initial, 0-100
    combined_confidence_history = deque(maxlen=30)

    last_fps_update = time.time()
    fps_frames = 0
    display_fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot get frame from camera.")
                break
            
            frame = cv2.flip(frame, 1) # Mirroring for more natural view
            fps_frames +=1

            # --- Visual Analysis ---
            visual_confidence, face_bboxes = visual_analyzer.analyze_frame(frame.copy()) # Pass a copy

            # --- Score Fusion Logic ---
            # Start with visual confidence as a base or major factor
            current_combined_score = visual_confidence 

            if audio_started and audio_is_speaking:
                # Adjust based on speech API confidence (0-1 scale)
                # If API confidence is high, it can boost the score, if low, slightly penalize.
                # Example: if api_speech_confidence is 0.8, add (0.8-0.5)*20 = 6 points
                # if api_speech_confidence is 0.3, add (0.3-0.5)*20 = -4 points
                speech_api_bonus = (audio_api_speech_confidence - 0.5) * 20 
                current_combined_score += speech_api_bonus

                # Adjust based on emotion
                if audio_emotion_label:
                    if audio_emotion_label in ['joy', 'surprise'] and audio_emotion_score > 0.7:
                        current_combined_score += 5
                    elif audio_emotion_label in ['anger'] and audio_emotion_score > 0.6: # Anger can be confident
                        current_combined_score += 3
                    elif audio_emotion_label in ['fear', 'sadness'] and audio_emotion_score > 0.6:
                        current_combined_score -= 10
                    elif audio_emotion_label in ['disgust'] and audio_emotion_score > 0.6:
                        current_combined_score -= 5
            
            current_combined_score = max(0.0, min(100.0, current_combined_score))
            combined_confidence_history.append(current_combined_score)
            if combined_confidence_history:
                combined_confidence_score = np.mean(combined_confidence_history)

            # --- Display ---
            # FPS
            if time.time() - last_fps_update >= 1.0:
                display_fps = fps_frames
                fps_frames = 0
                last_fps_update = time.time()
            cv2.putText(frame, f"FPS: {display_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw face boxes from visual analyzer
            for (top, right, bottom, left) in face_bboxes:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Display combined score near the first face (or all faces if logic is extended)
                score_text = f"Conf: {combined_confidence_score:.1f}"
                color = (0,0,255) # Red
                if combined_confidence_score > 66: color = (0,255,0) # Green
                elif combined_confidence_score > 33: color = (0,255,255) # Yellow
                cv2.putText(frame, score_text, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if audio_started and audio_is_speaking:
                    cv2.putText(frame, f"Speech: {audio_speech_text[:20]}", (left, bottom + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    if audio_emotion_label:
                         cv2.putText(frame, f"Emotion: {audio_emotion_label} ({audio_emotion_score:.2f})", (left, bottom + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)


            # Audio status
            if audio_started:
                audio_status_text = "Speaking" if audio_is_speaking else "Quiet"
                audio_status_color = (0,255,0) if audio_is_speaking else (0,0,255)
                cv2.putText(frame, audio_status_text, (frame_width -150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, audio_status_color, 2)


            cv2.imshow('Combined Confidence Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Cleaning up...")
        if audio_started:
            audio_processor.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Program finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combined Visual and Audio Confidence Analyzer.')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index.')
    args = parser.parse_args()
    process_webcam_combined(camera_index=args.camera)