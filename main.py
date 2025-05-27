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
from transformers import pipeline, logging as hf_logging
import face_recognition # For visual analysis
from collections import deque

# Suppress some informational messages from transformers
hf_logging.set_verbosity_error()


# --- Global states from AudioProcessor (can be refactored into a shared state object or class members) ---
audio_is_speaking = False
audio_speech_text = ""
audio_api_speech_confidence = 0.0 # Confidence from Google API (0-1)
audio_emotion_label = ""
audio_emotion_score = 0.0
# --- End Global states ---

class AudioProcessor:
    def __init__(self, rate=16000, chunk=1024, record_seconds=2):
        global audio_is_speaking, audio_speech_text, audio_api_speech_confidence, audio_emotion_label, audio_emotion_score
        self.rate = rate
        self.chunk = chunk
        self.record_seconds = record_seconds
        self.audio_queue = queue.Queue()
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5
        self.recognizer.energy_threshold = 400 # Adjust this based on your mic
        self.recognizer.dynamic_energy_threshold = True # Allow adjustment

        try:
            # Using a more general sentiment/emotion model, can be swapped
            self.emotion_analyzer = pipeline("text-classification",
                                            model="j-hartmann/emotion-english-distilroberta-base",
                                            top_k=1)
            self.emotion_model_loaded = True
            print("Audio emotion analysis model loaded.")
        except Exception as e:
            print(f"Could not load audio emotion model: {e}. Audio emotion analysis disabled.")
            self.emotion_model_loaded = False

        self.is_running = True
        self.audio_thread = threading.Thread(target=self._process_audio_loop)
        self.audio_thread.daemon = True
        self.recording_thread = None
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
            self.audio_thread.start()
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
                    if hasattr(e, 'errno') and e.errno == pyaudio.paInputOverflowed:
                        # print("DEBUG: Input overflowed. Skipping frame.") # Can be noisy
                        continue
                    print(f"Recording IO error: {e}")
                    # self.is_running = False # Commenting out to be more resilient
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
                audio_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                rms = librosa.feature.rms(y=audio_np)[0]
                avg_rms = np.mean(rms)
                
                # Increased VAD threshold slightly
                current_speaking_status = avg_rms > 0.008 # PDF: Focus on actual speech for confidence

                if current_speaking_status:
                    temp_filename = "temp_speech.wav"
                    wavfile.write(temp_filename, self.rate, (audio_np * 32767).astype(np.int16))

                    try:
                        with sr.AudioFile(temp_filename) as source:
                            audio_sr = self.recognizer.record(source)
                        
                        result = None
                        try:
                            # Recognize speech using Google Speech Recognition with a timeout
                            result = self.recognizer.recognize_google(audio_sr, language='zh-TW', show_all=True)
                        except sr.WaitTimeoutError:
                            print("DEBUG: Google Speech Recognition timed out.")
                        except sr.UnknownValueError:
                            # Speech not understood by Google
                            audio_speech_text = "[Unintelligible]"
                            audio_api_speech_confidence = 0.0
                            audio_is_speaking = False # PDF: If not understood, less likely confident speech
                            audio_emotion_label = ""
                            audio_emotion_score = 0.0
                        except sr.RequestError as e:
                            print(f"Could not request results from Google Speech Recognition service; {e}")
                            audio_speech_text = "[API Error]"
                            audio_api_speech_confidence = 0.0
                            audio_is_speaking = False
                            audio_emotion_label = ""
                            audio_emotion_score = 0.0

                        if result and isinstance(result, dict) and result.get('alternative'):
                            best_result = result['alternative'][0]
                            audio_speech_text = best_result.get('transcript', "")
                            audio_api_speech_confidence = float(best_result.get('confidence', 0.0))
                            
                            # PDF: Low confidence from STT itself might indicate lack of speaker confidence
                            if audio_api_speech_confidence < 0.5: # Threshold for "confident" speech
                                audio_is_speaking = False # Treat low-confidence STT as not speaking confidently
                            else:
                                audio_is_speaking = True

                            if self.emotion_model_loaded and audio_speech_text and audio_is_speaking:
                                try:
                                    # For non-English, this model won't be accurate. Need Chinese model or translate.
                                    # For now, assuming English for text emotion for demo purposes if zh-TW fails.
                                    emotion_result = self.emotion_analyzer(audio_speech_text if all(ord(c) < 128 for c in audio_speech_text) else "neutral speech")
                                    if emotion_result and emotion_result[0]:
                                        audio_emotion_label = emotion_result[0][0]['label']
                                        audio_emotion_score = emotion_result[0][0]['score']
                                    else: # Should not happen with top_k=1
                                        audio_emotion_label = ""
                                        audio_emotion_score = 0.0
                                except Exception as e_emo:
                                    print(f"Audio emotion analysis error: {e_emo}")
                                    audio_emotion_label = ""
                                    audio_emotion_score = 0.0
                        else: # Result is None or not in expected format
                            if not audio_speech_text : # If not already set by UnknownValueError
                                audio_speech_text = ""
                            if audio_api_speech_confidence == 0.0: # If not already set
                                audio_is_speaking = False
                            audio_emotion_label = ""
                            audio_emotion_score = 0.0

                    except Exception as e_sr:
                        # print(f"Speech recognition processing error: {e_sr}") # Can be too verbose
                        audio_speech_text = ""
                        audio_api_speech_confidence = 0.0
                        audio_is_speaking = False
                        audio_emotion_label = ""
                        audio_emotion_score = 0.0
                    finally:
                        if os.path.exists(temp_filename):
                            try:
                                os.remove(temp_filename)
                            except Exception: pass
                else: # Not speaking based on VAD
                    audio_is_speaking = False
                    # Don't clear speech_text immediately, let it persist for a bit for display
                    # audio_api_speech_confidence = 0.0 # Let fusion logic handle non-speaking
                    # audio_emotion_label = ""
                    # audio_emotion_score = 0.0

            except queue.Empty:
                pass # No new audio data
            except Exception as e_proc:
                print(f"Audio processing loop error: {e_proc}")
                audio_is_speaking = False
        print("Audio processing loop finished.")

    def stop(self):
        self.is_running = False
        print("Stopping audio processor...")
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1) # Shorter timeout
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1)
        
        if self.stream:
            if not self.stream.is_stopped(): self.stream.stop_stream()
            self.stream.close()
            print("Audio stream closed.")
        if self.audio:
            self.audio.terminate()
            print("PyAudio terminated.")


class VisualAnalyzer:
    EYE_CONTACT_WEIGHT = 0.35        # PDF: Eye behavior is a key nonverbal cue
    HEAD_STABILITY_WEIGHT = 0.30     # PDF: Excessive movement or rigidity can be indicators
    FACIAL_STABILITY_WEIGHT = 0.20   # PDF: Microexpressions vs. controlled expressions
    # Added micro-expression proxy
    MICRO_EXPRESSION_PROXY_WEIGHT = 0.15 # PDF: Brief emotional leakage

    COMPONENT_SCORE_NEUTRAL_POINT = 5.0 # 0-10 scale, 5 is neutral

    def __init__(self):
        self.confidence_score_visual = 50.0
        self.confidence_history_visual = deque(maxlen=10) # Shorter history for faster visual response

        self.eye_contact_frames = 0
        self.facial_landmarks_history = deque(maxlen=5) # Shorter history for micro-expression proxy
        self.head_position_history = deque(maxlen=10)
        self.no_face_in_previous_frame = True

        self.MAX_EYE_CONTACT_FRAMES_FOR_SCORE = 20 # Reduced for faster response
        self.HEAD_MOVEMENT_SENSITIVITY_DIVISOR = 1.5 # Increased sensitivity
        self.FACIAL_FEATURE_CHANGE_SENSITIVITY_MULTIPLIER = 1.5 # Increased sensitivity

        # For micro-expression proxy (sudden large changes in landmarks)
        self.last_overall_feature_positions = None
        self.micro_expression_proxy_score = self.COMPONENT_SCORE_NEUTRAL_POINT


    def _distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _eye_aspect_ratio(self, eye_landmarks):
        if len(eye_landmarks) != 6: return 0
        v1 = self._distance(eye_landmarks[1], eye_landmarks[5])
        v2 = self._distance(eye_landmarks[2], eye_landmarks[4])
        h = self._distance(eye_landmarks[0], eye_landmarks[3])
        return (v1 + v2) / (2.0 * h) if h > 0.001 else 0 # Avoid division by zero

    def _analyze_eye_contact_from_landmarks(self, landmarks):
        try:
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            is_eye_open_now = avg_ear > 0.18 # Adjusted EAR_THRESHOLD slightly lower

            if is_eye_open_now:
                self.eye_contact_frames = min(self.MAX_EYE_CONTACT_FRAMES_FOR_SCORE, self.eye_contact_frames + 1)
            else: # Eyes closed or mostly closed
                self.eye_contact_frames = max(0, self.eye_contact_frames - 1.5) # Penalize closed eyes more gently
            
            return min(10.0, (self.eye_contact_frames / self.MAX_EYE_CONTACT_FRAMES_FOR_SCORE) * 10.0)
        except (KeyError, IndexError):
            self.eye_contact_frames = max(0, self.eye_contact_frames - 1.5)
            return min(10.0, (self.eye_contact_frames / self.MAX_EYE_CONTACT_FRAMES_FOR_SCORE) * 10.0)

    def _analyze_head_stability_from_bbox(self, face_bbox_scaled):
        top, right, bottom, left = face_bbox_scaled
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        current_head_pos = (center_x, center_y)
        self.head_position_history.append(current_head_pos)

        if len(self.head_position_history) < 3: # Need at least 3 for some movement
            return self.COMPONENT_SCORE_NEUTRAL_POINT
        
        movements = [self._distance(self.head_position_history[i-1], self.head_position_history[i])
                     for i in range(1, len(self.head_position_history))]
        avg_movement = np.mean(movements) if movements else 0
        return max(0.0, 10.0 - (avg_movement / self.HEAD_MOVEMENT_SENSITIVITY_DIVISOR))

    def _calculate_feature_stability(self, feature_name, history):
        recent_feature_frames = []
        if not history: return self.COMPONENT_SCORE_NEUTRAL_POINT

        for landmarks_set in list(history): # Use provided history
            if feature_name in landmarks_set:
                recent_feature_frames.append(landmarks_set[feature_name])
        
        if len(recent_feature_frames) < 2:
            return self.COMPONENT_SCORE_NEUTRAL_POINT

        total_avg_changes_for_feature = []
        for i in range(1, len(recent_feature_frames)):
            points_prev_frame = recent_feature_frames[i-1]
            points_curr_frame = recent_feature_frames[i]
            if len(points_prev_frame) != len(points_curr_frame) or not points_curr_frame: continue
            frame_change = sum(self._distance(points_curr_frame[j], points_prev_frame[j]) for j in range(len(points_curr_frame)))
            avg_change_this_frame_pair = frame_change / len(points_curr_frame) if len(points_curr_frame) > 0 else 0
            total_avg_changes_for_feature.append(avg_change_this_frame_pair)
        
        if not total_avg_changes_for_feature: return self.COMPONENT_SCORE_NEUTRAL_POINT
        overall_avg_change = sum(total_avg_changes_for_feature) / len(total_avg_changes_for_feature)
        return max(0.0, 10.0 - (overall_avg_change * self.FACIAL_FEATURE_CHANGE_SENSITIVITY_MULTIPLIER))

    def _analyze_facial_stability_from_landmarks(self):
        if len(self.facial_landmarks_history) < 3:
            return self.COMPONENT_SCORE_NEUTRAL_POINT
        
        # More weight to mouth as it's active during speech
        mouth_stability = (self._calculate_feature_stability('top_lip', self.facial_landmarks_history) +
                           self._calculate_feature_stability('bottom_lip', self.facial_landmarks_history)) / 2.0
        eyebrow_stability = (self._calculate_feature_stability('left_eyebrow', self.facial_landmarks_history) +
                             self._calculate_feature_stability('right_eyebrow', self.facial_landmarks_history)) / 2.0
        
        return mouth_stability * 0.6 + eyebrow_stability * 0.4

    def _analyze_micro_expression_proxy(self, current_landmarks):
        # PDF: Microexpressions are rapid, involuntary. This is a HUGELY simplified proxy.
        # It looks for large, sudden shifts in overall facial feature positions.
        # Real microexpression detection is far more complex.
        
        if not current_landmarks:
            self.micro_expression_proxy_score = max(0, self.micro_expression_proxy_score - 0.5) # Penalize if no landmarks for this
            return self.micro_expression_proxy_score

        # Aggregate some key points (e.g., nose, chin, eyebrow centers)
        current_points = []
        for feature in ['nose_bridge', 'chin', 'left_eyebrow', 'right_eyebrow']:
            if feature in current_landmarks and current_landmarks[feature]:
                # Take average point of multi-point features, or first point
                avg_x = np.mean([p[0] for p in current_landmarks[feature]])
                avg_y = np.mean([p[1] for p in current_landmarks[feature]])
                current_points.append((avg_x, avg_y))
        
        if not current_points:
            self.micro_expression_proxy_score = max(0, self.micro_expression_proxy_score - 0.5)
            return self.micro_expression_proxy_score

        current_avg_pos = np.mean(current_points, axis=0)

        if self.last_overall_feature_positions is not None:
            change = self._distance(current_avg_pos, self.last_overall_feature_positions)
            # If change is large and sudden, it might be a "leak." Penalize confidence.
            # This is very heuristic. Thresholds need tuning.
            if change > 3.0: # Tunable threshold for "large" change in scaled frame
                self.micro_expression_proxy_score = max(0, self.micro_expression_proxy_score - 2.0) # Penalize
            else: # Small change, more stable
                self.micro_expression_proxy_score = min(10, self.micro_expression_proxy_score + 0.5) # Recover
        
        self.last_overall_feature_positions = current_avg_pos
        return self.micro_expression_proxy_score


    def analyze_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # Process smaller frame
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations_scaled = face_recognition.face_locations(rgb_small_frame, model="hog")
        all_face_landmarks_scaled = face_recognition.face_landmarks(rgb_small_frame, face_locations_scaled)

        current_visual_score_delta = 0
        processed_face_locations = []

        if face_locations_scaled and all_face_landmarks_scaled:
            primary_face_location_scaled = face_locations_scaled[0]
            primary_landmarks_scaled = all_face_landmarks_scaled[0]

            if self.no_face_in_previous_frame: # Face just appeared
                self.facial_landmarks_history.clear()
                self.head_position_history.clear()
                self.confidence_score_visual = 50.0 # Reset to neutral
                self.confidence_history_visual.clear()
                self.confidence_history_visual.append(50.0)
                self.eye_contact_frames = self.MAX_EYE_CONTACT_FRAMES_FOR_SCORE / 2 # Start neutral
                self.last_overall_feature_positions = None # Reset for micro-expression proxy
                self.micro_expression_proxy_score = self.COMPONENT_SCORE_NEUTRAL_POINT
            self.no_face_in_previous_frame = False

            self.facial_landmarks_history.append(primary_landmarks_scaled)

            eye_score = self._analyze_eye_contact_from_landmarks(primary_landmarks_scaled)
            head_score = self._analyze_head_stability_from_bbox(primary_face_location_scaled)
            face_expr_score = self._analyze_facial_stability_from_landmarks()
            micro_expr_proxy = self._analyze_micro_expression_proxy(primary_landmarks_scaled)

            # Convert component scores (0-10) to deltas from neutral (5.0)
            # Increased multiplier for more impact
            eye_delta = (eye_score - self.COMPONENT_SCORE_NEUTRAL_POINT) * 0.25 
            head_delta = (head_score - self.COMPONENT_SCORE_NEUTRAL_POINT) * 0.20
            face_expr_delta = (face_expr_score - self.COMPONENT_SCORE_NEUTRAL_POINT) * 0.15
            micro_expr_proxy_delta = (micro_expr_proxy - self.COMPONENT_SCORE_NEUTRAL_POINT) * 0.15


            current_visual_score_delta = (eye_delta * self.EYE_CONTACT_WEIGHT +
                                         head_delta * self.HEAD_STABILITY_WEIGHT +
                                         face_expr_delta * self.FACIAL_STABILITY_WEIGHT +
                                         micro_expr_proxy_delta * self.MICRO_EXPRESSION_PROXY_WEIGHT)
            
            for (top, right, bottom, left) in face_locations_scaled:
                processed_face_locations.append((top * 4, right * 4, bottom * 4, left * 4))
        else: # No face detected
            # Stronger penalty if no face, to pull score down faster
            current_visual_score_delta = -1.5 
            if not self.no_face_in_previous_frame: # Face was just lost
                 self.last_overall_feature_positions = None # Reset for micro-expression proxy
            self.no_face_in_previous_frame = True

        self.confidence_score_visual += current_visual_score_delta
        self.confidence_score_visual = max(0.0, min(100.0, self.confidence_score_visual))
        self.confidence_history_visual.append(self.confidence_score_visual)
        
        # Calculate mean of history only if it's not empty
        smoothed_visual_score = np.mean(self.confidence_history_visual) if self.confidence_history_visual else 50.0
        
        return smoothed_visual_score, processed_face_locations


def process_webcam_combined(camera_index=0):
    global audio_is_speaking, audio_speech_text, audio_api_speech_confidence, audio_emotion_label, audio_emotion_score
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera (index: {camera_index})")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cv2.namedWindow('Combined Confidence Analysis', cv2.WINDOW_NORMAL)

    audio_processor = AudioProcessor()
    audio_started = audio_processor.start_recording()
    if not audio_started: print("Audio processing disabled.")
    
    visual_analyzer = VisualAnalyzer()

    combined_confidence_score = 50.0
    # Shorter history for combined score for faster response
    combined_confidence_history = deque(maxlen=15) 

    last_fps_update = time.time()
    fps_frames = 0
    display_fps = 0
    
    # For persisting speech text a bit longer
    display_speech_text = ""
    display_speech_timer = 0
    DISPLAY_SPEECH_DURATION = 50 # frames

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            fps_frames +=1

            visual_confidence, face_bboxes = visual_analyzer.analyze_frame(frame.copy())

            # --- Score Fusion Logic (PDF Inspired) ---
            # Base score is visual, modulated by audio cues when speaking
            current_combined_score = visual_confidence 
            audio_adjustment = 0

            if audio_started and audio_is_speaking:
                # Update display text if new speech
                if audio_speech_text and audio_speech_text != display_speech_text:
                    display_speech_text = audio_speech_text
                    display_speech_timer = DISPLAY_SPEECH_DURATION

                # PDF: Speech API confidence matters
                # If API confidence is high, boost; if low, penalize.
                # Adjusted neutral point for STT confidence and multiplier
                speech_api_bonus = (audio_api_speech_confidence - 0.6) * 25 
                audio_adjustment += speech_api_bonus

                # PDF Table 1: Emotions and their link to confidence/deception
                if audio_emotion_label:
                    # Stronger impact for fear/sadness (lack of confidence)
                    if audio_emotion_label == 'fear' and audio_emotion_score > 0.5:
                        audio_adjustment -= 15 
                    elif audio_emotion_label == 'sadness' and audio_emotion_score > 0.5:
                        audio_adjustment -= 10
                    # Duping delight (joy incongruent) or genuine confidence
                    elif audio_emotion_label == 'joy' and audio_emotion_score > 0.6:
                        # Could check speech text for incongruence, complex. For now, small positive.
                        audio_adjustment += 5 
                    # Anger can be defensive (lower confidence) or assertive (higher)
                    elif audio_emotion_label == 'anger' and audio_emotion_score > 0.5:
                        audio_adjustment -= 5 # Assume defensive for now
                    elif audio_emotion_label == 'surprise' and audio_emotion_score > 0.5:
                        # PDF: Surprise can mean cognitive load, slight penalty
                        audio_adjustment -= 3
                    elif audio_emotion_label == 'disgust' and audio_emotion_score > 0.5:
                        # PDF: Disgust towards self/situation -> lower confidence
                        audio_adjustment -= 7
            else: # Not speaking or audio not reliable
                # If not speaking, the visual score is the primary driver.
                # No specific penalty here, as visual_confidence already reflects visual cues.
                # If visual_confidence is low (e.g. no face), combined score will be low.
                 if display_speech_timer > 0:
                    display_speech_timer -=1
                 else:
                    display_speech_text = ""


            current_combined_score += audio_adjustment
            current_combined_score = max(0.0, min(100.0, current_combined_score))
            combined_confidence_history.append(current_combined_score)
            
            # Calculate mean of history only if it's not empty
            combined_confidence_score = np.mean(combined_confidence_history) if combined_confidence_history else 50.0


            # --- Display ---
            if time.time() - last_fps_update >= 1.0:
                display_fps = fps_frames
                fps_frames = 0
                last_fps_update = time.time()
            cv2.putText(frame, f"FPS: {display_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            y_offset = 30
            for (top, right, bottom, left) in face_bboxes:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                score_text = f"Conf: {combined_confidence_score:.1f}"
                color = (0,0,255) # Red
                if combined_confidence_score > 66: color = (0,255,0) # Green
                elif combined_confidence_score > 33: color = (0,255,255) # Yellow
                cv2.putText(frame, score_text, (left, bottom + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 25

                if audio_started:
                    if display_speech_text: # Persist speech text display
                        cv2.putText(frame, f"Speech: {display_speech_text[:30]}", (left, bottom + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        y_offset += 20
                    if audio_is_speaking and audio_emotion_label: # Only show emotion if actively speaking
                         cv2.putText(frame, f"Emotion: {audio_emotion_label} ({audio_emotion_score:.2f})", (left, bottom + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                         y_offset += 20
                break # Only display for the first detected face for now

            if audio_started:
                audio_status_text = "Speaking" if audio_is_speaking else "Quiet"
                audio_status_color = (0,255,0) if audio_is_speaking else (128,128,128) # Grey for quiet
                cv2.putText(frame, audio_status_text, (frame_width -150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, audio_status_color, 2)
                if audio_is_speaking:
                    cv2.putText(frame, f"STT Conf: {audio_api_speech_confidence:.2f}", (frame_width -150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


            cv2.imshow('Combined Confidence Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Cleaning up...")
        if audio_started: audio_processor.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Program finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combined Visual and Audio Confidence Analyzer.')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index.')
    args = parser.parse_args()
    process_webcam_combined(camera_index=args.camera)
