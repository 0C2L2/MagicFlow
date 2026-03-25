import sys
import cv2
import numpy as np
import time
import math
import os
import mss
import wave
import pyaudio
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QTimer, QSize
from PyQt5.QtGui import QPainter, QPen, QColor, QCursor, QImage, QPixmap
import pyautogui

# Import our tracker 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class TrackerThread(QThread):
    # (x, y, state) where state: 0=Hover, 1=Draw
    coord_signal = pyqtSignal(int, int, int)
    # (event_id) where 1=Clear, 4=Exit, 5=SwipeRight, 6=SwipeLeft, 7=StartRec, 8=StopRec
    event_signal = pyqtSignal(int)
    # (QImage) for PiP view
    frame_signal = pyqtSignal(QImage)
    
    def __init__(self, model_path='hand_landmarker.task'):
        super().__init__()
        self.model_path = model_path
        self.running = True
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Smoothing variables - Preferred "Smooth" balance
        self.smooth_x, self.smooth_y = 0, 0
        self.alpha = 0.2 
        self.deadzone = 5 
        
        # State debouncing
        self.draw_count = 0
        self.draw_threshold = 4 
        self.is_drawing_active = False
        
        # Slide Control (Phase 3)
        self.can_trigger_slide = True 
        self.slide_reset_counter = 0 # Debounce for reset
        self.last_slide_time = 0 # Absolute cooldown
        
        # Recording Gesture (Phase 4)
        self.can_toggle_record = True # Hysteresis for recording trigger
        
    def run(self):
        # Initialize MediaPipe Tasks API
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            running_mode=vision.RunningMode.VIDEO
        )
        detector = vision.HandLandmarker.create_from_options(options)
        
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        
        while self.running:
            success, frame = cap.read()
            if not success:
                continue
                
            frame = cv2.flip(frame, 1) # Mirror for intuitive control
            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            results = detector.detect_for_video(mp_image, timestamp_ms)
            
            if results and results.hand_landmarks:
                landmarks = results.hand_landmarks[0]
                
                # Finger extension detection
                fingers = []
                # Thumb (different logic for horizontal)
                if landmarks[4].x < landmarks[3].x: 
                    fingers.append(1)
                else:
                    fingers.append(0)
                
                # 4 Fingers (Tip higher than middle joint PIP)
                for i in [8, 12, 16, 20]:
                    if landmarks[i].y < landmarks[i-2].y: 
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                # --- CONTROL GESTURES ---
                # 1. Peace Sign (Clear) - ✌️
                # Index (8) and Middle (12) are up. Others closed.
                if fingers == [0, 1, 1, 0, 0]:
                    self.event_signal.emit(1)
                    
                # 2. Pinky Up (Close) - 🤙
                elif fingers == [0, 0, 0, 0, 1]:
                    self.event_signal.emit(4)
                
                # 3. Ring Finger Up (Start Recording) - 💍
                elif fingers == [0, 0, 0, 1, 0]:
                    if self.can_toggle_record:
                        self.event_signal.emit(7) # START Recording
                        self.can_toggle_record = False
                
                # 4. Open Palm (Stop Recording) - ✋ (at least 4 fingers up)
                elif sum(fingers[1:]) >= 4: # Index, Middle, Ring, Pinky all up
                    if self.can_toggle_record:
                        self.event_signal.emit(8) # STOP Recording
                        self.can_toggle_record = False
                
                else:
                    self.can_toggle_record = True # Reset on palm change
                
                # --- PROFESSOR VIEW (PiP) ---
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_signal.emit(qt_img.copy())

                # Index Tip (8), Thumb Tip (4), and Index MCP (5 - Knuckle)
                idx_tip = landmarks[8]
                thumb_tip = landmarks[4]
                idx_mcp = landmarks[5]
                
                # Tracking logic...
                margin = 0.1
                raw_x = int(np.interp(idx_mcp.x, [margin, 1.0-margin], [0, self.screen_width]))
                raw_y = int(np.interp(idx_mcp.y, [margin, 1.0-margin], [0, self.screen_height]))
                
                if self.smooth_x == 0:
                    self.smooth_x, self.smooth_y = raw_x, raw_y
                else:
                    target_x = int(self.alpha * raw_x + (1 - self.alpha) * self.smooth_x)
                    target_y = int(self.alpha * raw_y + (1 - self.alpha) * self.smooth_y)
                    
                    dist_moved = math.hypot(target_x - self.smooth_x, target_y - self.smooth_y)
                    if dist_moved > self.deadzone or self.is_drawing_active:
                        self.smooth_x, self.smooth_y = target_x, target_y
                
                # --- SLIDE CONTROL (Phase 3 - Pointing Pose) ---
                # Index must be up, others down. Thumb flexible.
                is_pointing_pose = (fingers[1] == 1 and sum(fingers[2:]) == 0)
                
                if not self.is_drawing_active and is_pointing_pose:
                    # Check Horizontal Direction (Tip vs Knuckle)
                    dx = landmarks[8].x - landmarks[5].x # Current mirrored 
                    dy = landmarks[8].y - landmarks[5].y
                    
                    now = time.time()
                    if self.can_trigger_slide and (now - self.last_slide_time) > 0.6:
                        # Index must be clearly more horizontal than vertical
                        if abs(dx) > 0.08 and abs(dx) > abs(dy):
                            if dx > 0.08: # Pointing Right (Next)
                                self.event_signal.emit(5)
                                self.can_trigger_slide = False
                                self.last_slide_time = now
                            elif dx < -0.08: # Pointing Left (Prev)
                                self.event_signal.emit(6)
                                self.can_trigger_slide = False
                                self.last_slide_time = now
                    
                    self.slide_reset_counter = 0 # Don't reset if we're still pointing
                else:
                    # Increment counter when NOT in pointing pose
                    self.slide_reset_counter += 1
                    # Must be away from pointing pose for ~10 frames (~0.3s) to allow next slide
                    if self.slide_reset_counter > 10:
                        self.can_trigger_slide = True
                
                dist_pinch = math.hypot(idx_tip.x - thumb_tip.x, idx_tip.y - thumb_tip.y)
                
                if dist_pinch < 0.05: 
                    self.is_drawing_active = True
                    self.draw_count = self.draw_threshold 
                else:
                    if self.draw_count > 0:
                        self.draw_count -= 1
                    else:
                        self.is_drawing_active = False

                state = 1 if self.is_drawing_active else 0
                self.coord_signal.emit(self.smooth_x, self.smooth_y, state)
            else:
                # Immediate safety reset on hand loss
                if self.is_drawing_active:
                    self.is_drawing_active = False
                    self.coord_signal.emit(self.smooth_x, self.smooth_y, 0)
                self.smooth_x, self.smooth_y = 0, 0
                self.can_trigger_slide = True # Reset on hand loss
            
            time.sleep(0.005) # Back to stable 5ms sleep
            
        cap.release()
        detector.close()

class RecorderThread(QThread):
    def __init__(self, width, height, fps=20):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        
        if not os.path.exists("recordings"):
            os.makedirs("recordings")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_fn = f"recordings/temp_video_{timestamp}.mp4"
        self.audio_fn = f"recordings/temp_audio_{timestamp}.wav"
        self.final_fn = f"recordings/MagicFlow_{timestamp}.mp4"
        
    def run(self):
        # Video Setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_fn, fourcc, self.fps, (self.width, self.height))
        
        # Audio Setup
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        audio_frames = []
        
        self.running = True
        with mss.mss() as sct:
            monitor = sct.monitors[1] 
            while self.running:
                loop_start = time.time()
                
                # Video Capture
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                out.write(frame)
                
                # Audio Capture (non-blocking chunk)
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                    audio_frames.append(data)
                except:
                    pass
                
                # FPS Control
                elapsed = time.time() - loop_start
                wait_time = max(0, (1.0 / self.fps) - elapsed)
                if wait_time > 0:
                    time.sleep(wait_time)
                    
        out.release()
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save Audio
        wf = wave.open(self.audio_fn, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        
        # Mux with FFmpeg
        cmd = f'ffmpeg -y -i "{self.video_fn}" -i "{self.audio_fn}" -c:v copy -c:a aac -strict experimental "{self.final_fn}"'
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Cleanup temp files
            os.remove(self.video_fn)
            os.remove(self.audio_fn)
        except:
             print("FFmpeg muxing failed. Audio and video kept separate.")

class MagicCanvas(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("MagicFlow Overlay")
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.WindowTransparentForInput | 
            Qt.WA_TranslucentBackground
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.screen_w, self.screen_h = pyautogui.size()
        self.setGeometry(0, 0, self.screen_w, self.screen_h)
        
        self.points = [] 
        self.current_pos = QPoint(0, 0)
        self.is_drawing = False
        self.swipe_flash = 0 # Visual pulse counter
        
        self.recorder = None 
        self.is_recording = False
        
        # PiP View (Professor View)
        self.pip_label = QLabel(self)
        self.pip_label.setFixedSize(240, 180) # 4:3 aspect
        self.pip_label.move(self.screen_w - 260, self.screen_h - 200)
        self.pip_label.setStyleSheet("border: 2px solid green; border-radius: 10px; background: black;")
        self.pip_label.show()
        
        self.tracker = TrackerThread()
        self.tracker.coord_signal.connect(self.update_canvas)
        self.tracker.event_signal.connect(self.handle_event)
        self.tracker.frame_signal.connect(self.update_pip)
        self.tracker.start()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)
        
    def update_pip(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.pip_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.pip_label.setPixmap(pixmap)

    def handle_event(self, event_id):
        if event_id == 1: # Clear ✌️
            self.points = []
        elif event_id == 4: # Exit 🤙
            self.stop_recording_if_active()
            self.tracker.running = False
            self.tracker.wait()
            self.close()
            sys.exit(0)
        elif event_id == 6: # Swipe Left -> Prev Slide
            pyautogui.press('left')
            self.swipe_flash = 5
        elif event_id == 5: # Swipe Right -> Next Slide
            pyautogui.press('right')
            self.swipe_flash = 5 # Pulse yellow
        elif event_id == 7: # START Recording 💍
            if not self.is_recording:
                self.recorder = RecorderThread(self.screen_w, self.screen_h)
                self.recorder.start()
                self.is_recording = True
        elif event_id == 8: # STOP Recording ✋
            self.stop_recording_if_active()
            
        self.update()

    def stop_recording_if_active(self):
        if self.is_recording and self.recorder:
            self.recorder.running = False
            self.recorder.wait()
            self.is_recording = False
            self.recorder = None

    def update_canvas(self, x, y, state):
        self.current_pos = QPoint(x, y)
        self.is_drawing = (state == 1)
        
        if self.is_drawing:
            self.points.append((QPoint(x, y), True))
        else:
            if self.points and self.points[-1][1]:
                self.points.append((QPoint(x, y), False))
        
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Cursor Visuals
        painter.setPen(Qt.NoPen)
        if self.swipe_flash > 0:
            color = QColor(255, 255, 0, 200) # Yellow pulse on swipe
            self.swipe_flash -= 1
        else:
            color = QColor(0, 255, 0, 150) if self.is_drawing else QColor(255, 0, 255, 150)
        
        painter.setBrush(color)
        painter.drawEllipse(self.current_pos, 12 if self.swipe_flash > 0 else 10, 12 if self.swipe_flash > 0 else 10)
        
        # Recording Indicator
        if self.is_recording:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 0, 0, 200))
            painter.drawEllipse(self.screen_w - 100, 30, 15, 15)
            
            painter.setPen(QPen(QColor(255, 255, 255, 255)))
            painter.drawText(self.screen_w - 75, 43, "REC")
        
        if len(self.points) > 1:
            pen = QPen(QColor(0, 255, 0, 200), 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            
            for i in range(1, len(self.points)):
                p1, draw1 = self.points[i-1]
                p2, draw2 = self.points[i]
                
                if draw1 and draw2:
                    painter.drawLine(p1, p2)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.stop_recording_if_active()
            self.tracker.running = False
            self.tracker.wait()
            self.close()
        elif event.key() == Qt.Key_C:
            self.points = []
            self.update()
        elif event.key() == Qt.Key_R: # Toggle Recording
            if not self.is_recording:
                self.recorder = RecorderThread(self.screen_w, self.screen_h)
                self.recorder.start()
                self.is_recording = True
            else:
                self.stop_recording_if_active()
            self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MagicCanvas()
    window.show()
    sys.exit(app.exec_())
