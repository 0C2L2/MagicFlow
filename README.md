# MagicFlow: The Gesture-Based Teaching Studio

MagicFlow is a state-of-the-art, "always-on-top" teaching and recording interface. It allows educators to control their computer, draw on the screen, and record professional lectures using only intuitive hand gestures via a standard webcam.

---
<img width="1006" height="734" alt="image" src="https://github.com/user-attachments/assets/293a44fc-ba57-4819-9d75-71fee2df8f6c" />


## Key Features

- **🎨 Virtual Canvas:** Draw anywhere on your screen with a simple pinch.
- **🎙️ Full Recording Suite:** Capture your screen and voice simultaneously.
- **👁️ Professor View (PiP):** A live webcam overlay in the corner so students can see you.
- **📂 Interactive Dashboard:** Manage your recordings, learn gestures, and view your shortcuts in a sleek UI.
- **📄 Slide Control:** Point left or right to change slides in your presentation naturally.
- **🔇 Discrete Gestures:** No keyboards needed—start, stop, and clear using clear hand poses.

---

## 🖐️ Gesture Guide

MagicFlow is designed to be natural. Here are the core triggers:

| Gesture | Action | Description |
| :--- | :--- | :--- |
| **🤏 Pinch** | **Draw** | Touch Index + Thumb to begin writing on the screen. |
| **✌️ Peace Sign** | **Clear** | Show a peace sign to wipe the digital canvas clean. |
| **💍 Ring Finger** | **START Rec** | Raise only your ring finger to begin recording. |
| **✋ Open Palm** | **STOP Rec** | Show an open palm (at least 4 fingers) to end recording. |
| **☝️ Pointing** | **Slides** | Point index right/left to navigate PowerPoint/PDFs. |
| **🤙 Pinky Up** | **Exit** | Raise only your pinky to close the studio session. |

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- **Python 3.10+** (Optimized for 3.13)
- **FFmpeg:** Installed and added to your system PATH (required for audio/video merging).

### 2. Install Dependencies
```bash
pip install mediapipe opencv-python numpy pyautogui PyQt5 mss pyaudio
```

### 3. Model Download
Ensure the `hand_landmarker.task` file (MediaPipe Hand Landmarker model) is in the root directory.

### 4. Run the Studio
```bash
python app.py
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| **R** | Toggle Recording (Start/Stop) |
| **C** | Clear Canvas |
| **Esc** | Exit Studio & Return to Dashboard |
| **← / →** | Manual Slide Navigation |

---

## 📁 Project Structure

- `app.py`: The main application (Launcher Dashboard + Magic Studio).
- `recordings/`: Auto-generated folder for your saved sessions.
- `venv/`: Your Python virtual environment.
- `hand_landmarker.task`: The AI model for hand tracking.

---

