# Saudi National Day 96 AR Filter

A real-time Augmented Reality (AR) face filter application built with Computer Vision to overlay 96th Saudi National Day themed accessories and stickers onto the user's face via webcam. Developed as part of the projects and initiatives at **Fab Lab Hail**, University of Hail.

## Features
* **Real-Time Face Mesh Tracking:** Utilizes MediaPipe Face Mesh to detect key facial landmarks (eyes, nose, forehead, and ears) and calculate head tilt angles for dynamic scaling and rotation.
* **Head-Tilt Gesture Switching:** Automatically cycles through the 4 filter modes when tilting the head left or right beyond 15 degrees (with a 2-second cooldown).
* **Manual Keyboard Controls:** Instant filter switching using keys `1` through `4`, and `q` to quit the application.
* **4 Filter Modes:**
  1. **Mode 1:** Saudi National Day 96 Glasses.
  2. **Mode 2:** National Day Cap + Cheek Stickers.
  3. **Mode 3:** Cheek Stickers Only.
  4. **Mode 4:** National Day Cap Only.
* **Full HD Overlay Frame:** Integrates a custom 1920x1080 celebratory frame ("عزّنا بطبعنا | SEPT 23").

## Installation & Usage

1. **Install the required dependencies:**
   ```bash
   pip install opencv-python mediapipe numpy
   ```

2. **Project Structure:**
   Ensure all asset files are located in the same directory as `main.py`:
   * `main.py`
   * `frame.png`
   * `glasses.png`
   * `hat.png`
   * `sticker.png`

3. **Run the application:**
   ```bash
   python main.py
   ```
   *(Note: The camera index is set to `cv2.VideoCapture(2)` by default for an external webcam. Change it to `0` if you are using a built-in laptop camera).*

---

## Author
* **Talal Qassem Al-Qassem** — Head of Projects & Innovation Team (1st Semester 2026), Fab Lab Hail (University of Hail).
