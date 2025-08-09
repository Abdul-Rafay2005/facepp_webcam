# Face++ Age & Emotion Detection (Webcam)

A Python project that uses the **Face++ API** to detect faces from your webcam in real-time, predict the person's **age** and **emotion**, and display them on the video feed.

This version:
- Crops faces before sending to the API (better accuracy).
- Smooths age readings by averaging the last few predictions.
- Loads API credentials securely from a `.env` file.

---

## Features
✅ Real-time face detection using OpenCV  
✅ Age & emotion detection using Face++ API  
✅ Face cropping for improved accuracy  
✅ 5-frame moving average to stabilize age results  
✅ Secure credential storage via `.env`

---

## Requirements
- Python 3.8+
- Webcam
- Face++ API account ([Sign up here](https://www.faceplusplus.com/))

  
