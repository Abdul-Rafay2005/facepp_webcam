
import os
import cv2
import requests
import time
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read API credentials from .env
API_KEY = os.getenv("FACEPP_API_KEY")
API_SECRET = os.getenv("FACEPP_API_SECRET")

FACEPP_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"

params = {
    "api_key": API_KEY,
    "api_secret": API_SECRET,
    "return_attributes": "age,emotion"
}


# Function to call Face++ API
def detect_face_facepp(face_img):
    _, img_data = cv2.imencode(".jpg", face_img)
    files = {"image_file": img_data.tobytes()}
    try:
        response = requests.post(FACEPP_URL, data=params, files=files)
        if response.status_code == 200:
            return response.json()
        else:
            print("API Error:", response.status_code, response.text)
            return None
    except Exception as e:
        print("Error calling API:", e)
        return None

# Age history for smoothing
age_history = []

# Open webcam
cap = cv2.VideoCapture(0)
last_request_time = 0
cooldown = 2  # seconds between API calls
latest_results = []

# Use OpenCV Haar cascade to find faces locally before sending
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_local = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Call API only every cooldown seconds
    if time.time() - last_request_time > cooldown and len(faces_local) > 0:
        for (x, y, w, h) in faces_local:
            # Crop face from frame
            face_crop = frame[y:y+h, x:x+w]
            results = detect_face_facepp(face_crop)
            if results and "faces" in results and len(results["faces"]) > 0:
                face_data = results["faces"][0]
                attrs = face_data["attributes"]
                age = attrs["age"]["value"]
                emotion_dict = attrs["emotion"]
                main_emotion = max(emotion_dict, key=emotion_dict.get)

                # Add to history for smoothing
                age_history.append(age)
                if len(age_history) > 5:  # keep only last 5 readings
                    age_history.pop(0)
                avg_age = round(np.mean(age_history))

                latest_results = [(x, y, w, h, avg_age, main_emotion)]
        last_request_time = time.time()

    # Draw results
    for (x, y, w, h, avg_age, main_emotion) in latest_results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {avg_age}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Emotion: {main_emotion}", (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face++ Age & Emotion Detector (Smoothed)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
