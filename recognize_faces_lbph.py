import cv2
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
import pickle

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Load the trained LBPH recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load the names to IDs mapping
with open('data/names_to_ids.pkl', 'rb') as f:
    names_to_ids = pickle.load(f)
ids_to_names = {v: k for k, v in names_to_ids.items()}

# Load face detection model
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)
imgBackground = cv2.imread("background1.png")

COL_NAMES = ['NAME', 'TIME']

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         face_img = gray[y:y+h, x:x+w]
#         id, confidence = recognizer.predict(face_img)
        
#         ts = time.time()
#         date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
#         exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        
#         if confidence < 50:
#             name = ids_to_names.get(id, "Unknown")
#             confidence_text = f"{round(100 - confidence)}%"
#             attendance = [name, str(timestamp)]
#         else:
#             name = "Unknown"
#             confidence_text = f"{round(100 - confidence)}%"
        
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
#         cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
#         cv2.putText(frame, name, (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
#     imgBackground[162:162 + 480, 55:55 + 640] = frame
#     cv2.imshow("Frame", imgBackground)
    
#     k = cv2.waitKey(1)
#     if k == ord('o'):
#         speak("Attendance Taken..")
#         time.sleep(5)
#         if exist:
#             with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(attendance)
#             csvfile.close()
#         else:
#             with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(COL_NAMES)
#                 writer.writerow(attendance)
#             csvfile.close()
#     if k == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        id, confidence = recognizer.predict(face_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")        
        # print(f"Confidence: {confidence}")  # Print confidence value
        
        if confidence < 150:  # Try increasing the threshold value
            name = ids_to_names.get(id, "Unknown")
            confidence_text = f"{round(100 - confidence)}%"
            attendance = [name, str(timestamp)]
        else:
            name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, name, (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)
    
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
