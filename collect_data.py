import cv2
import pickle
import numpy as np
import os
import face_recognition

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
labels = []

i = 0

name = input("Enter Your Name: ")
if 'names_to_ids.pkl' not in os.listdir('data/'):
    names_to_ids = {}
    current_id = 0
else:
    with open('data/names_to_ids.pkl', 'rb') as f:
        names_to_ids = pickle.load(f)
    current_id = max(names_to_ids.values()) + 1

if name not in names_to_ids:
    names_to_ids[name] = current_id
label_id = names_to_ids[name]

while True:
    ret, frame = video.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
            labels.append(label_id)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break
video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

data = {'faces': faces_data, 'labels': labels}

if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(data, f)
else:
    with open('data/face_data.pkl', 'rb') as f:
        old_data = pickle.load(f)
    old_data['faces'] = np.concatenate((old_data['faces'], faces_data), axis=0)
    old_data['labels'] += labels
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(old_data, f)

with open('data/names_to_ids.pkl', 'wb') as f:
    pickle.dump(names_to_ids, f)

# Train a face recognition model
face_recognition_model = face_recognition.FaceRecognition('data/face_data.pkl')

while True:
    ret, frame = video.read() 
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
    face_locations = face_recognition_model.detect_faces(rgb_frame)
    face_names = []
    for face_location in face_locations:
        face_encoding = face_recognition_model.face_encodings(rgb_frame, [face_location])[0]
        matches = face_recognition.compare_faces(face_recognition_model.known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = list(face_recognition_model.known_face_names)[match_index]
        face_names.append(name)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()