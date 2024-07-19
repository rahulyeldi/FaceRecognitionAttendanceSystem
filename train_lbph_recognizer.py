import cv2
import numpy as np
import os
import pickle

# Load collected face data and names
with open('data/faces_data.pkl', 'rb') as f:
    faces_data = pickle.load(f)
with open('data/labels.pkl', 'rb') as f:
    labels_data = pickle.load(f)

# Reshape faces data to its original 2D shape
faces = [face.reshape(250, 30) for face in faces_data]

# Create an LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer with the face data and corresponding labels
recognizer.train(faces, np.array(labels_data))

# Create the 'trainer' directory if it doesn't exist
if not os.path.exists('trainer'):
    os.makedirs('trainer')

# Save the trained model
recognizer.save('trainer/trainer.yml')

print("Model trained and saved successfully.")