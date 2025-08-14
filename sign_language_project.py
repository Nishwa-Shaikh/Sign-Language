from google.colab import files
ZIPPED = files.upload()
import zipfile
zip_path = ("/content/archive.zip")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("unzipped_folder")
print("Unzipped successfully!")

import tensorflow as TF
import numpy as np
import cv2 as cv
import mediapipe as mp 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# Prepare training and validation data generators from folders
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    (r'C:/Users/PMLS/OneDrive/Desktop/Face detector/destination_folder/asl_alphabet_train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    (r'C:/Users/PMLS/OneDrive/Desktop/Face detector/destination_folder/asl_alphabet_train'),   # Same folder with subset validation
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the trained model
model.save('hand_sign_model.h5')

Hand_detect = mp.solutions.drawing_utils
style = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands 

cap = cv.VideoCapture(0)
hands = mp_hands.Hands()

cascade_path = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print('No face detected')

while True:
    frame, ret = cap.read()
    if not ret:
        print("Camera is not working")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(RGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            Hand_detect.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)
        
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv.imshow('face detector', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
