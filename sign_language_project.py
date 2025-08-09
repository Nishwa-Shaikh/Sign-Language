import cv2 as cv
import mediapipe as mp 

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
