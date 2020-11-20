import cv2
import  numpy as np

def Nothing(x):
    pass

cap = cv2.VideoCapture(1)

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detect = cv2.CascadeClassifier('haarcascade_eye.xml')

cv2.namedWindow('frame')
cv2.createTrackbar('neighbour', 'frame', 5, 20, Nothing)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    neighbour = cv2.getTrackbarPos('neighbour', 'frame')

    faces = face_detect.detectMultiScale(gray, 2, neighbour)
    for rectangle in faces:
        (x, y, w, h) = rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_colour = frame[y:y+h, x:x+w]

        eye = eye_detect.detectMultiScale(roi_gray)
        for rectangles in eye:
            (ex, ey, ew, eh) = rectangles
            cv2.rectangle(roi_colour, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
