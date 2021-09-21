import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

mask = cv2.imread('mask.png')
mask_height, mask_width, _ = mask.shape

print('mask_dimensions (HxW):', mask_height, 'x', mask_width)
print('frame_dimensions (HxW):', int(frame_height), 'x', int(frame_width))


def detect_face(feed):
    face_cascade = cv2.CascadeClassifier('haar-cascade-front-of-face.xml')
    gray = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # cv2.rectangle(feed, (x, y), (x+w, y+h), (255, 0, 0), 2)
        feed[y:y+mask_height, x:x+mask_width] = mask


while True:
    _, frame = cap.read()

    detect_face(frame)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
