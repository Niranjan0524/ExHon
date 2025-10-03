import dlib
import cv2
from math import hypot

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def findDist(a, b):
    return hypot(a[0]-b[0], a[1]-b[1])

def detectMouth(faces, frame):
    status = "Mouth Closed"
    for face in faces:
        lm = predictor(frame, face)
        top = (lm.part(51).x, lm.part(51).y)
        bottom = (lm.part(57).x, lm.part(57).y)
        dist = findDist(top, bottom)
        if dist > 23:
            status = "Mouth Open"
            cv2.putText(frame, status, (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            cv2.putText(frame, status, (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return status
