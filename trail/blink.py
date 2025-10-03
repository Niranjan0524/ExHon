import dlib
import cv2
from math import hypot

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midPoint(p1, p2):
    return ((p1.x + p2.x)//2, (p1.y + p2.y)//2)

def findDist(a, b):
    return hypot(a[0]-b[0], a[1]-b[1])

def detectBlink(faces, frame):
    blinked = False
    for face in faces:
        lm = predictor(frame, face)
        # left eye
        l_ratio = findDist((lm.part(36).x,lm.part(36).y), (lm.part(39).x,lm.part(39).y)) / \
                  (findDist(midPoint(lm.part(37), lm.part(38)), midPoint(lm.part(40), lm.part(41))) + 1e-6)
        # right eye
        r_ratio = findDist((lm.part(42).x,lm.part(42).y), (lm.part(45).x,lm.part(45).y)) / \
                  (findDist(midPoint(lm.part(43), lm.part(44)), midPoint(lm.part(46), lm.part(47))) + 1e-6)
        if l_ratio >= 3.6 or r_ratio >= 3.6:
            blinked = True
            cv2.putText(frame, "BLINK", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
    return blinked
