import cv2
import dlib
from blink import detectBlink
from mouth import detectMouth
from head_pose import detectHeadPose
from eye_gaze import eyeGaze

face_detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    detectBlink(faces, frame)
    detectMouth(faces, frame)
    detectHeadPose(faces, frame)
    eyeGaze(faces, frame)

    cv2.imshow("Exam Proctoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
