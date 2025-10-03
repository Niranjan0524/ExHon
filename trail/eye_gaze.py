import dlib
import cv2
import numpy as np

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_detector = dlib.get_frontal_face_detector()

def eyeGaze(faces, frame):
    gaze_status = "Center"
    for face in faces:
        lm = predictor(frame, face)
        # left eye
        left_eye = np.array([(lm.part(i).x, lm.part(i).y) for i in range(36,42)])
        right_eye = np.array([(lm.part(i).x, lm.part(i).y) for i in range(42,48)])

        # simple ratio of white pixels left/right
        def ratio(eye):
            min_x, max_x = np.min(eye[:,0]), np.max(eye[:,0])
            min_y, max_y = np.min(eye[:,1]), np.max(eye[:,1])
            eye_frame = frame[min_y:max_y, min_x:max_x]
            gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
            left = np.sum(th[:, :th.shape[1]//2] == 255)
            right = np.sum(th[:, th.shape[1]//2:] == 255)
            if left>right*1.2:
                return "Right"
            elif right>left*1.2:
                return "Left"
            else:
                return "Center"

        gaze_left = ratio(left_eye)
        gaze_right = ratio(right_eye)
        if gaze_left == "Left" or gaze_right == "Left":
            gaze_status = "Looking Left"
        elif gaze_left == "Right" or gaze_right == "Right":
            gaze_status = "Looking Right"
        cv2.putText(frame, gaze_status, (50,140), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
    return gaze_status
