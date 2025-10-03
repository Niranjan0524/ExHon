import dlib
import cv2
import numpy as np
import math

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
])

def detectHeadPose(faces, frame):
    size = frame.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length,0,center[0]], [0,focal_length,center[1]], [0,0,1]], dtype="double")

    status = "Center"
    for face in faces:
        lm = predictor(frame, face)
        image_points = np.array([
            [lm.part(30).x, lm.part(30).y],  # Nose tip
            [lm.part(8).x, lm.part(8).y],    # Chin
            [lm.part(36).x, lm.part(36).y],  # Left eye
            [lm.part(45).x, lm.part(45).y],  # Right eye
            [lm.part(48).x, lm.part(48).y],  # Left mouth
            [lm.part(54).x, lm.part(54).y]   # Right mouth
        ], dtype="double")

        dist_coeffs = np.zeros((4,1))
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        nose_end = np.array([(0.0,0.0,1000.0)])
        nose_proj, _ = cv2.projectPoints(nose_end, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_proj[0][0][0]), int(nose_proj[0][0][1]))

        try:
            m = (p2[1]-p1[1])/(p2[0]-p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            ang1 = 0
        if ang1 > 20:
            status = "Head Up"
        elif ang1 < -20:
            status = "Head Down"
        elif p2[0] > p1[0]+15:
            status = "Head Right"
        elif p2[0] < p1[0]-15:
            status = "Head Left"
        cv2.putText(frame, status, (50,110), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
    return status
