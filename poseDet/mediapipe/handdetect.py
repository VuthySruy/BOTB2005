import mediapipe as mp 

import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
        ) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("ignorinig empty camera frame")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow("md pose", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
