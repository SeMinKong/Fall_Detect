import mediapipe as mp
import numpy as np
import cv2

# Initialize Mediapipe pose estimation globally for reuse
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Mediapipe Pose 객체를 외부에서 초기화
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

def mediapipe_pose_detection(crop_img_rgb, fall_velocity_threshold, counter):
    results = pose.process(crop_img_rgb)
    fall_detected = False
    recovery_detected = False

    if results.pose_landmarks:
        # Draw the pose landmarks on the cropped image
        mp_drawing.draw_landmarks(
            crop_img_rgb,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        # Update fall detection logic based on landmarks
        landmarks = results.pose_landmarks.landmark

        # Extract relevant landmarks for fall detection
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_foot = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
        right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]

        # Calculate average positions for hips and feet
        hip_avg_y = (left_hip.y + right_hip.y) / 2
        foot_avg_y = (left_foot.y + right_foot.y) / 2

        # Check if the hips are close to the feet (possible fall)
        if hip_avg_y > (foot_avg_y - 0.2):
            counter += 1
            if counter >= 5:  # Fall detection threshold (number of consecutive frames)
                fall_detected = True
                counter = 0
        else:
            counter = 0

        # Check if recovery is detected (hips sufficiently above feet)
        if hip_avg_y < (foot_avg_y - 0.05):  # Adjust 0.05 as needed for sensitivity
            recovery_detected = True

    # Convert the cropped image back to BGR for further processing
    crop_img_bgr = cv2.cvtColor(crop_img_rgb, cv2.COLOR_RGB2BGR)

    return fall_detected, crop_img_bgr, counter, recovery_detected

