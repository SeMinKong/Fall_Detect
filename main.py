from Yolo_Detect import yolo_detect
from Mediapipe_Pose import mediapipe_pose_detection
import cv2
import numpy as np

# Main loop to process each frame from the video file
video_path = "FD_In_H11H21H31_0009_20201229_14.mp4"  # Replace with the actual path to your video file
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Define a fixed position for the overlay text
text_position = (10, 50)  # Adjust the coordinates as needed
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize variables and parameters for fall detection
counter = 0
fall_detected_flag = False  # Keep track of fall detection status
fall_persistent = False  # Flag to maintain fall status
recovery_counter = 0  # Counter to track recovery
RECOVERY_THRESHOLD = 10  # Number of frames needed for recovery

# Main loop to process each frame from the video file
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print("End of video file reached or error reading the frame.")
        break

    # Resize the frame to reduce processing load (optional, can adjust size as needed)
    img = cv2.resize(img, (640, 360))

    # Perform object detection using YOLO
    boxes = yolo_detect(img, confidence_threshold=0.4)  # Add confidence threshold parameter

    for box in boxes:
        x1, y1, x2, y2 = box['coords']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green rectangle
        crop_img = img[y1:y2, x1:x2]

        # Skip invalid cropping
        if crop_img.size == 0:
            continue

        # Convert the cropped image to RGB for Mediapipe
        crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        # Perform pose estimation using Mediapipe
        fall_detected, crop_img, counter, recovery_detected = mediapipe_pose_detection(crop_img_rgb, 0.5, counter)

        # Update fall detected flag if a fall is detected in any person
        if fall_detected:
            fall_detected_flag = True
            fall_persistent = True  # Maintain the fall status until reset
            recovery_counter = 0  # Reset recovery counter when a fall is detected

        # Increment recovery counter if recovery is detected
        if recovery_detected:
            recovery_counter += 1
            if recovery_counter >= RECOVERY_THRESHOLD:
                fall_persistent = False  # Reset fall status when recovery is consistent
                recovery_counter = 0  # Reset recovery counter

        # Replace the cropped image back into the original image
        img[y1:y2, x1:x2] = crop_img

    # If a fall is detected, maintain the status until the user stands up
    if fall_persistent:
        cv2.putText(img, 'Fall Detected!', text_position, font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        # Display "No Fall Detected" text only if no fall is detected in any person
        cv2.putText(img, 'No Fall Detected', text_position, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Reset fall_detected_flag for the next frame
    fall_detected_flag = False

    # Show the frame
    cv2.imshow('Fall Detection Feed', img)

    # Wait for 30 ms between frames, and break if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

