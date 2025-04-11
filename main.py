import cv2
import numpy as np
import mediapipe as mp
from playsound import playsound
import threading
import time  # For cooldown tracking

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Cooldown settings
COOLDOWN_TIME = 2  # Time interval between beeps in seconds
last_beep_time = 0  # Track the last time the beep was played

def calculate_spine_angle(landmarks):
    """
    Calculate the spine angle using shoulder and hip midpoints.
    """
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Calculate midpoints
    shoulder_mid = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
    hip_mid = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2])

    # Calculate spine vector (from hip midpoint to shoulder midpoint)
    spine_vector = shoulder_mid - hip_mid
    vertical_vector = np.array([0, -1])  # Vertical line (upwards)

    # Calculate the angle between spine vector and vertical axis
    angle = np.degrees(np.arccos(np.dot(spine_vector, vertical_vector) / (np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))))
    return angle

def calculate_head_position_change(landmarks, initial_head_mid):
    """
    Calculate the vertical change in the head position.
    """
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

    # Calculate head midpoint
    head_mid = np.array([(left_ear.x + right_ear.x) / 2, (left_ear.y + right_ear.y) / 2])

    # Calculate vertical change in the head position
    vertical_change = initial_head_mid[1] - head_mid[1]
    return vertical_change

def calculate_shoulder_breadth_change(landmarks, initial_shoulder_breadth):
    """
    Calculate the change in shoulder breadth.
    """
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate current shoulder breadth
    current_shoulder_breadth = abs(left_shoulder.x - right_shoulder.x)

    # Calculate change in shoulder breadth
    breadth_change = current_shoulder_breadth - initial_shoulder_breadth
    return breadth_change

def play_beep():
    """
    Play a custom beep sound asynchronously to avoid blocking the main thread.
    """
    threading.Thread(target=playsound, args=("beep.mp3",), daemon=True).start()

def main():
    global last_beep_time

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Calibration phase
    print("Calibrating... Please sit in a neutral upright posture.")
    calibration_frames = 100  # Number of frames to use for calibration
    calibration_spine_angles = []
    calibration_head_positions = []
    calibration_shoulder_breadths = []

    while len(calibration_spine_angles) < calibration_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark

            # Calculate spine angle
            spine_angle = calculate_spine_angle(landmarks)
            calibration_spine_angles.append(spine_angle)

            # Calculate head position
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
            head_mid = np.array([(left_ear.x + right_ear.x) / 2, (left_ear.y + right_ear.y) / 2])
            calibration_head_positions.append(head_mid)

            # Calculate shoulder breadth
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_breadth = abs(left_shoulder.x - right_shoulder.x)
            calibration_shoulder_breadths.append(shoulder_breadth)

        # Show frame
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate thresholds
    if calibration_spine_angles and calibration_head_positions and calibration_shoulder_breadths:
        spine_angle_threshold = np.mean(calibration_spine_angles) + 10  # Add buffer for slouching
        initial_head_mid = np.mean(calibration_head_positions, axis=0)  # Average head position
        initial_shoulder_breadth = np.mean(calibration_shoulder_breadths)  # Average shoulder breadth
        print(f"Calibration complete. Spine angle threshold: {spine_angle_threshold:.2f}°, Head position threshold: -0.01, Shoulder breadth threshold: -0.02")

    # Main loop for posture detection
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark

            # Calculate spine angle
            spine_angle = calculate_spine_angle(landmarks)

            # Calculate head position change
            head_position_change = calculate_head_position_change(landmarks, initial_head_mid)

            # Calculate shoulder breadth change
            shoulder_breadth_change = calculate_shoulder_breadth_change(landmarks, initial_shoulder_breadth)

            # Display information on the screen
            cv2.putText(frame, f"Spine Angle: {spine_angle:.2f}°", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Head Position Change: {head_position_change:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Shoulder Breadth Change: {shoulder_breadth_change:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Check for slouching
            if spine_angle > spine_angle_threshold or head_position_change < -0.01 or shoulder_breadth_change < -0.02:
                    cv2.putText(frame, "Slouching Detected!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    play_beep()
                    

                # Display warning
            

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show frame
        cv2.imshow("Posture Correction", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
