import cv2
import numpy as np
import mediapipe as mp
import winsound
import time
import threading
import tkinter as tk

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate spine angle
def calculate_spine_angle(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    shoulder_mid = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
    hip_mid = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2])

    spine_vector = shoulder_mid - hip_mid
    vertical_vector = np.array([0, -1])

    angle = np.degrees(np.arccos(np.dot(spine_vector, vertical_vector) / (np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))))
    return angle

# Function to calculate head position change
def calculate_head_position_change(landmarks, initial_head_mid):
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
    head_mid = np.array([(left_ear.x + right_ear.x) / 2, (left_ear.y + right_ear.y) / 2])
    return initial_head_mid[1] - head_mid[1]

# Function to calculate shoulder breadth change
def calculate_shoulder_breadth_change(landmarks, initial_shoulder_breadth):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    current_breadth = abs(left_shoulder.x - right_shoulder.x)
    return current_breadth - initial_shoulder_breadth

# Function to play beep sound
def play_beep():
    frequency = 900
    duration = 200
    winsound.Beep(frequency, duration)

# Non-blocking floating popup
def show_floating_popup(message):
    def popup_thread():
        root = tk.Tk()
        root.overrideredirect(True)  # Remove window border
        root.attributes("-topmost", True)  # Always on top
        root.geometry("300x60+100+100")  # Width x Height + X + Y

        label = tk.Label(root, text=message, bg="black", fg="white", font=("Arial", 14), padx=20, pady=10)
        label.pack()

        # Auto-close after 3 seconds
        root.after(3000, root.destroy)
        root.mainloop()
    
    threading.Thread(target=popup_thread, daemon=True).start()

# Trigger both beep and floating popup
def trigger_alert(message):
    play_beep()
    show_floating_popup(message)

# Main posture detection function
def main():
    cap = cv2.VideoCapture(0)

    print("Calibrating... Please sit upright in a neutral posture.")
    calibration_frames = 100
    calibration_spine_angles = []
    calibration_head_positions = []
    calibration_shoulder_breadths = []

    while len(calibration_spine_angles) < calibration_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            spine_angle = calculate_spine_angle(landmarks)
            calibration_spine_angles.append(spine_angle)

            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
            head_mid = np.array([(left_ear.x + right_ear.x) / 2, (left_ear.y + right_ear.y) / 2])
            calibration_head_positions.append(head_mid)

            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_breadth = abs(left_shoulder.x - right_shoulder.x)
            calibration_shoulder_breadths.append(shoulder_breadth)

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if calibration_spine_angles and calibration_head_positions and calibration_shoulder_breadths:
        spine_angle_threshold = np.mean(calibration_spine_angles) + 10
        initial_head_mid = np.mean(calibration_head_positions, axis=0)
        initial_shoulder_breadth = np.mean(calibration_shoulder_breadths)
        head_position_threshold = -0.04
        shoulder_breadth_threshold = -0.04
        print("Calibration complete.")

    # Cooldown control
    cooldown_seconds = 5
    last_alert_time = {"spine": 0, "head": 0, "shoulders": 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            spine_angle = calculate_spine_angle(landmarks)
            head_position_change = calculate_head_position_change(landmarks, initial_head_mid)
            shoulder_breadth_change = calculate_shoulder_breadth_change(landmarks, initial_shoulder_breadth)

            # Overlay text
            cv2.putText(frame, f"Spine Angle: {spine_angle:.2f}°", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Head Change: {head_position_change:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Shoulder Change: {shoulder_breadth_change:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            current_time = time.time()

            # Spine alert
            if spine_angle > spine_angle_threshold and current_time - last_alert_time["spine"] > cooldown_seconds:
                cv2.putText(frame, "Slouching: Spine", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                trigger_alert("⚠️ Slouching Detected: Spine")
                last_alert_time["spine"] = current_time

            # Head alert
            if head_position_change < head_position_threshold and current_time - last_alert_time["head"] > cooldown_seconds:
                cv2.putText(frame, "Slouching: Head", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                trigger_alert("⚠️ Slouching Detected: Head Forward")
                last_alert_time["head"] = current_time

            # Shoulders alert
            if shoulder_breadth_change < shoulder_breadth_threshold and current_time - last_alert_time["shoulders"] > cooldown_seconds:
                cv2.putText(frame, "Slouching: Shoulders", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                trigger_alert("⚠️ Slouching Detected: Shoulders")
                last_alert_time["shoulders"] = current_time

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Posture Correction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
