# Slouchometer 🧍‍♂️➡️🪑

A real-time **posture correction system** using **MediaPipe** and **OpenCV**, built to help users improve their sitting posture and reduce health issues caused by prolonged slouching.

## 🧠 Project Overview

Slouchometer is a Python-based application that:
- Tracks a user's posture using their **webcam**
- Uses **MediaPipe's Pose estimation** to detect key body landmarks
- Calculates the **angle of the back and neck**
- Displays an **alert popup** when poor posture is detected
- Minimizes popup frequency to avoid spam

This tool is ideal for remote workers, students, or anyone spending long hours at a desk!

---

## ⚙️ Features

- 🧍‍♀️ Real-time body tracking via webcam
- 📐 Posture angle analysis
- 🔔 Gentle desktop alerts on slouch detection
- ⏱️ Cooldown mechanism to prevent repetitive alerts
- 💻 Lightweight and easy to run

---

## 🛠️ Tech Stack

- **Python 3**
- [**MediaPipe**](https://google.github.io/mediapipe/) (for pose estimation)
- **OpenCV** (for webcam and image processing)
- **Tkinter** (for alert popup UI)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone git@github.com:Vaibhav-Krishna-S/Posture-Correction-using-Mediapipe.git
cd Posture-Correction-using-Mediapipe
pip install -r requirements.txt
pip install mediapipe opencv-python
python slouchometer.py
