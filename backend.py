from flask import Flask, send_from_directory, send_file, Response
import webbrowser
from threading import Timer
import os
import cv2
import mediapipe as mp

app = Flask(__name__)
BASE_DIR = r"e:\Yoga-Posture-Detection"

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False, 
    model_complexity=1, 
    smooth_landmarks=True,
    enable_segmentation=False, 
    refine_face_landmarks=False
)
mp_drawing = mp.solutions.drawing_utils

# Video capture variable
cap = None

def generate_frames():
    global cap
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(image_rgb)

        # Draw landmarks
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if result.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if result.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Convert frame to bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    try:
        html_path = os.path.join(BASE_DIR, 'yogapose.html')
        if os.path.exists(html_path):
            return send_file(html_path)
        else:
            return f"File not found at {html_path}", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection')
def stop_detection():
    global cap
    if cap is not None:
        cap.release()
    return "Detection stopped"

@app.route('/<path:path>')
def serve_file(path):
    try:
        return send_from_directory(BASE_DIR, path)
    except Exception as e:
        return f"Error serving {path}: {str(e)}", 404

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    print(f"Looking for files in: {BASE_DIR}")
    print(f"Files in directory: {os.listdir(BASE_DIR)}")
    Timer(1, open_browser).start()
    app.run(port=5000, debug=False)