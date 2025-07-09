from flask import Flask, send_from_directory, send_file, Response
import webbrowser
from threading import Timer
import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

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

# Load the model
model = tf.keras.models.load_model('yoga_poses_model.h5')

# Video capture variable
cap = None

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def validate_pose(landmarks, pose_name):
    points = {}
    points['left_shoulder'] = [landmarks[11].x, landmarks[11].y]
    points['right_shoulder'] = [landmarks[12].x, landmarks[12].y]
    points['left_hip'] = [landmarks[23].x, landmarks[23].y]
    points['right_hip'] = [landmarks[24].x, landmarks[24].y]
    points['left_knee'] = [landmarks[25].x, landmarks[25].y]
    points['right_knee'] = [landmarks[26].x, landmarks[26].y]
    points['left_ankle'] = [landmarks[27].x, landmarks[27].y]
    points['right_ankle'] = [landmarks[28].x, landmarks[28].y]

    pose_criteria = {
        'noPose': {
            'spine_alignment': (160, 200),
            'shoulder_level': (170, 190),
            'hip_level': (170, 190),
            'knee_extension': (160, 200)
        },
        'tree': {
            'standing_leg': (170, 190),
            'hip_alignment': (80, 100),
            'raised_knee': (35, 90)
        },
        'warrior': {
            'front_knee': (80, 100),
            'back_leg': (150, 170),
            'hip_alignment': (170, 190)
        },
        'downdog': {
            'hip_angle': (50, 80),
            'knee_extension': (160, 180),
            'shoulder_alignment': (170, 190)
        }
    }

    if pose_name not in pose_criteria:
        return None

    feedback = []
    score = 0
    criteria = pose_criteria[pose_name]

    if pose_name == 'noPose':
        spine = calculate_angle(points['right_shoulder'], points['right_hip'], points['right_ankle'])
        shoulder_level = calculate_angle(points['left_shoulder'], points['right_shoulder'], points['right_hip'])
        hip_level = calculate_angle(points['left_hip'], points['right_hip'], points['right_shoulder'])
        knee_straight = calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle'])
        
        if not (criteria['spine_alignment'][0] <= spine <= criteria['spine_alignment'][1]):
            feedback.append("Stand straight")
        else:
            score += 1
            
        if not (criteria['shoulder_level'][0] <= shoulder_level <= criteria['shoulder_level'][1]):
            feedback.append("Level your shoulders")
        else:
            score += 1
            
        if not (criteria['hip_level'][0] <= hip_level <= criteria['hip_level'][1]):
            feedback.append("Level your hips")
        else:
            score += 1
            
        if not (criteria['knee_extension'][0] <= knee_straight <= criteria['knee_extension'][1]):
            feedback.append("Keep your legs straight")
        else:
            score += 1

    elif pose_name == 'tree':
        standing_leg = calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle'])
        hip_level = calculate_angle(points['left_hip'], points['right_hip'], points['right_shoulder'])
        knee_bend = calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle'])
        
        if not (criteria['standing_leg'][0] <= standing_leg <= criteria['standing_leg'][1]):
            feedback.append("Straighten your standing leg")
        else:
            score += 1
            
        if not (criteria['hip_alignment'][0] <= hip_level <= criteria['hip_alignment'][1]):
            feedback.append("Level your hips")
        else:
            score += 1
            
        if not (criteria['raised_knee'][0] <= knee_bend <= criteria['raised_knee'][1]):
            feedback.append("Adjust your raised knee")
        else:
            score += 1

    accuracy = (score / len(criteria)) * 100
    return {
        'accuracy': accuracy,
        'feedback': feedback
    }

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    normalized_frame = rgb_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame

def get_pose_name(prediction):
    pose_classes = ['tree', 'warrior', 'downdog', 'noPose']
    return pose_classes[np.argmax(prediction)]

def generate_frames():
    global cap
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(image_rgb)

        input_frame = preprocess_frame(frame)
        prediction = model.predict(input_frame)
        pose_name = get_pose_name(prediction[0])
        
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            validation_result = validate_pose(result.pose_landmarks.landmark, pose_name)
            
            if validation_result:
                accuracy = validation_result['accuracy']
                
                if accuracy > 80:
                    color = (0, 255, 0)  # Green
                    status = "CORRECT"
                elif accuracy > 60:
                    color = (0, 165, 255)  # Orange
                    status = "ADJUST POSE"
                else:
                    color = (0, 0, 255)  # Red
                    status = "INCORRECT"
                
                cv2.rectangle(frame, (10, 10), (200, 90), color, -1)
                cv2.putText(frame, status, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                y_offset = 120
                for feedback in validation_result['feedback']:
                    cv2.putText(frame, f"• {feedback}", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 30

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