from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import mediapipe as mp

app = Flask(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_fingers(frame):
    finger_count = 0
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

                # Thumb logic
                if landmarks[4].x < landmarks[3].x:
                    finger_count += 1

                # Other fingers logic
                for tip in [8, 12, 16, 20]:
                    if landmarks[tip].y < landmarks[tip - 2].y:
                        finger_count += 1

    return finger_count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    frame_data = data['frame']
    frame_bytes = base64.b64decode(frame_data.split(',')[1])
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect fingers
    finger_count = detect_fingers(frame)
    return jsonify({"finger_count": finger_count})

if __name__ == '__main__':
    app.run(debug=True)
