from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import copy
import itertools
import numpy as np
import pandas as pd
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont
import io
import os
import time
from threading import Lock

# Initialize Flask app
app = Flask(__name__)

# Global variables for text accumulation
current_text = ""
prediction_history = []
text_lock = Lock()
last_prediction_time = 0
prediction_cooldown = 1.0  # seconds between adding predictions

# Load the trained model
try:
    model = keras.models.load_model("model.h5")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a dummy model for testing if the real one fails
    model = None

# MediaPipe and Drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define the Gujarati alphabet (Unicode representation)
gujarati_alphabet = [
    'અ', 'આ', 'ઇ', 'ઈ', 'ઉ', 'ઊ', 'ઋ', 'એ', 'ઐ', 'ઓ', 'ઔ',
    'ક', 'ખ', 'ગ', 'ઘ', 'ઙ', 'ચ', 'છ', 'જ', 'ઝ', 'ઞ',
    'ટ', 'ઠ', 'ડ', 'ઢ', 'ણ', 'ત', 'થ', 'દ', 'ધ', 'ન',
    'પ', 'ફ', 'બ', 'ભ', 'મ', 'ય', 'ર', 'લ', 'વ', 'શ',
    'ષ', 'સ', 'હ', 'ળ', 'ક્', 'ખ્', 'ગ્', 'ઘ્', 'ઙ્',
    'ચ્', 'છ્', 'જ્', 'ઝ્', 'ઞ્', 'ટ્', 'ઠ્', 'ડ્', 'ઢ્', 'ણ્',
    'ત્', 'થ્', 'દ્', 'ધ્', 'ન્', 'પ્', 'ફ્', 'બ્', 'ભ્', 'મ્',
    'ય્', 'ર્', 'લ્', 'વ્', 'શ્', 'ષ્', 'સ્', 'હ્', 'ળ્'
]

# Load a font that supports Gujarati script
def load_font(size=32):
    # Try multiple possible font paths
    font_paths = [
        "gujarati_font.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",  # Common Linux path
        "/System/Library/Fonts/Symbol.ttf",  # macOS
        "C:/Windows/Fonts/arial.ttf"  # Windows
    ]
    
    for font_path in font_paths:
        if os.path.isfile(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue
    
    print("No suitable font found, using default")
    return ImageFont.load_default()

# Functions for processing landmarks
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

# Generate video frames for live prediction
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        # Return a blank frame if camera can't be opened
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Camera not available", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', blank_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    font = load_font(32)
    small_font = load_font(20)
    
    # For prediction stabilization
    prediction_buffer = []
    buffer_size = 5
    current_stable_prediction = ""
    
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera")
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            debug_image = copy.deepcopy(image)
            
            # Create a black rectangle at the top for text display
            cv2.rectangle(image, (0, 0), (image.shape[1], 70), (0, 0, 0), -1)
            
            current_prediction = ""
            confidence = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    # Ensure we have the right number of landmarks
                    if len(pre_processed_landmark_list) == 42:  # 21 landmarks * 2 (x,y)
                        df = pd.DataFrame([pre_processed_landmark_list])

                        # Predict the sign language using the trained model
                        if model is not None:
                            try:
                                predictions = model.predict(df, verbose=0)
                                predicted_classes = np.argmax(predictions, axis=1)
                                confidence = np.max(predictions)
                                if predicted_classes[0] < len(gujarati_alphabet):
                                    current_prediction = gujarati_alphabet[predicted_classes[0]]
                                else:
                                    current_prediction = "?"
                            except Exception as e:
                                print(f"Prediction error: {e}")
                                current_prediction = "અ"
                                confidence = 0.8
                        else:
                            # Demo mode with dummy predictions
                            current_prediction = "અ"
                            confidence = 0.8

                        # Stabilize predictions using a buffer
                        prediction_buffer.append(current_prediction)
                        if len(prediction_buffer) > buffer_size:
                            prediction_buffer.pop(0)
                        
                        # Get the most frequent prediction in the buffer
                        if prediction_buffer:
                            stable_pred = max(set(prediction_buffer), key=prediction_buffer.count)
                            if stable_pred != current_stable_prediction:
                                current_stable_prediction = stable_pred
                        
                        # Draw the hand landmarks on the image
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
            
            # Convert OpenCV image to PIL for text rendering
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Display current prediction and confidence
            if current_stable_prediction:
                draw.text((50, 10), f"હાથની ભાષા: {current_stable_prediction}", font=small_font, fill=(255, 255, 255))
                draw.text((50, 40), f"વિશ્વાસ: {confidence:.2f}", font=small_font, fill=(255, 255, 255))
            
            # Convert PIL image back to OpenCV format
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode('.jpg', image)
            if not ret:
                print("Failed to encode frame")
                continue
                
            image_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')
    
    cap.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    with text_lock:
        return jsonify({'text': current_text, 'history': prediction_history})

@app.route('/add_char', methods=['POST'])
def add_char():
    global current_text, last_prediction_time
    char = request.json.get('char', '')
    
    with text_lock:
        current_time = time.time()
        # Prevent adding characters too frequently
        if current_time - last_prediction_time > prediction_cooldown:
            current_text += char
            prediction_history.append(char)
            last_prediction_time = current_time
    
    return jsonify({'status': 'success', 'text': current_text})

@app.route('/add_space')
def add_space():
    global current_text
    with text_lock:
        current_text += ' '
        prediction_history.append(' ')
    return jsonify({'status': 'success', 'text': current_text})

@app.route('/backspace')
def backspace():
    global current_text
    with text_lock:
        if current_text:
            current_text = current_text[:-1]
            if prediction_history:
                prediction_history.pop()
    return jsonify({'status': 'success', 'text': current_text})

@app.route('/clear_text')
def clear_text():
    global current_text
    with text_lock:
        current_text = ""
        prediction_history.clear()
    return jsonify({'status': 'success', 'text': current_text})

if __name__ == '__main__':
    # Use PORT environment variable if available (for Render)
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
