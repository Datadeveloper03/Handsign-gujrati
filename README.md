
## Indian Sign Language to Gujrati Translation

This project is a Flask web application that uses a pre-trained machine learning model to detect Indian Sign Language (ISL) gestures in real-time using a webcam feed and convert them to Gujrati text. The application uses OpenCV, MediaPipe, and TensorFlow for video processing and prediction.

### Features
1. **Real-Time ISL Detection**: Detects ISL gestures from the webcam feed.
2. **Hand Landmark Detection**: Uses MediaPipe to detect hand landmarks for sign recognition.
3. **Flask Web Application**: Runs as a Flask web app that streams the webcam feed and displays predictions.

### Installation

1. **Clone the Repository**

2. **Create a Virtual Environment (optional but recommended)**

It's best practice to create a virtual environment to keep your dependencies isolated.
```
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```
3. **Install Dependencies**
Install the required dependencies from the requirements.txt file using pip.
```
pip install -r requirements.txt
```
These dependencies include:  

* Flask for building the web application

* OpenCV for video capture and processing

* MediaPipe for hand landmark detection

* TensorFlow for loading the pre-trained model

* Pillow for rendering Gujarati text on the video

### Run the Application
Start the Flask application by running:
```
python isl_to_gujrati.py
```
This will start the application on http://127.0.0.1:8000/.

### Access the Application

Open your web browser and go to http://127.0.0.1:8000/ to see the live webcam feed and ISL to Gujrati translation.
