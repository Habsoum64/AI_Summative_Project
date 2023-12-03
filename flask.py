from flask import Flask, render_template, request
import cv2
import numpy as np


# app.py
class MyDetector:
    def __init__(self):
        self.app = Flask(my_detector)

from my_detector import MyDetector

detector = MyDetector()
app = detector.app

if __name__ == '__main__':
    app.run(debug=True)

# Placeholder for object detection function
def perform_object_detection(image):
    # Replace this function with your actual object detection logic
    # The function should take an image as input and return the detection results
    return [("Object1", (10, 20, 30, 40)), ("Object2", (50, 60, 70, 80))]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('Demo.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('Demo.html', error='No selected file')

        # Read the image
        image_np = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Perform object detection
        detection_results = perform_object_detection(image_np)

        return render_template('Demo.html', image=image_np, results=detection_results)

    return render_template('Demo.html')

if __name__ == '__main__':
    app.run(debug=True)
