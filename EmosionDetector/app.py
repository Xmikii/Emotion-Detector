import os
import cv2
import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from tensorflow.keras.models import model_from_json
from flask import Flask, render_template, request

app = Flask(__name__)

# Set up face detection cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load emotion detection model
json_file = open('facial_expression_model_structure.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights('facial_expression_model_weights.h5')

# List of emotions
emotions = ('Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Biasa')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Get the uploaded image from the request
        uploaded_image = request.files['image']

        # Save the uploaded image temporarily
        temp_image_path = 'static/temp.jpg'
        uploaded_image.save(temp_image_path)

        # Detect emotion and visualize on the temporary image
        detected_emotion, detected_image = detect_emotion_and_visualize(temp_image_path)
        
        # Render the HTML template with detected emotion and image 
        return render_template('index.html', detected_emotion=detected_emotion, detected_image=detected_image)
    except Exception as e:
        return str(e)

def detect_emotion_and_visualize(image_path):
    # Load the input image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Crop and resize the detected face region
        detected_face = gray[y:y + h, x:x + w]
        detected_face = cv2.resize(detected_face, (48, 48))

        # Preprocess the face image for model input
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        # Use the emotion model to predict emotions
        predictions = emotion_model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]
        
        # Draw a rectangle around the detected face and overlay the predicted emotion label
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, emotion, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    

    # Save the processed image with emotion labels
    output_image_path = 'static/output.jpg'
    cv2.imwrite(output_image_path, img)

    # Return the detected emotion and path of the output image
    detected_image = output_image_path
    detected_emotion = emotion
    return detected_emotion, detected_image


if __name__ == '__main__':
    app.run(debug=True)
