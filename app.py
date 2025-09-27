from flask import Flask, render_template, request, redirect, url_for, Response
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image
import cv2

# -------------------------------------
# Initialize Flask app
# -------------------------------------
app = Flask(__name__)

# Load your trained model
model = load_model("face_mask_detector_model.h5")

# Folder to store uploaded images
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------------
# Routes
# -------------------------------------

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Upload Image & Predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        img = Image.open(file_path).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            label = "Without Mask 😷❌"
            color = "red"
        else:
            label = "With Mask 😷✅"
            color = "green"

        return render_template('result.html', label=label, color=color, image_path=file_path)

    return redirect(request.url)

# -------------------------------------
# Live Detection Routes
# -------------------------------------

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/live')
def live():
    return render_template('live.html')

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_img, (150, 150))
                face_array = np.array(face_resized) / 255.0
                face_array = np.expand_dims(face_array, axis=0)

                pred = model.predict(face_array)[0][0]
                label = "Mask" if pred < 0.5 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------------------------
# Run App
# -------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
