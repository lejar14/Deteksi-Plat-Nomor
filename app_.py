from flask import Flask, render_template, request, redirect, url_for
import os
import random
import string
from ultralytics import YOLO
import cv2
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/deteksi/'



def generate_random_filename():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))


def delete_previous_uploads():
    # Delete files in the static/uploads/ folder
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print("Error while deleting file:", e)

    # Delete the 'deteksi' folder along with its contents
    hasil_folder = app.config['RESULT_FOLDER']
    if os.path.exists(hasil_folder):
        try:
            shutil.rmtree(hasil_folder)
        except Exception as e:
            print("Error while deleting folder:", e)


def detect_platnomer(filepath):
    # Load a pretrained YOLOv8n model
    model = YOLO('platnomor.pt')

    # Run inference on the image
    results = model.predict(filepath, save=True, conf=0.5, project="static", name="deteksi", save_crop=True)
    return results


@app.route('/')
def index():
    delete_previous_uploads()  # Delete previous uploaded files and the 'hasil' folder
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = generate_random_filename() + '.' + file.filename.rsplit('.', 1)[1].lower()
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Check if the uploaded file is an image or video using OpenCV
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cap = cv2.VideoCapture(filepath)
        is_image = cap.isOpened()
        cap.release()

        result_file_path = None

        if is_image:
            detect_platnomer(filepath)
            result_file_path = 'static/deteksi/' + filename

        return render_template('uploaded.html', filename=filename, is_image=is_image, result_file=result_file_path)
    else:
        return 'File format not allowed.'



@app.route('/upload_again', methods=['POST'])
def upload_again():

    return redirect(url_for('index'))
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm',
                          'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts',
                          'wmv', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True)
