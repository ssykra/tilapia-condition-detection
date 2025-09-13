from flask import Flask, request, render_template, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import datetime

app = Flask(__name__)

app.config['SECRET_KEY'] = b'\xf9^\xe8\x91\xb1J6.\xf9\xe3\xcf\x03T|)_\xc9Y\xfb\xec\\\x1f\x15]'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Load model
model_ikan_nila = load_model("mini-model_dataset1.keras")   # model ikan nila vs non ikan nila
model_kesegaran = load_model("model_final-3.keras")         # model klasifikasi kesegaran ikan nila

img_height = 224
img_width = 224

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(filepath):
    img = image.load_img(filepath, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(img_array, filename, img_path):
    threshold_ikan_nila = 0.35  
    threshold_kesegaran = 0.4

    prob_non_ikan = model_ikan_nila.predict(img_array)[0][0]

    if prob_non_ikan > threshold_ikan_nila:
        # Prediksi non ikan nila
        return {
            'filename': filename,
            'predicted_label': 'Bukan ikan nila',
            # 'confidence_message': 'Gambar bukan ikan nila, klasifikasi kesegaran tidak dilakukan.',
            'prob_ikan_nila': float(1 - prob_non_ikan),  # probabilitas ikan nila
            'img_path': img_path
        }
    else:
        # Prediksi ikan nila, lanjut klasifikasi kesegaran
        prob_kesegaran = model_kesegaran.predict(img_array)[0][0]
        if prob_kesegaran < threshold_kesegaran:
            predicted_label = 'Segar'
        else:
            predicted_label = 'Tidak Segar'

        return {
            'filename': filename,
            'predicted_label': predicted_label,
            'prob_segar': float(1 - prob_kesegaran),
            'prob_tidak_segar': float(prob_kesegaran),
            # 'confidence_message': None,
            'prob_ikan_nila': float(1 - prob_non_ikan),
            'img_path': img_path
        }

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/detail')
def detail():
    return render_template("detail.html")

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    result = None
    img_path = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_path = os.path.join('uploads', filename)

            img_array = preprocess_image(filepath)
            result = classify_image(img_array, filename, img_path)

            session['result'] = result
            return redirect(url_for('classification'))

    result = session.get('result')
    session.pop('result', None)

    return render_template("classification.html", result=result, img_path=img_path)

@app.route('/capture_from_mobile', methods=['POST'])
def capture_from_mobile():
    image_data = request.form.get('image_data')
    if not image_data:
        return redirect(url_for('classification'))

    header, encoded = image_data.split(",", 1)
    binary_data = base64.b64decode(encoded)
    image_pil = Image.open(BytesIO(binary_data)).convert("RGB")

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"mobile_{now}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_pil.save(filepath)
    img_path = os.path.join('uploads', filename)

    img_array = preprocess_image(filepath)
    result = classify_image(img_array, filename, img_path)

    session['result'] = result
    return redirect(url_for('classification'))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
