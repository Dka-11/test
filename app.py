import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Upload Photo Directory
UPLOAD_FOLDER = './images/'
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}

# Load Model
model = load_model('./Model/flowers_246_205.h5')

# Syarat file yang diperbolehkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Membagi string menjadi 2 bagian, berdasarkan titik terakhir yang ditemukan <-- rsplit('.',1)
# Mengambil bagian ektensi dari nama file setelah titik dan mengkonversi menjadi huruf kecil <-- [1].lower()

# Kelas bunga yang diakui oleh model
flower_classes = ["Anyelir", "Aster Cina", "Gerbera", "Lily Peruvian", "Lisianthus", "Matahari", "Mulut Naga"]

def predict_flower(files_path):
    img = image.load_img(files_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    images = np.vstack([img_array])
    
    flower_prediction = model.predict(images, batch_size=7)
    max_prob = np.max(flower_prediction)
    
    # if max_prob < 0.70:
    #     result = "Bukan Bunga"
    # else:
    #     index = np.argmax(flower_prediction)
    #     if index < len(flower_classes):
    #         result = "{}".format(flower_classes[index])
    #     else:
    #         result = "Tidak dapat diklasifikasi"
    
    index = np.argmax(flower_prediction)
    if index < len(flower_classes):
        result = "{}".format(flower_classes[index])
    else:
        result = "Tidak dapat diklasifikasi"
        
    # Menghapus file setelah dilakukannya klasifikasi
    os.remove(files_path)
    
    # Fungsi mengembalikan nama kelas dan akurasi klasifikasi
    return result, max_prob

# Create a Flask web application
app = Flask(__name__)

# Semua File yang akan diupload akan tersimpan
save_upload = app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def hello_world():
    return 'Response succes!'

@app.route('/Upload', methods=['POST'])
def upload_file():
    # Membuat key API
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Belum mengunggah foto'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(save_upload, filename)
        file.save(filepath)
        # Melakukan klasifikasi
        classification_result, accuracy = predict_flower(filepath)
        # Definisi Akurasi 2 Angka dibalik koma
        accuracy = "{:.2f}".format(accuracy)
        
        return jsonify({'classification result' : classification_result, 'accuracy' : accuracy}), 200
    
if __name__ == '__main__':
    # Port 8080 --> Google port
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT",8080)))
    # Default Port
    # app.run(debug=True, host='0.0.0.0', port=5000)