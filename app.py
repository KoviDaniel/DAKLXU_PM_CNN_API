from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Betöltjük a mentett modellt
model = load_model('C:\CNN_model\model.h5')

classes = ['Acer_campestre','Acer_platanoides','Acer_tataricum',
           'Alnus_glutinosa', 'Alnus_incana', 'Betula_pendula',
           'Betula_pubescens', 'Castanea_savita', 'Fagus_sylvatica',
           'Fraxinus_excelsior', 'Fraxinus_ornus', 'Populus_alba',
           'Populus_canescens', 'Populus_tremula', 'Prunus_mahaleb',
           'Prunus_padus', 'Quercus_petraea', 'Salix_caprea',
           'Sorbus_aria', 'Sorbus_aucuparia', 'Sorbus_domestica',
           'Sorbus_torminalis', 'Tilia_cordata', 'Tilia_platyphyllos',
           'Tilia_tomentosa', 'Ulmus_glabra', 'Ulmus_laevis', 'Ulmus_minor']

# Kép előfeldolgozása a modell számára
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # A modell VGG-16-hoz 224x224-es méretű képet vár
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# API végpont a képek fogadására és predikciójának visszaadására
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img_path = 'temp_image.jpg'
        file.save(img_path)

        processed_image = preprocess_image(img_path)
        prediction = model.predict(processed_image)

        max_prob_index = np.argmax(prediction)
        predicted_class = classes[max_prob_index]
        result = {
            'PredictedClass': predicted_class,
        }
        os.remove(img_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)