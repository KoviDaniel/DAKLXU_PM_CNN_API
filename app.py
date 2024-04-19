from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from skimage import filters, restoration, exposure
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Betöltjük a mentett modellt
#model = load_model('C:\CNN_model\model.h5')
file_path = os.path.join(os.getcwd(), 'KDmodel3.h5')
model = load_model(file_path)

classes = ['Acer_campestre','Acer_platanoides','Acer_tataricum',
           'Alnus_glutinosa', 'Alnus_incana', 'Betula_pendula',
           'Betula_pubescens', 'Castanea_sativa', 'Fagus_sylvatica',
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
    # img = Image.open(image_path).convert('L')
    # img = img.resize((224, 224))
    # img_array = np.array(img)
    # threshold = filters.threshold_otsu(img_array)
    # binary_img = img_array > threshold
    # denoised_img = restoration.denoise_tv_chambolle(binary_img, weight=0.1)
    # p2, p98 = np.percentile(img_array, (2, 98))
    # img_rescale = exposure.rescale_intensity(img_array, in_range=(p2, p98))
    # inverted_img = np.invert(img_rescale)
    # edges = filters.sobel(inverted_img)
    
    # # Az előfeldolgozott kép visszaadása a modell számára
    # processed_img_array = np.expand_dims(edges, axis=0)
    # return processed_img_array

def transform_image(image_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    threshold = filters.threshold_otsu(img_array)
    binary_img = img_array > threshold
    denoised_img = restoration.denoise_tv_chambolle(binary_img, weight=0.1)
    p2, p98 = np.percentile(img_array, (2, 98))
    img_rescale = exposure.rescale_intensity(img_array, in_range=(p2, p98))
    inverted_img = np.invert(img_rescale)
    edges = filters.sobel(inverted_img)
    inverted_edges =  1 - edges
    enhanced_edges = inverted_edges > 0.93
    plt.imsave('transformed_image.jpg', enhanced_edges, cmap='gray')

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
        tr_img_path = 'transformed_image.jpg'
        file.save(img_path)

        transform_image(img_path)
        processed_image = preprocess_image(tr_img_path)
        prediction = model.predict(processed_image)

        max_prob_index = np.argmax(prediction)
        predicted_class = classes[max_prob_index]
        result = {
            'PredictedClass': predicted_class,
        }
        os.remove(img_path)
        os.remove(tr_img_path)

        return jsonify(result)

    except Exception as e:
        os.remove(img_path)
        os.remove(tr_img_path)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)