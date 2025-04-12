from flask import Flask, request, jsonify
from PIL import Image
from rembg import remove
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Define class names
class_names = ['Bay leaf', 'Cardamom', 'Cinnamon', 'Garlic', 'Ginger', 
               'Green chili', 'Onion', 'Red chili', 'Star anise']

# Load model
model = tf.keras.models.load_model("spicy_model.keras")

def preprocess_image(image):
    """Process image for model prediction"""
    image = image.convert("RGBA")
    no_bg_image = remove(image)
    rgb_image = no_bg_image.convert("RGB")
    rgb_image = rgb_image.resize((128, 128))
    image_array = np.array(rgb_image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions[0])
        
        return jsonify({
            'predicted_class': class_names[predicted_index],
            'confidence': round(float(np.max(predictions[0])) * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Spice Classifier API is running',
        'model_loaded': True,
        'ready_for_predictions': True
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5012))
    app.run(host='0.0.0.0', port=port)
