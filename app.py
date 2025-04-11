from flask import Flask, request, jsonify
from PIL import Image
from rembg import remove
import numpy as np
import tensorflow as tf
import io

# Define class names
class_names = ['bay leaf', 'cardamom', 'cinnamon', 'garlic', 'ginger', 
               'green chili', 'onion', 'red chili', 'star anise']

# Load your trained model (adjust file name if needed)
model = tf.keras.models.load_model("spicy_model.keras")

# Initialize Flask app
app = Flask(__name__)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    try:
        # Read image and remove background
        input_image = Image.open(file.stream).convert("RGBA")
        no_bg_image = remove(input_image)

        # Convert to RGB and resize
        image = no_bg_image.convert("RGB")
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return '🧪 Spicy Classifier Backend is running!'

