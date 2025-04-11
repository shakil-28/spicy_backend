from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from rembg import remove
import io
import base64
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("spicy_model.keras")

# Define your class names
class_names = [
    'bay leaf',
    'cardamom',
    'cinnamon',
    'garlic',
    'ginger',
    'green chili',
    'onion',
    'red chili',
    'star anise'
]

# Preprocess the image
def preprocess_image(image_bytes):
    # Remove background
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    output_image = remove(input_image)
    rgb_image = output_image.convert("RGB")

    # Resize and normalize
    resized_image = rgb_image.resize((128, 128))
    image_array = np.array(resized_image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'No image provided'}), 400

        image_bytes = base64.b64decode(data)
        image = preprocess_image(image_bytes)

        predictions = model.predict(image)
        predicted_index = int(np.argmax(predictions))
        predicted_label = class_names[predicted_index]

        return jsonify({
            'prediction': predicted_label,
            'confidence': float(np.max(predictions))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run server locally (won't affect Render)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
