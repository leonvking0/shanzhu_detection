from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import torch
from transformers import pipeline

app = Flask(__name__)

# Initialize the object detection pipeline
pipe = pipeline("object-detection", model="hustvl/yolos-small", device=1)

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    # Decode base64 image
    image_data = base64.b64decode(request.json['image'])
    image = Image.open(io.BytesIO(image_data))
    
    # Perform object detection
    results = pipe(image)
    
    # Format results
    detections = [
        {
            'label': result['label'],
            'score': result['score'],
            'box': result['box']
        }
        for result in results
    ]
    
    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
