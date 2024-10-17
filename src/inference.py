from data_processing import apply_pca
from flask import Flask, request, render_template
import tensorflow as tf
import rasterio
import numpy as np

app = Flask(__name__, template_folder='../templates')

# Load the trained model
model = tf.keras.models.load_model('dist/cnn_model.h5')

def preprocess_image(file):
    """
    Preprocess the uploaded .tif file for PCA and model prediction.
    """
    with rasterio.open(file) as src:
        img = src.read()  # Read image
    # Apply PCA for dimensionality reduction
    reduced_img = apply_pca(img, n_components=64)  # Adjust n_components as needed
    return reduced_img

@app.route('/')
def index():
    return render_template('index.html')  # Simple HTML form to upload an image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {"error": "No file part"}, 400

    file = request.files['file']
    
    if file.filename == '':
        return {"error": "No selected file"}, 400

    # Preprocess the image and predict
    image = preprocess_image(file)
    prediction = model.predict(image[np.newaxis, ...])  # Add batch dimension
    print(prediction)
    # Extract the predicted class and confidence score
    class_label = int(np.argmax(prediction))  # 0 or 1
    confidence_score = np.max(prediction)  # Confidence of the prediction

    # Map the class label to actual category names
    class_name = "Health" if class_label == 0 else "Rust"

    # Render the result template with prediction and confidence score
    return render_template('result.html', 
                           prediction=class_name, 
                           confidence=round(confidence_score * 100, 2))  # Display percentage confidence

if __name__ == "__main__":
    app.run(debug=True)
