import os
import numpy as np
import tensorflow as tf
from data_processing import process_images_sequential, get_tif_files
from model import get_labels_from_folder
from model import train_model_with_cross_validation
from inference import app  

def main():
    # Data Processing 
    print("Processing the training data...")
    train_dir = 'archive/train'
    train_files = get_tif_files(train_dir)
    X_train = process_images_sequential(train_files)
    print(f"Processed {len(X_train)} training images.")

    # Generate labels for the training set (Health = 0, Rust = 1)
    y_train = np.array(get_labels_from_folder(train_dir))
    print(f"Generated {len(y_train)} labels for training.")

    # Train Model 
    print("\nStarting model training with cross-validation...")
    train_model_with_cross_validation()  # This trains the model and saves the best one

    # Inference Engine 
    print("\nStarting the inference engine (Flask app)...")
    app.run(debug=True, host='0.0.0.0', port=5000)  # Runs Flask server

if __name__ == "__main__":
    main()
