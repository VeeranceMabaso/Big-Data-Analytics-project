# data_processing.py
import os
from pyspark.sql import SparkSession
import rasterio
import numpy as np
from sklearn.decomposition import PCA

# Directory paths for the dataset
train_dir = 'archive/train'
val_dir = 'archive/val'

def get_tif_files(directory):
    """Get all .tif file paths recursively from a directory and its subdirectories."""
    tif_files = []
    print(f"Scanning directory: {directory}")  # Debug log
    for root, _, files in os.walk(directory):
        root = os.path.normpath(root)  # Ensure consistent path formatting
        print(f"Found directory: {root}, containing files: {files}")  # Debug log
        for file in files:
            if file.endswith('.tif'):
                file_path = os.path.join(root, file)
                print(f"Adding file: {file_path}")  # Debug log to show exact file path
                tif_files.append(file_path)
    return tif_files

def read_tif(file_path):
    """Read .tif image using rasterio."""
    print(f"Reading file: {file_path}")  # Debug log
    with rasterio.open(file_path) as src:
        return src.read()

def apply_pca(image, n_components=64):
    """Apply PCA to reduce dimensionality of hyperspectral image."""
    reshaped_image = image.reshape(image.shape[0], -1).T  # Flatten
    pca = PCA(n_components=n_components)
    reduced_image = pca.fit_transform(reshaped_image)
    return reduced_image.T.reshape(n_components, image.shape[1], image.shape[2])

def process_images_sequential(file_paths):
    """Process image files sequentially (without Spark) for debugging."""
    processed_images = []
    for file_path in file_paths:
        try:
            print(f"Reading file: {file_path}")  # Debug log
            image = read_tif(file_path)  # Read the image
            print(f"Original image shape: {image.shape}")  # Debug print: Check original shape
            
            # Apply PCA
            processed_image = apply_pca(image)  
            print(f"Processed image shape after PCA: {processed_image.shape}")  # Debug print: Check shape after PCA
            
            # Ensure the image is in the shape (height, width, bands)
            processed_image = np.transpose(processed_image, (1, 2, 0))  # Change shape to (height, width, bands)
            print(f"Final image shape: {processed_image.shape}")  # Debug print: Final shape
            
            # Append the reshaped image
            processed_images.append(processed_image)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Return processed images
    return processed_images


if __name__ == "__main__":
    # Get all .tif files from train and val folders
    train_files = get_tif_files(train_dir)
    val_files = get_tif_files(val_dir)

    # Process images sequentially (for debugging)
    print(f"Processing {len(train_files)} training images...")
    processed_train_images = process_images_sequential(train_files)
    
    print(f"Processing {len(val_files)} validation images...")
    processed_val_images = process_images_sequential(val_files)
    
    # Output result
    print(f"Processed {len(processed_train_images)} train images and {len(processed_val_images)} val images.")
