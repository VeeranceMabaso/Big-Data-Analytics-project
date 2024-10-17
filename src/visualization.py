import matplotlib.pyplot as plt
import rasterio
from data_processing import apply_pca  # Import the PCA function

def visualize_images(original_image, reduced_image):
    plt.subplot(1, 2, 1)
    plt.imshow(original_image[0], cmap='gray')  # Visualize one band of the original image
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(reduced_image[0], cmap='gray')  # Visualize the PCA-reduced image
    plt.title("Reduced Image (PCA)")
    
    plt.show()

if __name__ == "__main__":
    # Example of visualizing a .tif file
    with rasterio.open('archive/train/Health/hyper (1).tif') as src:
        original_image = src.read()  # Read the original image
    
    # Apply PCA on the original image (assuming it has many bands)
    reduced_image = apply_pca(original_image)  # Apply PCA function to reduce dimensions
    
    # Visualize the original image and PCA-reduced image side by side
    visualize_images(original_image, reduced_image)
