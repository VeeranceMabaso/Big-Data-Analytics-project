import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from data_processing import process_images_sequential, get_tif_files

# Directory paths for the dataset
train_dir = 'archive/train'

def create_cnn(input_shape):
    """Create a CNN model for classifying the hyperspectral images."""
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Use sigmoid for binary classification
    return model

def get_labels_from_folder(directory):
    """
    Generate labels from the folder structure.
    Assumes that 'Health' corresponds to class 0 (Mild FHB)
    and 'Rust' corresponds to class 1 (Serious FHB).
    """
    labels = []
    for root, dirs, files in os.walk(directory):
        if "Health" in root:
            labels.extend([0] * len(files))  # Class 0 for 'Health'
        elif "Rust" in root:
            labels.extend([1] * len(files))  # Class 1 for 'Rust'
    return labels

def plot_loss(history, fold):
    """Plot the loss of each fold with a different color and add a legend."""
    epochs = range(1, len(history.history['loss']) + 1)
    colors = ['green', 'blue', 'red', 'purple', 'orange']  # Define colors for each fold
    plt.plot(epochs, history.history['loss'], label=f'Fold {fold + 1}', color=colors[fold])



def save_confusion_matrix(cm):
    """Save the best confusion matrix as an image."""
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f'{z}', ha='center', va='center')

    plt.title('Best Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('static/best_confusion_matrix.png')  # Save the best confusion matrix
    plt.show()

def train_model_with_cross_validation():
    # Get all .tif files from train folder
    train_files = get_tif_files(train_dir)

    # Process the images
    print(f"Processing {len(train_files)} training images...")
    X_train = process_images_sequential(train_files)
    print(f"Processed {len(X_train)} training images.")

    # Generate labels for the training set (Health = 0, Rust = 1)
    y_train = np.array(get_labels_from_folder(train_dir))
    print(f"Generated {len(y_train)} labels for training.")

    # Define input shape based on processed data (e.g., after PCA)
    input_shape = (X_train[0].shape[1], X_train[0].shape[2], X_train[0].shape[0])  # (height, width, bands)
    print(f"Input shape for CNN: {input_shape}")

    # K-fold Cross-Validation (5-fold)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    f1_scores = []
    confusion_matrices = []
    best_f1_score = 0
    best_conf_matrix = None
    best_model = None

    plt.figure(figsize=(10, 6))  # For the loss plot

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        print(f"\nTraining fold {fold + 1}/{kfold.n_splits}...")

        # Split the data into training and validation sets for this fold
        X_train_fold, X_val_fold = np.array(X_train)[train_idx], np.array(X_train)[val_idx]
        y_train_fold, y_val_fold = np.array(y_train)[train_idx], np.array(y_train)[val_idx]

        # Create and compile the model
        model = create_cnn(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model and get the training history
        history = model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=32, verbose=0)

        # Plot the loss for this fold
        plot_loss(history, fold)

        # Evaluate the model on the validation set
        val_predictions = model.predict(X_val_fold)
        val_predictions_classes = (val_predictions > 0.5).astype("int32")

        # Compute confusion matrix and F1 score for this fold
        fold_conf_matrix = confusion_matrix(y_val_fold, val_predictions_classes)
        fold_f1_score = f1_score(y_val_fold, val_predictions_classes, average='weighted')

        confusion_matrices.append(fold_conf_matrix)
        f1_scores.append(fold_f1_score)

        print(f"Fold {fold + 1} Confusion Matrix:\n", fold_conf_matrix)
        print(f"Fold {fold + 1} F1 Score: {fold_f1_score:.4f}")

        # Track the best model
        if fold_f1_score > best_f1_score:
            best_f1_score = fold_f1_score
            best_conf_matrix = fold_conf_matrix
            best_model = model

    if best_model:
        best_model.save('dist/cnn_model.h5')
        print(f"Best model saved with F1 Score: {best_f1_score:.4f}")
              
    plt.title('Loss per Epoch for Each Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')  # Add a legend for all folds
    plt.savefig('static/loss_per_fold.png')  # Save the loss plot
    plt.show()

    # Plot and save the confusion matrix for the best model
    if best_conf_matrix is not None:
        save_confusion_matrix(best_conf_matrix)

    # Print average F1 score
    avg_f1_score = np.mean(f1_scores)
    print(f"\nAverage F1 Score: {avg_f1_score:.4f}")

if __name__ == "__main__":
    train_model_with_cross_validation()
