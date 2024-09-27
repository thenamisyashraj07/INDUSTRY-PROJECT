import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Correct directory for dataset
dataset_dir = r'D:\coding for funn\endangered Animal.v2i.tensorflow'

# Custom callback to format accuracy and loss in percentage
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy') * 100
        val_accuracy = logs.get('val_accuracy') * 100
        loss = logs.get('loss') * 100
        val_loss = logs.get('val_loss') * 100
        print(f"Epoch {epoch + 1}: accuracy: {accuracy:.2f}% - val_accuracy: {val_accuracy:.2f}% - loss: {loss:.4f}% - val_loss: {val_loss:.4f}%")

# Model architecture
def create_cnn_model(input_shape=(128, 128, 3), num_classes=10):
    model = Sequential([
        Input(shape=input_shape),  # Input layer
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # For multi-class classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to load all images from subdirectories
def load_images_from_directory(directory, img_size=(128, 128)):
    images = []
    labels = []
    label_names = []

    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory does not exist - {directory}")
        return images, labels, label_names
    
    print(f"Loading images from: {directory}")
    
    # Process each subdirectory
    for label_name in os.listdir(directory):
        subdirectory = os.path.join(directory, label_name)
        
        if os.path.isdir(subdirectory):
            print(f"Processing subdirectory: {subdirectory}")
            for filename in os.listdir(subdirectory):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdirectory, filename)
                    print(f"Found image: {img_path}")  # Debug: Print image path
                    
                    try:
                        # Load the image and resize
                        img = load_img(img_path, target_size=img_size)
                        img_array = img_to_array(img)
                        images.append(img_array)
                        labels.append(label_name)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
    
    # Convert to numpy arrays and normalize the image data
    images = np.array(images) / 255.0
    labels = np.array(labels)
    
    # Convert labels to integers
    label_names = sorted(set(labels))
    label_map = {name: index for index, name in enumerate(label_names)}
    labels = np.array([label_map[label] for label in labels])
    
    # One-hot encode labels
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_names))
    
    # Debug: Print the number of loaded images and labels
    print(f"Total images loaded: {len(images)}")
    print(f"Unique labels found: {label_names}")
    
    return images, labels, label_names

# Main execution block
if __name__ == "__main__":  # Corrected this line
    # Load images and labels from the dataset directory
    images, labels, label_names = load_images_from_directory(dataset_dir)

    # Check if any images were loaded
    if len(images) == 0:
        print("Error: No images were loaded. Exiting...")
    else:
        # Split the dataset into training and validation sets
        train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Create CNN model
        model = create_cnn_model(num_classes=len(label_names))  # Update num_classes based on actual label count

        # Train the model and capture history, using CustomCallback for percentage formatting
        history = model.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            epochs=10,
            verbose=0,  # Suppress default verbose
            callbacks=[CustomCallback()]  # Add the custom callback
        )

        # Ensure 'models' directory exists
        if not os.path.exists('models'):
            os.makedirs('models')

        # Save the trained model
        model.save('models/model.keras')

        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        # Display the plots
        plt.tight_layout()
        plt.show()

        # Get final accuracy on the validation set
        loss, accuracy = model.evaluate(val_images, val_labels)
        print(f"Final validation accuracy: {accuracy * 100:.2f}%")
