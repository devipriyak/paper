import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7, InceptionV3, ResNet50V2, ResNet101V2, MobileNetV3Large
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Define directories
data_dir = '/content/drive/MyDrive/MP_Pics Classification Dataset'
train_dir = '/content/drive/MyDrive/MP_Pics Classification Dataset/TRAIN'
test_dir = '/content/drive/MyDrive/MP_Pics Classification Dataset/TEST'

# Create training and testing directories if they don't exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Function to split data into training and testing sets
def split_data(train_size=0.7):
    for class_name in ['Bead', 'Fiber', 'Fragment']:
        class_folder = os.path.join(data_dir, class_name)
        train_class_folder = os.path.join(train_dir, class_name)
        test_class_folder = os.path.join(test_dir, class_name)
        if not os.path.exists(train_class_folder):
            os.makedirs(train_class_folder)
        if not os.path.exists(test_class_folder):
            os.makedirs(test_class_folder)

        # Get list of image files
        image_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Split files into training and testing sets
        train_files, test_files = train_test_split(image_files, test_size=1 - train_size, random_state=42)

        # Copy files to train folder
        for file in train_files:
            shutil.copy(file, os.path.join(train_class_folder, os.path.basename(file)))

        # Copy files to test folder
        for file in test_files:
            shutil.copy(file, os.path.join(test_class_folder, os.path.basename(file)))

# Split data into training and testing sets
split_data(train_size=0.8)

# Parameters
img_width, img_height = 224, 224
batch_size = 32
epochs = 20  # Increased epochs for better convergence

# Data preprocessing with increased augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,  # Adding vertical flipping
    brightness_range=[0.8, 1.2],  # Adjusting brightness
    channel_shift_range=0.2,  # Random channel shifts
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Define a list of models to iterate over
models_to_use = ['EfficientNetB7', 'InceptionV3', 'ResNet50V2', 'ResNet101V2', 'MobileNetV3Large']

# Function to load a model dynamically
def load_model(model_name, input_shape):
    if model_name == 'EfficientNetB7':
        base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'ResNet50V2':
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'ResNet101V2':
        base_model = ResNet101V2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'MobileNetV3Large':
        base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError('Unsupported model architecture.')

    return base_model

# Iterate over each model and train it
for model_name in models_to_use:
    print(f"Training model: {model_name}")

    # Load selected model
    base_model = load_model(model_name, (img_width, img_height, 3))

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)  # Increased units in the dense layer
    x = Dense(128, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    # Combine base model and custom classification head
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model with a lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=test_generator.samples // batch_size
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Accuracy for {model_name}: {test_accuracy}")

    # Predict classes for test set
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    # Get true labels
    true_labels = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Print classification report
    print(f"Classification Report for {model_name}:")
    print(classification_report(true_labels, y_pred, target_names=class_labels))

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    cmd.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.show()

    # Save the model
    model.save(f'{model_name}_model.h5')
    print(f"Model {model_name} saved successfully.")

    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(f'{model_name}_accuracy.png')
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(f'{model_name}_loss.png')
    plt.show()
