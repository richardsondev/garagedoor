import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Paths
DATASET_DIR = "./data"  # Path to dataset
MODEL_SAVE_PATH = "./app/model/garage_door_classifier.h5"

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def train_model():
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,  # Normalize pixel values
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2  # 80-20 train-validation split
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training"
    )

    validation_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    )

    # Build the model
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
    base_model.trainable = False  # Freeze base layers

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation="sigmoid")  # Binary classification
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # Train the model
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS
    )

    # Save the model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
