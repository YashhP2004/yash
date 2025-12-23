import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub
import os
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Download and Setup Data (THE FIX)
# ==========================================
print("Downloading dataset...")
try:
    path = kagglehub.dataset_download("anshulm257/rice-disease-dataset")
    print("Dataset downloaded to:", path)

    # --- CRITICAL FIX START ---
    # The dataset has an inner folder named 'Rice_Leaf_AUG' that contains the actual classes.
    # We must point data_dir to this inner folder.
    data_dir = os.path.join(path, "Rice_Leaf_AUG")

    if not os.path.exists(data_dir):
        # Fallback: If the folder structure is different, look for it
        print("Standard path not found. Searching for class folders...")
        for root, dirs, files in os.walk(path):
            if "Brown_Spot" in dirs or "Healthy" in dirs:
                data_dir = root
                break
    # --- CRITICAL FIX END ---

    print(f"✅ Using Data Directory: {data_dir}")
    print("Contents:", os.listdir(data_dir))

    # ==========================================
    # 2. Data Preprocessing & Augmentation
    # ==========================================
    BATCH_SIZE = 32
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    EPOCHS = 15  # Increased slightly for better convergence

    train_datagen = ImageDataGenerator(
        rescale=1./255,             # Normalize pixel values
        rotation_range=20,          # Random rotation
        width_shift_range=0.2,      # Horizontal shift
        height_shift_range=0.2,     # Vertical shift
        shear_range=0.2,            # Shear
        zoom_range=0.2,             # Zoom
        horizontal_flip=True,       # Flip
        validation_split=0.2        # 20% validation split
    )

    print("\n--- Loading Training Data ---")
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    print("\n--- Loading Validation Data ---")
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Verify we found 6 classes
    class_names = list(train_generator.class_indices.keys())
    print(f"\n✅ Classes found ({len(class_names)}): {class_names}")

    # ==========================================
    # 3. Build CNN Model
    # ==========================================
    model = models.Sequential([
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten & Dense
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), # Prevent overfitting

        # Output Layer (Dynamic size based on actual classes found)
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # ==========================================
    # 4. Train the Model
    # ==========================================
    print("\nStarting Training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # ==========================================
    # 5. Visualizing Results
    # ==========================================
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('training_history.png')
    print("Training history saved as training_history.png")

    # ==========================================
    # 6. Save Model
    # ==========================================
    model.save('rice_leaf_disease_model.h5')
    print("\n✅ Model saved as 'rice_leaf_disease_model.h5'")

except Exception as e:
    print(f"An error occurred: {e}")
