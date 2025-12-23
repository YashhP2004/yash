import sys
import os

# Add custom library path for TensorFlow to avoid Long Path issues
CUSTOM_LIB_PATH = r"C:\Users\HP\tf_libs"
LOCAL_LIBS_PATH = os.path.join(os.getcwd(), "libs")
SYSTEM_SITE_PACKAGES = r"C:\Program Files\Python312\Lib\site-packages"

if os.path.exists(LOCAL_LIBS_PATH) and LOCAL_LIBS_PATH not in sys.path:
    sys.path.insert(0, LOCAL_LIBS_PATH)

if os.path.exists(CUSTOM_LIB_PATH) and CUSTOM_LIB_PATH not in sys.path:
    sys.path.append(CUSTOM_LIB_PATH)

if os.path.exists(SYSTEM_SITE_PACKAGES) and SYSTEM_SITE_PACKAGES not in sys.path:
    sys.path.append(SYSTEM_SITE_PACKAGES)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    print(f"TensorFlow not available. Prediction will be disabled. Error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    TF_AVAILABLE = False
    print(f"TensorFlow not available (General Exception). Error: {e}")
    import traceback
    traceback.print_exc()

import numpy as np
from PIL import Image

def load_model(model_path):
    """Loads the trained Keras model."""
    if not TF_AVAILABLE:
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_image(model, image_file, class_names):
    """
    Predicts the class of the image using the loaded model.
    """
    if not TF_AVAILABLE:
        return None, 0.0

    try:
        # Load and preprocess the image
        img = Image.open(image_file)
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Handle alpha channel if present
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
            
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array / 255.0  # Normalize pixel values

        # Make prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) # Get probability scores

        # Get result
        predicted_class = class_names[np.argmax(predictions)]
        confidence = 100 * np.max(predictions)
        
        return predicted_class, confidence

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, 0.0
