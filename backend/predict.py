import os
import sys
import json
import numpy as np
import joblib
from spectral import open_image
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
import absl.logging

# Suppress TensorFlow logs and progress bars
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings and info logs
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logs
absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress Abseil logs

# Load pre-trained models
KMEANS_MODEL = 'kmeans_model.pkl'
PCA_MODEL = 'pca_model.pkl'
RF_MODEL = 'random_forest_moisture_model.pkl'
CNN_MODEL = 'piperine_prediction_cnn_model.h5'

# Load models
try:
    kmeans = joblib.load(KMEANS_MODEL)
except Exception as e:
    print(json.dumps({"error": f"KMeans model loading error: {str(e)}"}), file=sys.stderr)
    sys.exit(1)

try:
    pca_model = joblib.load(PCA_MODEL)
except Exception as e:
    print(json.dumps({"error": f"PCA model loading error: {str(e)}"}), file=sys.stderr)
    sys.exit(1)

try:
    rf_model = joblib.load(RF_MODEL)
except Exception as e:
    print(json.dumps({"error": f"Random Forest model loading error: {str(e)}"}), file=sys.stderr)
    sys.exit(1)

try:
    cnn_model = load_model(CNN_MODEL)
except Exception as e:
    print(json.dumps({"error": f"CNN model loading error: {str(e)}"}), file=sys.stderr)
    sys.exit(1)

def mask_black_pepper(image):
    """Applies KMeans clustering to create a binary mask for black pepper."""
    pixels = image[:, :, 140].reshape(-1, 1)
    labels = kmeans.predict(pixels).reshape(image.shape[0], image.shape[1])
    return labels == 0  # Assuming pepper is cluster 0 (red)

def predict_moisture(img_path, hdr_path):
    """Runs the moisture prediction pipeline."""
    try:
        # Redirect TensorFlow logs to stderr
        original_stdout = sys.stdout
        sys.stdout = sys.stderr

        image = open_image(hdr_path).load()
        clusters = mask_black_pepper(image)

        # Apply the mask
        masked_image = image * clusters[..., np.newaxis]

        # Normalize the image
        scaler = MinMaxScaler()
        reshaped_img = masked_image.reshape(-1, masked_image.shape[-1])
        normalized_img = scaler.fit_transform(reshaped_img)
        normalized_img = normalized_img.reshape(masked_image.shape)

        # Extract average spectrum
        avg_spectrum = np.mean(normalized_img, axis=(0, 1))

        # Apply PCA
        pca_transformed = pca_model.transform(avg_spectrum.reshape(1, -1))

        # Predict moisture
        moisture_prediction = rf_model.predict(pca_transformed)[0]  # Extract scalar
        peperine_prediction = cnn_model.predict(pca_transformed)[0][0]  # Extract scalar

        # Restore original stdout
        sys.stdout = original_stdout

        return {
            "success": True,
            "moisture_prediction": float(moisture_prediction),  # Ensure scalar
            "peperine_prediction": float(peperine_prediction)   # Ensure scalar
        }

    except Exception as e:
        # Restore original stdout in case of an error
        sys.stdout = original_stdout
        return {"error": str(e)}

# Read input file paths
if len(sys.argv) != 3:
    print(json.dumps({"error": "Invalid number of arguments"}), file=sys.stderr)
    sys.exit(1)

img_file = sys.argv[1]
hdr_file = sys.argv[2]

# Run prediction and return JSON response
result = predict_moisture(img_file, hdr_file)
print(json.dumps(result))  # Ensure only JSON is printed