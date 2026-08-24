"""
Manufacturing Defect Detection Microservice
Exposes a Flask API to serve image defect predictions using a trained TensorFlow model.
Implements MLOps Observability, Alerting, and Graceful Degradation.
"""
import os
import json
import logging
# pylint: disable=import-error,no-member
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# --- Observability Gate: Structured Logging Setup ---
# In production, these logs are ingested by tools like Datadog or CloudWatch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops_inference_service")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

MODEL_PATH = "defect_detection_model.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# --- Configuration Thresholds ---
# If the score is between 0.35 and 0.65, the model is uncertain
UNCERTAINTY_LOWER_BOUND = 0.35
UNCERTAINTY_UPPER_BOUND = 0.65

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "Manufacturing Defect Detection API"})

@app.route("/predict", methods=["POST"])
def predict():
    """Inference endpoint to evaluate component condition."""
    try:
        data = request.get_json(silent=True)
        is_simulated = False
        
        # Data validation and fallback
        if data and "features" in data:
            sample_input = np.array(data["features"], dtype=np.float32)
        else:
            # Fallback for testing/simulation
            sample_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
            is_simulated = True

        # Inference
        preds = MODEL.predict(sample_input)
        score = float(preds[0][0].item())
        
        # --- Graceful Degradation & Alerting Logic ---
        requires_review = False
        if UNCERTAINTY_LOWER_BOUND <= score <= UNCERTAINTY_UPPER_BOUND:
            result = "Requires Human Review"
            requires_review = True
            
            # Trigger an alert payload (simulated via log for this environment)
            alert_payload = {
                "event": "MODEL_UNCERTAINTY_ALERT",
                "message": "Confidence score fell into uncertainty threshold. Potential data drift.",
                "score": score
            }
            logger.warning(json.dumps(alert_payload))
            
        else:
            result = "Defective" if score > 0.5 else "Normal"

        # --- Observability Gate: Telemetry Payload ---
        # Log every transaction for dashboard monitoring
        telemetry = {
            "event": "INFERENCE_TRANSACTION",
            "is_simulated_input": is_simulated,
            "prediction": result,
            "confidence_score": score,
            "routed_for_review": requires_review
        }
        logger.info(json.dumps(telemetry))

        return jsonify({
            "prediction": result,
            "confidence_score": score,
            "routed_for_review": requires_review
        })
        
    except (ValueError, TypeError, KeyError) as e:
        logger.error(json.dumps({"event": "INFERENCE_ERROR", "error": str(e)}))
        return jsonify({"error": f"Invalid payload structure: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)