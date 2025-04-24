import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the request
        data = request.get_json()
        input_values = data.get('values', [])

        # Validate input
        if len(input_values) != 187:
            return jsonify({'error': f'Input must contain exactly 187 values. Received {len(input_values)}.'}), 400

        # Prepare input for prediction
        input_array = np.array(input_values).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = int(model.predict(scaled_input)[0])

        # Return prediction
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Railway's PORT env var
    app.run(debug=False, host='0.0.0.0', port=port)

