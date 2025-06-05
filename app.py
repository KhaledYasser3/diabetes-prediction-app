import numpy as np
import joblib
from flask import Flask, request, render_template
import os
from datetime import datetime
import json

app = Flask(__name__)

# Load the model and scaler
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# File to store history
HISTORY_FILE = 'prediction_history.json'

# Load history from file if exists
def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

# Save history to file
def save_history(history):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving history: {e}")

# Initialize history
prediction_history = load_history()

@app.route('/')
def home():
    return render_template('index.html', history=prediction_history)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('index.html', prediction_text='Error: Model or scaler not loaded. Please contact support.', history=prediction_history)

    try:
        # Extract and validate form inputs
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        # Validate non-negative values
        if any(f < 0 for f in features):
            return render_template('index.html', prediction_text='Error: All values must be non-negative.', history=prediction_history)
    except (ValueError, KeyError) as e:
        return render_template('index.html', prediction_text='Error: Invalid input. Please ensure all fields are filled with numeric values.', history=prediction_history)

    # Convert to numpy array and scale
    features = np.array([features])
    features_scaled = scaler.transform(features)

    # Make prediction
    try:
        prediction = model.predict(features_scaled)[0]
        result = 'Diabetes' if prediction == 1 else 'Healthy'
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error during prediction: {str(e)}', history=prediction_history)

    # Add to history with timestamp
    prediction_entry = {
        'features': features[0].tolist(),
        'result': result,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    prediction_history.append(prediction_entry)
    # Limit history to last 5 entries
    if len(prediction_history) > 5:
        prediction_history.pop(0)
    # Save to file
    save_history(prediction_history)

    # Return result in Arabic
    return render_template('index.html', prediction_text=f'النتيجة: {result} - Prediction completed successfully!', history=prediction_history)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
