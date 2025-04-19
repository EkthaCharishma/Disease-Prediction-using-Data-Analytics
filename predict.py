import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Load model, scaler, and label encoder
with open('disease_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('disease_mapping.pkl', 'rb') as f:
    disease_mapping = pickle.load(f)
reverse_mapping = {v: k for k, v in disease_mapping.items()}  # Reverse mapping for prediction

# Treatment mapping
treatment_map = {
    "Diabetes": "Maintain a balanced diet, exercise regularly, and monitor blood sugar levels.",
    "Hypertension": "Reduce sodium intake, increase physical activity, and manage stress.",
    "Heart Condition": "Adopt a heart-healthy diet, avoid smoking, and take prescribed medications.",
    "Fever": "Stay hydrated, rest, and take fever-reducing medications if necessary.",
    "Respiratory Issue": "Use prescribed inhalers, avoid allergens, and maintain proper ventilation.",
    "Stress Induced Condition": "Practice mindfulness, engage in relaxation exercises, and seek therapy if needed.",
    "Lifestyle Concern": "Improve sleep patterns, adopt an active lifestyle, and maintain a balanced diet.",
    "Obesity": "Follow a structured diet, increase physical activity, and consult a nutritionist if needed.",
    "Sedentary Lifestyle": "Increase daily movement, schedule regular exercise, and reduce prolonged sitting time.",
    "Healthy": "No treatment needed. Maintain a healthy lifestyle!"
}

# Mapping functions
def map_activity(value):
    return {'resting': 0, 'walking': 1, 'running': 2}.get(value, 1)

def map_sleep(value):
    return {'excellent': 3, 'good': 2, 'fair': 1, 'poor': 0}.get(value, 2)

def map_stress(value):
    return {'low': 0, 'moderate': 1, 'high': 2}.get(value, 1)

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == 'admin' and password == 'password':
        return redirect(url_for('index'))
    else:
        return render_template('login.html', error="Invalid username or password.")

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capture input data
        input_data = pd.DataFrame([{
            'Age': float(request.form['age']),
            'Gender': 1 if request.form['gender'] == 'Male' else 0,
            'SystolicBP': float(request.form['systolic_bp']),
            'DiastolicBP': float(request.form['diastolic_bp']),
            'HeartRate': float(request.form['heart_rate']),
            'BodyTemperature': float(request.form['body_temperature']),
            'OxygenSaturation': float(request.form['oxygen_saturation']),
            'ActivityLevel': map_activity(request.form['activity']),
            'SleepQuality': map_sleep(request.form['sleep']),
            'StressLevel': map_stress(request.form['stress']),
            'Height': float(request.form['height']),
            'Weight': float(request.form['weight']),
            'BMI': float(request.form['bmi']),
            'Diabetes': 1 if request.form['diabetes'] == 'Yes' else 0,
            'ExerciseFrequency': float(request.form['exercise_frequency'])
        }])

        # Ensure feature alignment
        expected_columns = ['Age', 'Gender', 'SystolicBP', 'DiastolicBP', 'HeartRate', 'BodyTemperature',
                            'OxygenSaturation', 'ActivityLevel', 'SleepQuality', 'StressLevel', 'Height',
                            'Weight', 'BMI', 'Diabetes', 'ExerciseFrequency']
        input_data = input_data[expected_columns]

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict
        prediction_numeric = model.predict(input_data_scaled)[0]
        prediction = reverse_mapping.get(prediction_numeric, "Unknown Condition")

        # Map prediction to treatment
        treatment = treatment_map.get(prediction, "Consult your doctor for more guidance.")

        # Generate Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(input_data_scaled, annot=True, cmap='YlGnBu', cbar=True)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        heatmap_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        # Display result
        return render_template('result.html',
                                prediction=prediction,
                                treatment=treatment,
                                heatmap_url=f"data:image/png;base64,{heatmap_url}")
    except KeyError as e:
        return f"Missing form field: {str(e)}"
    except ValueError as e:
        return f"Invalid input format: {str(e)}. Please enter valid numbers."
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
