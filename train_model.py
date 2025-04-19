import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = "cleaned_healthmonitoring.csv"
df = pd.read_csv(file_path)

# Split BloodPressure into SystolicBP and DiastolicBP
df[['SystolicBP', 'DiastolicBP']] = df['BloodPressure'].str.split('/', expand=True)

# Convert relevant columns to numeric
numeric_columns = ['SystolicBP', 'DiastolicBP', 'HeartRate', 'BodyTemperature', 'OxygenSaturation', 'Height', 'Weight', 'BMI', 'ExerciseFrequency']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Map categorical features
df['ActivityLevel'] = df['ActivityLevel'].map({'resting': 0, 'walking': 1, 'running': 2})
df['SleepQuality'] = df['SleepQuality'].map({'excellent': 3, 'good': 2, 'fair': 1, 'poor': 0})
df['StressLevel'] = df['StressLevel'].map({'low': 0, 'moderate': 1, 'high': 2})
df['Diabetes'] = df['Diabetes'].map({'Yes': 1, 'No': 0})

# Handle missing values with default healthy values
default_values = {
    'SystolicBP': 120, 'DiastolicBP': 80, 'HeartRate': 70, 'BodyTemperature': 98.6,
    'OxygenSaturation': 97, 'ActivityLevel': 1, 'SleepQuality': 2, 'StressLevel': 0,
    'Height': 170, 'Weight': 70, 'BMI': 24, 'Diabetes': 0, 'ExerciseFrequency': 3
}
df.fillna(default_values, inplace=True)

# Define disease classification logic
def classify_disease(row):
    if row['SystolicBP'] >= 130 or row['DiastolicBP'] > 85:
        return 'Hypertension'
    elif row['HeartRate'] >= 95:
        return 'Heart Condition'
    elif row['BodyTemperature'] > 99.5:
        return 'Fever'
    elif row['OxygenSaturation'] < 94:
        return 'Respiratory Issue'
    elif row['StressLevel'] == 2:
        return 'Stress Induced Condition'
    elif row['ActivityLevel'] == 0 or row['SleepQuality'] <= 1:
        return 'Lifestyle Concern'
    elif row['Diabetes'] == 1:
        return 'Diabetes'
    elif row['BMI'] >= 30:
        return 'Obesity'
    elif row['ExerciseFrequency'] < 2:
        return 'Sedentary Lifestyle'
    else:
        return 'Healthy'

# Add Disease column
df['Disease'] = df.apply(classify_disease, axis=1)

# Check disease distribution
print("Disease Distribution:")
print(df['Disease'].value_counts())

# Features and target
X = df[['Age', 'Gender', 'SystolicBP', 'DiastolicBP', 'HeartRate', 'BodyTemperature', 'OxygenSaturation', 
        'ActivityLevel', 'SleepQuality', 'StressLevel', 'Height', 'Weight', 'BMI', 'Diabetes', 'ExerciseFrequency']]
y = df['Disease']

# Encode Gender column
X.loc[:, 'Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save the label encoder mapping
disease_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
with open("disease_mapping.pkl", "wb") as f:
    pickle.dump(disease_mapping, f)

print("Disease Mapping:", disease_mapping)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Save model
with open('disease_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and scaler trained and saved successfully!")

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy:.2%}')

# Test with a Healthy input
healthy_input = np.array([[25, 1, 120, 80, 70, 98.6, 97, 1, 2, 0, 170, 70, 24, 0, 3]])
healthy_input_scaled = scaler.transform(healthy_input)
predicted = model.predict(healthy_input_scaled)
predicted_label = label_encoder.inverse_transform(predicted)
print("Predicted Label for Healthy Input:", predicted_label)
