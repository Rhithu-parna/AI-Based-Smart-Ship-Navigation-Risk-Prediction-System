import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

np.random.seed(42)
number_of_records = 1000

weather = np.random.choice(['Clear', 'Rain', 'Storm'], number_of_records, p=[0.55, 0.30, 0.15])
nearby_vessel = np.random.choice(['No', 'Yes'], number_of_records, p=[0.60, 0.40])
vessel_distance = np.round(np.random.uniform(0.5, 12, number_of_records), 2)
vessel_speed = np.round(np.random.uniform(0, 25, number_of_records), 2)
radar_object = np.random.choice(['No', 'Yes'], number_of_records, p=[0.70, 0.30])
radar_distance = np.round(np.random.uniform(0.5, 10, number_of_records), 2)
ship_speed = np.round(np.random.uniform(5, 25, number_of_records), 2)

risk_levels = []
for w, vessel, v_distance, radar, r_distance in zip(
    weather, nearby_vessel, vessel_distance, radar_object, radar_distance
):
    risk_score = 0
    if w == 'Rain':
        risk_score += 1
    elif w == 'Storm':
        risk_score += 3
    if vessel == 'Yes' and v_distance < 5:
        risk_score += 2
    elif vessel == 'Yes' and v_distance < 8:
        risk_score += 1
    if radar == 'Yes' and r_distance < 3:
        risk_score += 3
    elif radar == 'Yes' and r_distance < 6:
        risk_score += 1

    if risk_score >= 5:
        risk = 'Critical'
    elif risk_score >= 3:
        risk = 'High'
    elif risk_score >= 1:
        risk = 'Moderate'
    else:
        risk = 'Safe'
    risk_levels.append(risk)

data = pd.DataFrame({
    'Weather_Condition': weather,
    'Nearby_Vessel': nearby_vessel,
    'Vessel_Distance_NM': vessel_distance,
    'Vessel_Speed_Knots': vessel_speed,
    'Radar_Object_Detected': radar_object,
    'Radar_Distance_NM': radar_distance,
    'Ship_Speed_Knots': ship_speed,
    'Risk_Level': risk_levels
})

dataset_path = 'data/synthetic_ship_navigation_dataset.csv'
data.to_csv(dataset_path, index=False)

X = data.drop('Risk_Level', axis=1)
y = data['Risk_Level']

categorical_features = ['Weather_Condition', 'Nearby_Vessel', 'Radar_Object_Detected']
numerical_features = ['Vessel_Distance_NM', 'Vessel_Speed_Knots', 'Radar_Distance_NM', 'Ship_Speed_Knots']

preprocessor = ColumnTransformer([
    ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('numerical', 'passthrough', numerical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('Training Random Forest model...')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f'Model Accuracy: {accuracy * 100:.2f}%')
print('\nClassification Report:')
print(classification_report(y_test, predictions))

model_path = 'models/random_forest_navigation_model.joblib'
joblib.dump(model, model_path)
print(f'Random Forest model saved successfully: {model_path}')
print('PROJECT MODEL READY!')
