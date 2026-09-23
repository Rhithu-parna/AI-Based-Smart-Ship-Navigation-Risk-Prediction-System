from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = 'models/random_forest_navigation_model.joblib'
model = joblib.load(MODEL_PATH)

REC = {
    'Safe': 'Maintain the current course and speed. Continue routine monitoring.',
    'Moderate': 'Monitor weather and traffic conditions closely and be prepared to adjust course.',
    'High': 'Reduce speed, increase watchkeeping, and avoid nearby vessels where necessary.',
    'Critical': 'Issue an immediate navigation warning and consider an alternative route.'
}

EXP = {
    'Safe': 'No major navigation hazards were identified from the supplied vessel, radar, and weather conditions.',
    'Moderate': 'One or more conditions require closer monitoring before continuing the planned route.',
    'High': 'Elevated navigation risk was detected. Traffic, radar, or weather conditions may require intervention.',
    'Critical': 'Multiple serious hazards were detected. Immediate action is recommended.'
}

RISK_META = {
    'Safe': {'icon': '✓', 'class': 'safe'},
    'Moderate': {'icon': '!', 'class': 'moderate'},
    'High': {'icon': '!', 'class': 'high'},
    'Critical': {'icon': '×', 'class': 'critical'}
}


def build_input(form):
    return {
        'Weather_Condition': form.get('weather', 'Clear'),
        'Nearby_Vessel': form.get('nearby_vessel', 'No'),
        'Vessel_Distance_NM': float(form.get('vessel_distance', 10)),
        'Vessel_Speed_Knots': float(form.get('vessel_speed', 10)),
        'Radar_Object_Detected': form.get('radar_object', 'No'),
        'Radar_Distance_NM': float(form.get('radar_distance', 10)),
        'Ship_Speed_Knots': float(form.get('ship_speed', 15)),
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    values = {
        'weather': 'Clear',
        'nearby_vessel': 'No',
        'vessel_distance': 10,
        'vessel_speed': 10,
        'radar_object': 'No',
        'radar_distance': 10,
        'ship_speed': 15,
    }

    if request.method == 'POST':
        try:
            values.update({
                'weather': request.form.get('weather', 'Clear'),
                'nearby_vessel': request.form.get('nearby_vessel', 'No'),
                'vessel_distance': float(request.form.get('vessel_distance', 10)),
                'vessel_speed': float(request.form.get('vessel_speed', 10)),
                'radar_object': request.form.get('radar_object', 'No'),
                'radar_distance': float(request.form.get('radar_distance', 10)),
                'ship_speed': float(request.form.get('ship_speed', 15)),
            })

            data = build_input(request.form)
            input_data = pd.DataFrame([data])
            risk = model.predict(input_data)[0]

            probabilities = {label: 0.0 for label in ['Safe', 'Moderate', 'High', 'Critical']}
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(input_data)[0]
                classes = model.named_steps['classifier'].classes_
                probabilities.update({label: float(prob) * 100 for label, prob in zip(classes, probs)})

            confidence = probabilities.get(risk, 0.0)

            result = {
                'risk': risk,
                'risk_class': RISK_META[risk]['class'],
                'icon': RISK_META[risk]['icon'],
                'recommendation': REC[risk],
                'explanation': EXP[risk],
                'confidence': round(confidence, 1),
                'probabilities': {k: round(v, 1) for k, v in probabilities.items()},
                'model': 'Random Forest Classifier',
                'timestamp': datetime.now().strftime('%d %b %Y, %H:%M:%S'),
                'input_data': data,
            }
        except (ValueError, KeyError) as exc:
            result = {'error': f'Please enter valid navigation values. {exc}'}

    return render_template('index.html', result=result, values=values)


if __name__ == '__main__':
    app.run(debug=True)
