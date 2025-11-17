from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models and scaler
rf_model = joblib.load('models/rf_smartphone_addiction_model.joblib')
svm_model = joblib.load('models/svm_smartphone_addiction_model.joblib')
scaler = joblib.load('models/scaler_svm.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    rf_prediction = None
    svm_prediction = None
    rf_proba = None
    svm_proba = None

    if request.method == 'POST':
        feature_names = [
            'daily_screen_time', 'app_sessions', 'social_media_usage',
            'gaming_time', 'notifications', 'night_usage',
            'age', 'work_study_hours', 'stress_level', 'apps_installed'
        ]

        # Get input values
        input_values = [float(request.form.get(f)) for f in feature_names]
        input_array = np.array(input_values).reshape(1, -1)

        # Scale input for SVM
        input_scaled = scaler.transform(input_array)

        # Random Forest prediction
        rf_pred = rf_model.predict(input_array)[0]
        rf_prob_values = rf_model.predict_proba(input_array)[0]

        # Flip logic: 0 = Addicted, 1 = Not Addicted
        rf_prediction = "Addicted" if rf_pred == 0 else "Not Addicted"
        rf_proba = {
            "Not Addicted": round(rf_prob_values[1]*100, 2),
            "Addicted": round(rf_prob_values[0]*100, 2)
        }

        # SVM prediction
        svm_pred = svm_model.predict(input_scaled)[0]
        svm_prob_values = svm_model.predict_proba(input_scaled)[0]

        svm_prediction = "Addicted" if svm_pred == 0 else "Not Addicted"
        svm_proba = {
            "Not Addicted": round(svm_prob_values[1]*100, 2),
            "Addicted": round(svm_prob_values[0]*100, 2)
        }

    return render_template('index.html',
                           rf_prediction=rf_prediction, rf_proba=rf_proba,
                           svm_prediction=svm_prediction, svm_proba=svm_proba)

if __name__ == '__main__':
    app.run(debug=True)
