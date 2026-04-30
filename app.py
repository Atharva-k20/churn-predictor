from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model         = joblib.load('model.pkl')
scaler        = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

THRESHOLD = 0.35

@app.route('/predict', methods=['POST'])
def predict():
    data  = request.get_json()
    df    = pd.DataFrame([data])
    df    = df.reindex(columns=feature_names, fill_value=0)
    X_sc  = scaler.transform(df)
    prob  = model.predict_proba(X_sc)[0][1]
    return jsonify({
        'churn_probability': round(float(prob), 4),
        'prediction':        'CHURN' if prob > THRESHOLD else 'STAY'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)