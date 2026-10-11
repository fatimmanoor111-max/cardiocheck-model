import json
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "model.pkl"), "rb") as f:
    bundle = pickle.load(f)
model, FEATURES = bundle["model"], bundle["feature_columns"]

try:
    with open(os.path.join(BASE, "metrics.json")) as f:
        METRICS = json.load(f)
except FileNotFoundError:
    METRICS = {}

# field: (min, max, allowed integer values or None)
RULES = {
    "age": (1, 120), "sex": (0, 1), "cp": (0, 3), "trestbps": (60, 260),
    "chol": (80, 700), "fbs": (0, 1), "restecg": (0, 2), "thalach": (50, 250),
    "exang": (0, 1), "oldpeak": (0, 10), "slope": (0, 2), "ca": (0, 4), "thal": (0, 3),
}
LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting blood pressure",
    "chol": "Cholesterol", "fbs": "Fasting blood sugar", "restecg": "Resting ECG",
    "thalach": "Max heart rate", "exang": "Exercise-induced angina", "oldpeak": "ST depression",
    "slope": "ST slope", "ca": "Major vessels", "thal": "Thalassemia",
}


def build_row(raw):
    """Same encoding as training (one-hot, drop_first) built by hand for a single row."""
    row = {}
    for col in FEATURES:
        if col in raw:
            row[col] = raw[col]
        else:
            base, val = col.rsplit("_", 1)
            row[col] = int(raw[base] == int(val))
    return pd.DataFrame([row], columns=FEATURES)


@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form
    raw = {}
    for key, (lo, hi) in RULES.items():
        try:
            val = float(data.get(key))
        except (TypeError, ValueError):
            return jsonify(error=f"Please enter a valid value for {LABELS[key]}."), 400
        if not lo <= val <= hi:
            return jsonify(error=f"{LABELS[key]} must be between {lo} and {hi}."), 400
        raw[key] = val

    X = build_row(raw)
    prob = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    level = "high" if prob >= 0.66 else "moderate" if prob >= 0.34 else "low"
    return jsonify(prediction=pred, probability=round(prob * 100, 1), level=level)


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(debug=True, port=8000)
