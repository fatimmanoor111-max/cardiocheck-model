# CardioCheck - Heart Disease Predictor

Flask web app + Logistic Regression model trained on the UCI Heart Disease dataset (`heart.csv`).

## Run locally
```bash
python -m venv venv
venv\Scripts\activate.bat    # Windows cmd  |  source venv/bin/activate  (Mac/Linux)
pip install -r requirements.txt
python train_model.py        # creates model.pkl + metrics.json
python app.py                # http://127.0.0.1:8000
```

## Deploy on Render
1. Push this folder to a GitHub repo.
2. Render > **New > Blueprint** (uses `render.yaml`), or **New > Web Service** with:
   - Build Command: `pip install -r requirements.txt && python train_model.py`
   - Start Command: `gunicorn app:app`
3. Deploy. The model is trained during build, so it always matches the server's scikit-learn version.

*For educational use only; not a medical diagnosis.*
