"""Trains the Logistic Regression model on heart.csv and saves model.pkl (+ metrics).
Render runs this at build time so the pickle always matches the server's scikit-learn version."""
import json
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]

df = pd.read_csv("heart.csv").drop_duplicates().reset_index(drop=True)
df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True, dtype=int)
X, y = df.drop(columns="target"), df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

pipe = Pipeline([
    ("prep", ColumnTransformer([("scale", StandardScaler(), NUMERIC)], remainder="passthrough")),
    ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
])
grid = GridSearchCV(
    pipe,
    {"model__C": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
     "model__solver": ["lbfgs", "liblinear"]},
    cv=cv, scoring="f1", n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_

pred = best.predict(X_test)
proba = best.predict_proba(X_test)[:, 1]
metrics = {
    "model": "Logistic Regression",
    "accuracy": round(accuracy_score(y_test, pred), 3),
    "precision": round(precision_score(y_test, pred), 3),
    "recall": round(recall_score(y_test, pred), 3),
    "f1": round(f1_score(y_test, pred), 3),
    "auc": round(roc_auc_score(y_test, proba), 3),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
}

with open("model.pkl", "wb") as f:
    pickle.dump({"model": best, "feature_columns": list(X.columns)}, f)
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Best params:", grid.best_params_)
print(metrics)
