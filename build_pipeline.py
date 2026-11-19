"""
Train the Crop Recommendation pipeline and save it for the Streamlit app.

Run from the project folder:
    python build_pipeline.py

Output:
    crop_pipeline.joblib  (pipeline + feature order + input ranges + sklearn version)
"""

import sys

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = "Crop_recommendation.csv"   # must match the file name in your repo exactly
OUTPUT_PATH = "crop_pipeline.joblib"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
RANDOM_STATE = 42


def make_pipeline() -> Pipeline:
    """Random Forest wrapped so the feature order is enforced by column NAME."""
    selector = ColumnTransformer(
        [("features", "passthrough", FEATURES)],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        criterion="entropy",
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("select", selector), ("model", model)])


def main() -> None:
    # ---- 1. Load and validate data -------------------------------------
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        sys.exit(f"Could not find '{DATA_PATH}'. Put the CSV in this folder "
                 f"and check the capitalisation of the file name.")

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        sys.exit(f"Missing columns in the CSV: {missing_cols}")

    if df[FEATURES + [TARGET]].isnull().any().any():
        print("Warning: missing values found, dropping those rows.")
        df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]          # crop names stay as text, so no LabelEncoder is needed
    print(f"Loaded {len(df)} rows, {y.nunique()} crops.")

    # ---- 2. Honest evaluation on a held-out split ----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    eval_pipe = make_pipeline().fit(X_train, y_train)
    test_pred = eval_pipe.predict(X_test)
    print(f"\nHold-out test accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(classification_report(y_test, test_pred))

    # ---- 3. 5-fold stratified cross-validation -------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(make_pipeline(), X, y, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    # ---- 4. Final model: refit on ALL data for deployment --------------
    final_pipe = make_pipeline().fit(X, y)

    bundle = {
        "pipeline": final_pipe,
        "features": FEATURES,
        "ranges": {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES},
        "defaults": {f: float(df[f].median()) for f in FEATURES},
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, OUTPUT_PATH)
    print(f"\nSaved pipeline to '{OUTPUT_PATH}' (scikit-learn {sklearn.__version__}).")


if __name__ == "__main__":
    main()
