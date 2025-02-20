import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import threading
import optuna

def objective(trial, X_train, y_train, X_valid, y_valid):
    """Objective function for Optuna hyperparameter tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=True)

    preds = model.predict(X_valid)
    mse = np.mean((preds - y_valid) ** 2)
    return mse  # Optuna minimizes this value

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

dataset = None  # Store dataset metadata
encoders = {}  # Store encoders for categorical features
scaler = None  # Store scaler for numerical features
original_columns = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_and_train", methods=["POST"])
def upload_and_train():
    global dataset, encoders, scaler, original_columns
    try:
        file = request.files["file"]
        if not file:
            return jsonify({"error": "No file uploaded"})

        df = pd.read_csv(file, dtype=str)  # Read all as strings to analyze
        original_columns = df.columns.tolist()
        # Identify column types
        num_cols = {}
        cat_cols = {}

        for col in df.columns[:-1]:  # Ignore last column (target variable)
            df[col] = df[col].replace(["nan", "NaN"], np.nan)  # Handle text null values

            # Attempt to convert to numeric
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            num_valid = numeric_values.notna().sum()
            str_valid = df[col].notna().sum() - num_valid  # Non-numeric values

            if num_valid > str_valid:
                df[col] = df[col].str.replace(',', '.', regex=True)  # Normalize decimal separators
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert column to numeric
                has_decimal = df[col].dropna().apply(lambda x: isinstance(x, float) and not x.is_integer()).any()

                if has_decimal:
                    num_cols[col] = "decimal"
                else:
                    num_cols[col] = "integer"
            else:
                cat_cols[col] = sorted(df[col].dropna().unique().tolist())

        dataset = {"num_cols": num_cols, "cat_cols": cat_cols, "df": df.to_dict(orient='list')}

        # Start training in a separate thread
        threading.Thread(target=train_model, args=(df, num_cols, cat_cols)).start()

        return jsonify({"num_cols": num_cols, "cat_cols": cat_cols, "message": "Training started!"})
    except Exception as e:
        return jsonify({"error": str(e)})


def train_model(df, num_cols, cat_cols):
    """Background training process"""
    global encoders, scaler, test_data
    try:
        target_col = df.columns[-1]  # Last column is target variable

        # Encode categorical data
        encoders = {}
        for col in cat_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder

        # Normalize numerical data
        scaler = StandardScaler()
        df[list(num_cols.keys())] = scaler.fit_transform(df[list(num_cols.keys())])

        # Split dataset
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Optimize hyperparameters using Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=300)

        # Train best model
        best_params = study.best_params
        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        # # XGBoost Hyperparameter Tuning
        # param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'max_depth': [3, 5, 10],
        #     'learning_rate': [0.01, 0.1, 0.2],
        #     'subsample': [0.8, 1.0]
        # }

        # xgb_model = xgb.XGBRegressor()
        # search = RandomizedSearchCV(xgb_model, param_grid, cv=3, n_iter=10, verbose=2, n_jobs=-1)
        # search.fit(X_train, y_train)

        # best_model = search.best_estimator_
        # print("Best model:")
        # print(best_model)
        with open("best_model.pkl", "wb") as f:
            pickle.dump((best_model, encoders, scaler, original_columns), f)

        print("Training completed! Model saved as best_model.pkl")
    except Exception as e:
        print(f"Training failed: {str(e)}")


@app.route("/predict", methods=["POST"])
def predict():
    global dataset, test_data
    try:
        if not os.path.exists("best_model.pkl"):
            return jsonify({"error": "Model not trained yet"})

        with open("best_model.pkl", "rb") as f:
            model, encoders, scaler, original_columns = pickle.load(f)
        print("Model:")
        print(model)
        data = request.json
        df = pd.DataFrame([data])

        # Convert numerical inputs
        for col in dataset["num_cols"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Encode categorical inputs using the trained encoders
        for col in dataset["cat_cols"]:
            if col in encoders:
                df[col] = encoders[col].transform(df[col].astype(str))
            else:
                return jsonify({"error": f"Unexpected categorical value in {col}"})

        # Standardize numerical inputs
        df[list(dataset["num_cols"].keys())] = scaler.transform(df[list(dataset["num_cols"].keys())])
        
        # Ensure correct column order
        df = df[original_columns[:-1]]  # Exclude target column
        prediction = model.predict(df)
        print(f"Prediction: {prediction[0]}")
        return jsonify({"prediction": round(float(prediction[0]), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    socketio.run(app, debug=True)
