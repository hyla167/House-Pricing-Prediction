import os
import asyncio
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import optuna

def objective(trial, X_train, y_train, X_valid, y_valid):
    """Objective function for Optuna hyperparameter tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=True)

    preds = model.predict(X_valid)
    return mean_squared_error(preds, y_valid, squared=False)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

dataset = None  # Store dataset metadata
encoders = {}  # Store encoders for categorical features
scaler = None  # Store scaler for numerical features
original_columns = []

def translate_fields(data):
    translation_map = {
        "district": "Quận",
        "ward": "Huyện",
        "house_type": "Loại hình nhà ở",
        "legal_papers": "Giấy tờ pháp lý",
        "floors": "Số tầng",
        "bedrooms": "Số phòng ngủ",
        "area": "Diện tích",
        "length": "Dài (m)",
        "width": "Rộng (m)"
    }
    return {translation_map[key]: value for key, value in data.items() if key in translation_map}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    global dataset, encoders, scaler, original_columns
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"})
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, dtype=str)
        elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            print("Hello")
            df = pd.read_excel(file, dtype=str)  # Read Excel file
            print("DEBUG")
            print(df.head())  # Print first few rows to debug
        # df = pd.read_csv(file, dtype=str)
        original_columns = df.columns.tolist()

        # Identify column types
        num_cols = {}
        cat_cols = {}

        for col in df.columns[:-1]:  # Ignore last column (target variable)
            df[col] = df[col].replace(["nan", "NaN"], np.nan)  # Handle text null values

            # Attempt to convert to numeric
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            num_valid = numeric_values.notna().sum()
            str_valid = df[col].notna().sum() - num_valid

            if num_valid > str_valid:
                df[col] = df[col].str.replace(',', '.', regex=True)  # Normalize decimal separators
                df[col] = pd.to_numeric(df[col], errors='coerce')
                has_decimal = df[col].dropna().apply(lambda x: isinstance(x, float) and not x.is_integer()).any()

                if has_decimal:
                    num_cols[col] = "decimal"
                else:
                    num_cols[col] = "integer"
            else:
                cat_cols[col] = sorted(df[col].dropna().unique().tolist())
                cat_cols[col].insert(0, "Không có thông tin")

        dataset = {"num_cols": num_cols, "cat_cols": cat_cols, "df": df.to_dict(orient='list')}

        train_model(df, num_cols, cat_cols)
        return jsonify({"num_cols": num_cols, "cat_cols": cat_cols, "message": "Training completed!"})
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
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=20)

        # Train best model
        best_params = study.best_params
        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        # # XGBoost Hyperparameter Tuning
        # param_grid = {
        #     'n_estimators': [150, 200, 250],
        #     'max_depth': [3, 5, 10],
        #     'learning_rate': [0.15, 0.2, 0.25],
        #     'subsample': [0.6, 0.8, 1.0],
        #     'colsample_bytree': [0.6, 0.8, 1.0]
        # }

        # xgb_model = xgb.XGBRegressor()
        # search = RandomizedSearchCV(xgb_model, param_grid, cv=3, n_iter=30, verbose=2, n_jobs=-1)
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
    try:
        if not os.path.exists("best_model.pkl"):
            return jsonify({"error": "Model not trained yet"})

        with open("best_model.pkl", "rb") as f:
            model, encoders, scaler, original_columns = pickle.load(f)

        data = request.json
        data = translate_fields(data)
        df = pd.DataFrame([data])
        # Define expected fields
        expected_fields = ["Quận", "Huyện", "Loại hình nhà ở", "Giấy tờ pháp lý", "Số tầng", "Số phòng ngủ", "Diện tích", "Dài (m)", "Rộng (m)"]
        df = df.reindex(columns=expected_fields)

        # Convert numerical inputs
        num_fields = ["Số tầng", "Số phòng ngủ", "Diện tích", "Dài (m)", "Rộng (m)"]
        for col in num_fields:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col in ["Diện tích", "Dài (m)", "Rộng (m)"]:
                df[col] = df[col].round(2)

        # Encode categorical inputs using trained encoders
        cat_fields = ["Quận", "Huyện", "Loại hình nhà ở", "Giấy tờ pháp lý"]
        for col in cat_fields:
            if col in encoders:
                df[col] = encoders[col].transform(df[col].astype(str))
            else:
                return jsonify({"error": f"Unexpected categorical value in {col}"})

        # Standardize numerical inputs
        df[num_fields] = scaler.transform(df[num_fields])
        print("DEBUG")
        print(df)
        prediction = model.predict(df)
        print(f"prediction: {prediction}")
        return jsonify({"predicted_price": round(float(prediction[0]), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    socketio.run(app, debug=True)
