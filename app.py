import os
# import asyncio
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
# from googletrans import Translator
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

async def translate_text(text):
    async with Translator() as translator:
        result = await translator.translate(text, src="vi", dest="en")     
        return result.text


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

        translated_num_cols = {}
        translated_cat_cols = {}
        # Translate column names while keeping structure
        # for key in num_cols.keys():
        #     translated_key = asyncio.run(translate_text(key))
        #     translated_num_cols[translated_key] = num_cols[key] 
        # for key in cat_cols.keys():
        #     translated_key = asyncio.run(translate_text(key))
        #     translated_cat_cols[translated_key] = cat_cols[key] 

        # dataset = {"num_cols": translated_num_cols, "cat_cols": translated_cat_cols, "df": df.to_dict(orient='list')}

        dataset = {"num_cols": num_cols, "cat_cols": cat_cols, "df": df.to_dict(orient='list')}
        
        # Start training in a separate thread
        threading.Thread(target=train_model, args=(df, num_cols, cat_cols)).start()

        return jsonify({"num_cols": num_cols, "cat_cols": cat_cols, "message": "Dataset processed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/upload", methods=["POST"])
def upload():
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

        # translated_num_cols = {}
        # translated_cat_cols = {}
        
        # Translate column names while keeping structure
        # for key in num_cols.keys(): # comment if use vietnamese
        #     translated_key = asyncio.run(translate_text(key))
        #     print(f"DEBUG: key={key}, translated={translated_key}")
        #     translated_num_cols[translated_key] = num_cols[key] 
        #     print("Done")
        # print("HELELO")
        # for key in cat_cols.keys(): # comment if use vietnamese
        #     translated_key = asyncio.run(translate_text(key))
        #     print(f"DEBUG: key={key}, translated={translated_key}")
        #     translated_cat_cols[translated_key] = cat_cols[key] 
        #     print("Done")

        # dataset = {"num_cols": translated_num_cols, "cat_cols": translated_cat_cols, "df": df.to_dict(orient='list')}

        # return jsonify({"num_cols": translated_num_cols, "cat_cols": translated_cat_cols, "message": "Dataset processed successfully!"}) # changed to num_cols and cat_cols if use vietnamese
        
        dataset = {"num_cols": num_cols, "cat_cols": cat_cols, "df": df.to_dict(orient='list')}

        return jsonify({"num_cols": num_cols, "cat_cols": cat_cols, "message": "Dataset processed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/train", methods=["POST"])
def train():
    global encoders, scaler
    try:
        if dataset is None:
            return jsonify({"error": "No dataset uploaded yet"})

        df = pd.DataFrame(dataset["df"])
        num_cols = dataset["num_cols"]
        cat_cols = dataset["cat_cols"]
        train_model(df, num_cols, cat_cols)
        return jsonify({"message": "Training completed successfully!"})
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
    global dataset, test_data
    try:
        if not os.path.exists("best_model.pkl"):
            return jsonify({"error": "Model not trained yet"})

        with open("best_model.pkl", "rb") as f:
            model, encoders, scaler, original_columns = pickle.load(f)
        print("Model:")
        print(model)
        data = request.json
        print(f"DATA: {data}")
        df = pd.DataFrame([data])
        print(df)
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
        print(f"DEBUG: df")
        print(df)
        prediction = model.predict(df)
        print(f"Prediction: {prediction[0]}")
        return jsonify({"prediction": round(float(prediction[0]), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    socketio.run(app, debug=True)
