import pickle
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values    
cat_feat_names = ['Quận', 'Huyện', 'Loại hình nhà ở', 'Giấy tờ pháp lý'] 
num_feat_names = ['Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài (m)', 'Rộng (m)'] 

# Pipeline for categorical features:
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)),
    ('cat_encoder', OneHotEncoder()) ])    


# Pipeline for numerical features:
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)),  
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) ])  
  
# Combine features transformed by two above pipelines:
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  

# Load the preprocessing pipeline and trained model
with open("full_pipeline.pkl", "rb") as f:
    full_pipeline = joblib.load(f)

### Preprocessing dataset and fitting model
raw_data = pd.read_csv('dataset/VN_housing_dataset_processed1.csv')
# Define the target column (adjust 'target_col' to match the actual label column in your dataset)
target_col = "Giá (triệu/m2)"  # Change this to the actual target variable

# Split features (X) and target labels (y)
X = raw_data.drop(columns=[target_col])  # Features
y = raw_data[target_col]  # Labels

# Create train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing pipeline
X_train_processed = full_pipeline.fit_transform(X_train)
X_test_processed = full_pipeline.transform(X_test)
#################

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()

        # Extract input values
        area = float(data["area"])
        bedrooms = int(data["bedrooms"])
        floors = int(data["floors"])
        length = float(data["length"])
        width = float(data["width"])
        quan = data["quan"]
        huyen = data["huyen"]
        legal = data.get("legal", None)  # Can be null
        typ = data.get("type", None)

        # Create a DataFrame for the input
        input_data = pd.DataFrame([[quan, huyen, typ, legal, floors, bedrooms, area, length, width]],
                                  columns=["Quận", "Huyện", "Loại hình nhà ở", "Giấy tờ pháp lý", "Số tầng", "Số phòng ngủ", "Diện tích", "Dài (m)", "Rộng (m)"])

        # Preprocess input
        processed_input = full_pipeline.transform(input_data)

        # load model
        with open("best_model.pkl", "rb") as f:
            model = joblib.load(f)
            model.fit(X_train_processed, y_train)
    
        # Predict price
        prediction = model.predict(processed_input)[0] * 1000 # account for thousands of VND unit

        # Return JSON response
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/train", methods=["POST"])
def train():
    try:
        file = request.files["file"]
        if not file:
            return jsonify({"error": "No file uploaded"})
        
        df = pd.read_csv(file)
        if "Giá (triệu/m2)" not in df.columns:
            return jsonify({"error": "Dataset must contain 'Giá (triệu/m2)' column as the target variable"})
        
        # Identify column types
        num_cols = []
        cat_cols = []
        
        for col in df.columns:
            if col == "Giá (triệu/m2)":
                continue
            num_count = df[col].apply(lambda x: isinstance(x, (int, float))).sum()
            str_count = df[col].apply(lambda x: isinstance(x, str)).sum()
            
            if num_count > str_count:
                df = df[pd.to_numeric(df[col], errors='coerce').notna()]
                num_cols.append(col)
            else:
                df = df[df[col].apply(lambda x: isinstance(x, str))]
                cat_cols.append(col)
        
        # Encode categorical data
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
        
        # Normalize numerical data
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        
        # Split dataset
        X = df.drop(columns=["Giá (triệu/m2)"])
        y = df["Giá (triệu/m2)"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost Hyperparameter Tuning
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        model = xgb.XGBRegressor()
        search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=20, verbose=1, n_jobs=-1)
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        with open("best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        
        return jsonify({"message": "Training complete! Best model saved as best_model.pkl"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    socketio.run(app, debug=True)

