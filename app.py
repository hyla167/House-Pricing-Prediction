import pickle
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder 

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values    
num_feat_names = ['DIỆN TÍCH - M2', 'SỐ PHÒNG', 'SỐ TOILETS'] 
cat_feat_names = ['QUẬN HUYỆN', 'HƯỚNG', 'GIẤY TỜ PHÁP LÝ'] 

# Pipeline for categorical features:
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)),
    ('cat_encoder', OneHotEncoder()) ])    

# Define MyFeatureAdder: a transformer for adding features "TỔNG SỐ PHÒNG",...  
class MyFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_TONG_SO_PHONG = True): 
        self.add_TONG_SO_PHONG = add_TONG_SO_PHONG
    def fit(self, feature_values, labels = None):
        return self   
    def transform(self, feature_values, labels = None):
        SO_PHONG_id, SO_TOILETS_id = 1, 2 
        TONG_SO_PHONG = feature_values[:, SO_PHONG_id] + feature_values[:, SO_TOILETS_id]
        if self.add_TONG_SO_PHONG:
            feature_values = np.c_[feature_values, TONG_SO_PHONG] 
        return feature_values

# Pipeline for numerical features:
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)),  
    ('attribs_adder', MyFeatureAdder(add_TONG_SO_PHONG = True)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) ])  
  
# Combine features transformed by two above pipelines:
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  

# Load the preprocessing pipeline and trained model
with open("full_pipeline.pkl", "rb") as f:
    full_pipeline = joblib.load(f)
with open("SOLUTION_model.pkl", "rb") as f:
    model = joblib.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    
    if request.method == "POST":
        # Get input values from form
        try:
            area = float(request.form["area"])
            rooms = int(request.form["rooms"])
            toilets = int(request.form["toilets"])
            district = request.form["district"]
            direction = request.form.get("direction", None)
            legal = request.form.get("legal", None)
            
            # Create a DataFrame for the input
            input_data = pd.DataFrame([[area, rooms, toilets, district, direction, legal]],
                                      columns=["DIỆN TÍCH - M2", "SỐ PHÒNG", "SỐ TOILETS", "QUẬN HUYỆN", "HƯỚNG", "GIẤY TỜ PHÁP LÝ"])
            
            # Preprocess input
            processed_input = full_pipeline.transform(input_data)
            
            # Predict price
            prediction = model.predict(processed_input)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
