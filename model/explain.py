import shap
import pickle
import pandas as pd

model = pickle.load(open("model/model.pkl", "rb"))

def get_shap_values(input_data):
    df = pd.DataFrame([input_data])

    # Step 1: Transform data using pipeline
    transformed = model.named_steps['preprocessing'].transform(df)

    # Step 2: Get model
    rf_model = model.named_steps['classifier']

    # Step 3: SHAP Explainer
    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer.shap_values(transformed)

    return shap_values, transformed