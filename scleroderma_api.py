from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import spacy
import uvicorn
from typing import List, Optional

# Load spaCy model for NLP (using en_core_web_sm, replace with scispaCy for medical terms if needed)
nlp = spacy.load('en_core_web_sm')

# Load model and preprocessing objects
clf = joblib.load('scleroderma_rf_model.joblib')
imputer = joblib.load('scleroderma_imputer.joblib')
feature_columns = joblib.load('scleroderma_feature_columns.joblib')

import shap

def recommend_tests(patient_dict, model, imputer, feature_list, top_n=3):
    x = []
    for f in feature_list:
        x.append(patient_dict.get(f, np.nan))
    x_df = pd.DataFrame([x], columns=feature_list)
    x_imp = pd.DataFrame(imputer.transform(x_df), columns=feature_list)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_imp)
    missing = [f for f in feature_list if patient_dict.get(f, np.nan) in [None, np.nan, '', 0]]
    # Robustly handle both binary and single-class SHAP output
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_abs = np.abs(shap_values[1][0])
    else:
        shap_abs = np.abs(shap_values[0][0]) if isinstance(shap_values[0], np.ndarray) else np.abs(shap_values[0])
    ranked = sorted([(f, shap_abs[i]) for i, f in enumerate(feature_list) if f in missing], key=lambda x: x[1], reverse=True)
    top1 = ranked[0][0] if ranked else None
    top3 = [f for f, _ in ranked[:top_n]]
    return top1, top3

# Map simple symptom keywords to feature columns (expand as needed)
symptom_to_feature = {
    'swollen joints': 'Swollen joints',
    'skin thickening': 'Skin thickening of the whole finger distal to MCP (Sclerodactyly)',
    'esr': 'DAS 28 (ESR, calculated)',
    'crp': 'DAS 28 (CRP, calculated)',
    'body weight': 'Body weight (kg)',
    'bnp': 'BNP (pg/ml)',
    'ntprobnp': 'NTproBNP (pg/ml)',
    'fvc': 'Forced Vital Capacity (FVC - % predicted)',
    'dlco': 'DLCO/SB (% predicted)',
    # Add more mappings as needed
}

def extract_features_from_text(text: str) -> dict:
    """Extracts relevant features from free-text using simple keyword matching."""
    doc = nlp(text.lower())
    features = {col: None for col in feature_columns}
    for keyword, feature in symptom_to_feature.items():
        if keyword in text.lower():
            features[feature] = 1  # Mark as present
    # Optionally, extract numbers for lab values with regex
    return features

app = FastAPI()

class PredictRequest(BaseModel):
    text: Optional[str] = None
    features: Optional[dict] = None

@app.post('/predict')
def predict(request: PredictRequest):
    # 1. Extract features from text or use provided features
    if request.text:
        features = extract_features_from_text(request.text)
    elif request.features:
        features = {col: request.features.get(col, None) for col in feature_columns}
    else:
        return JSONResponse({'error': 'No input provided'}, status_code=400)

    # 2. Create DataFrame and preprocess
    X_input = pd.DataFrame([features])
    for col in feature_columns:
        if col not in X_input:
            X_input[col] = None
    X_input = X_input[feature_columns]
    X_input = X_input.astype(object).replace({None: np.nan})
    X_input = pd.DataFrame(imputer.transform(X_input), columns=feature_columns)

    # 3. Predict
    proba = clf.predict_proba(X_input)[0, 1]
    prediction = int(proba > 0.5)

    # 4. SHAP-based test recommendation
    top1, top3 = recommend_tests(features, clf, imputer, feature_columns)

    return {
        'scleroderma_probability': float(proba),
        'prediction': 'Likely Scleroderma' if prediction else 'Unlikely Scleroderma',
        'top_1_recommended_test': top1,
        'top_3_recommended_tests': top3
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
