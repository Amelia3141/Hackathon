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
imputer = joblib.load('/Users/ameliag/Downloads/HACKATHON/scleroderma_imputer.joblib')
feature_columns = joblib.load('/Users/ameliag/Downloads/HACKATHON/scleroderma_feature_columns.joblib')

# Top features for test recommendation (from previous training)
top_features = [
    'DAS 28 (ESR, calculated)', 'Swollen joints', 'Modified Rodnan Skin Score, only imported value',
    'Body weight (kg)', 'Left ventricular ejection fraction (%)', 'NTproBNP (pg/ml)', 'BNP (pg/ml)',
    'DAS 28 (CRP, calculated)', 'TAPSE: tricuspid annular plane systolic excursion in cm',
    'Pulmonary wedge pressure (mmHg)', 'Forced Vital Capacity (FVC - % predicted)', 'DLCOc/VA (% predicted)',
    'Right ventricular area (cm2) (right ventricular dilation)', 'Tricuspid regurgitation velocity (m/sec)',
    '6 Minute walk test (distance in m)', 'PAPsys (mmHg)', 'Pulmonary resistance (dyn.s.cm-5)',
    'DLCO/SB (% predicted)', 'Skin thickening of the whole finger distal to MCP (Sclerodactyly)',
    'Skin thickening of the fingers of both hands extending proximal to the MCP joints'
]

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
    # Fill missing columns if any
    for col in feature_columns:
        if col not in X_input:
            X_input[col] = None
    # Encode categorical (already numeric in this simple approach)
    X_input = X_input[feature_columns]
    X_input = X_input.astype(object).replace({None: np.nan})
    X_input = pd.DataFrame(imputer.transform(X_input), columns=feature_columns)

    # 3. Predict
    proba = clf.predict_proba(X_input)[0, 1]  # Probability of scleroderma
    prediction = int(proba > 0.5)

    # 4. Recommend tests (top features that are missing or 0)
    missing_tests = [feat for feat in top_features if (features.get(feat) in [None, 0])]
    recommended_tests = missing_tests[:5]  # Recommend up to 5

    return {
        'scleroderma_probability': float(proba),
        'prediction': 'Likely Scleroderma' if prediction else 'Unlikely Scleroderma',
        'recommended_tests': recommended_tests
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
