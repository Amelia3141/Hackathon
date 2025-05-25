from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import spacy
import uvicorn
from typing import List, Optional
import pickle
from datetime import datetime

# Load spaCy model for NLP (using en_core_web_sm, replace with scispaCy for medical terms if needed)
nlp = spacy.load('en_core_web_sm')

# Load GARNN model and preprocessing objects
import torch
from garnn_model import GARNN

garnn_model = torch.load('garnn_best_model.pt', map_location=torch.device('cpu'))
garnn_model.eval()
# TODO: load scaler, feature graph, and other preprocessing as needed
imputer = joblib.load('scleroderma_imputer.joblib')
feature_columns = joblib.load('scleroderma_feature_columns.joblib')
# If you want to keep RF for fallback, keep the following line commented:
# clf = joblib.load('scleroderma_rf_model.joblib')

# Load patient demographic/clinical data
try:
    with open('pats', 'rb') as f:
        pats_df = pickle.load(f)
    if not isinstance(pats_df, pd.DataFrame):
        pats_df = pd.DataFrame(pats_df)
except Exception as e:
    pats_df = None
    print('Could not load pats:', e)

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
    # Map SHAP values to features by name for robust matching
    shap_feature_names = x_imp.columns.tolist()
    shap_value_map = dict(zip(shap_feature_names, shap_abs))
    ranked = sorted(
        [(f, shap_value_map.get(f, 0)) for f in missing if f in shap_feature_names],
        key=lambda x: x[1],
        reverse=True
    )
    top1 = ranked[0][0] if ranked else None
    top3 = [f for f, _ in ranked[:top_n]]

    # --- Expanded antibody logic per PMC6546089 ---
    # Mapping: antibody -> (short name, clinical association)
    antibody_info = [
        ("Anti-centromere antibody (ACA)", "Limited cutaneous/CREST, lower cancer risk, digital ischemia risk (if IFI16+)"),
        ("Anti-topoisomerase I antibody (Scl-70)", "Diffuse cutaneous, ILD risk"),
        ("Anti-RNA polymerase III antibody", "Cancer risk (esp. breast/lung), rapid skin progression"),
        ("Anti-U1RNP antibody", "Overlap syndromes, myositis, ILD"),
        ("Anti-Th/To antibody (Rpp25)", "Limited cutaneous, PAH, ILD, digital ischemia"),
        ("Anti-PM/Scl antibody", "Myositis overlap, ILD"),
        ("Anti-Ku antibody", "Myositis overlap"),
        ("Anti-fibrillarin antibody (U3-RNP)", "Diffuse cutaneous, PAH, GI involvement, cancer risk"),
        ("Anti-RNPC3 antibody", "Cancer-associated, ILD, GI dysmotility, myopathy"),
        ("Anti-eIF2B antibody", "Diffuse cutaneous, ILD"),
        ("Anti-RuvBL1/2 antibody", "Diffuse cutaneous, myositis overlap, older onset"),
        ("Anti-BICD2 antibody", "ILD, myositis, often with ACA"),
        ("Anti-IFI16 antibody", "Digital ischemia/gangrene, vascular risk"),
        ("Anti-AT1R antibody", "Vascular disease, PAH, digital ischemia"),
        ("Anti-ETAR antibody", "Vascular disease, PAH, digital ischemia"),
        ("Anti-M3R antibody", "GI dysmotility, autonomic dysfunction"),
        ("Anti-PDGFR antibody", "Possible profibrotic role, controversial"),
        ("ANA", "General screening for autoimmune disease")
    ]
    # Feature/phenotype triggers for each antibody
    antibody_triggers = {
        "Anti-centromere antibody (ACA)": ['crest', 'limited', 'telangiectasia', 'digital ischemia', 'calcinosis', 'sclerodactyly', 'raynaud'],
        "Anti-topoisomerase I antibody (Scl-70)": ['diffuse', 'ild', 'lung fibrosis', 'rapid skin'],
        "Anti-RNA polymerase III antibody": ['malignancy', 'cancer', 'rapid skin', 'diffuse'],
        "Anti-U1RNP antibody": ['overlap', 'myositis', 'ild'],
        "Anti-Th/To antibody (Rpp25)": ['limited', 'pa', 'pulmonary hypertension', 'digital ischemia', 'ild'],
        "Anti-PM/Scl antibody": ['myositis', 'overlap', 'ild'],
        "Anti-Ku antibody": ['myositis', 'overlap'],
        "Anti-fibrillarin antibody (U3-RNP)": ['diffuse', 'gi', 'pa', 'cancer'],
        "Anti-RNPC3 antibody": ['cancer', 'malignancy', 'ild', 'gi', 'myopathy'],
        "Anti-eIF2B antibody": ['diffuse', 'ild'],
        "Anti-RuvBL1/2 antibody": ['diffuse', 'myositis', 'older onset'],
        "Anti-BICD2 antibody": ['ild', 'myositis', 'aca'],
        "Anti-IFI16 antibody": ['digital ischemia', 'gangrene', 'vascular'],
        "Anti-AT1R antibody": ['vascular', 'pa', 'digital ischemia'],
        "Anti-ETAR antibody": ['vascular', 'pa', 'digital ischemia'],
        "Anti-M3R antibody": ['gi', 'dysmotility', 'autonomic'],
        "Anti-PDGFR antibody": ['fibrosis', 'profibrotic'],
    }
    # Collect features present
    text_features = [str(k).lower() for k, v in patient_dict.items() if v not in [None, np.nan, '', 0]]
    antibody_suggestions = []
    for ab, triggers in antibody_triggers.items():
        if any(any(trigger in tf for trigger in triggers) for tf in text_features):
            antibody_suggestions.append(ab)
    # Always recommend ANA if not present
    if 'ANA' not in antibody_suggestions:
        antibody_suggestions.insert(0, 'ANA')
    # If no specific suggestion, fallback to ANA
    if not antibody_suggestions:
        antibody_suggestions = ['ANA']
    # Attach clinical associations
    ab_assoc_map = {ab: assoc for ab, assoc in antibody_info}
    antibody_suggestions = [(ab, ab_assoc_map.get(ab, "")) for ab in antibody_suggestions]
    return top1, top3, antibody_suggestions

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

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def root():
    return FileResponse("index.html")

class PredictRequest(BaseModel):
    text: Optional[str] = None
    features: Optional[dict] = None
    patient_id: Optional[str] = None  # NEW: Optional patient_id field

@app.post('/predict')
def predict(request: PredictRequest):
    # 1. Extract features from text or use provided features
    if request.text:
        features = extract_features_from_text(request.text)
    elif request.features:
        features = {col: request.features.get(col, None) for col in feature_columns}
    else:
        return JSONResponse({'error': 'No input provided'}, status_code=400)

    # 1b. If patient_id is provided and pats_df is loaded, merge demographic/clinical info
    # This improves accuracy by using real patient features (age, sex, subset, etc.)
    if hasattr(request, 'patient_id') and request.patient_id and pats_df is not None:
        pats_row = pats_df[pats_df['Id Patient V2018'] == request.patient_id]
        if not pats_row.empty:
            row = pats_row.iloc[0]
            # Example: Add sex, subset, race, age at onset, disease duration, etc.
            features['Sex'] = 1 if str(row.get('Sex','')).lower().startswith('f') else 0 if str(row.get('Sex','')).lower().startswith('m') else None
            features['Subset_limited'] = 1 if 'limited' in str(row.get('Subsets of SSc according to LeRoy (1988)','')).lower() else 0
            features['Subset_diffuse'] = 1 if 'diffuse' in str(row.get('Subsets of SSc according to LeRoy (1988)','')).lower() else 0
            # Calculate age at onset
            try:
                dob = pd.to_datetime(row.get('Date of birth', None), errors='coerce')
                onset = pd.to_datetime(row.get('Onset of first non-Raynaud?s of the disease', None), errors='coerce')
                if pd.notnull(dob) and pd.notnull(onset):
                    features['Age_at_onset'] = (onset - dob).days // 365
                    features['Disease_duration'] = (datetime.now() - onset).days // 365
                else:
                    features['Age_at_onset'] = None
                    features['Disease_duration'] = None
            except Exception as e:
                features['Age_at_onset'] = None
                features['Disease_duration'] = None
            # Race/ethnicity flags
            features['Race_white'] = 1 if str(row.get('Race white','')).strip().lower() == 'white' else 0
            features['Race_asian'] = 1 if pd.notnull(row.get('Race asian')) else 0
            features['Race_black'] = 1 if pd.notnull(row.get('Race black')) else 0
            features['Hispanic'] = 1 if pd.notnull(row.get('Hispanic')) else 0
            # Height if available
            try:
                features['Height'] = float(row.get('Height')) if pd.notnull(row.get('Height')) else None
            except:
                features['Height'] = None
            # Add more fields as needed
    
    # 2. Create DataFrame and preprocess
    x_df = pd.DataFrame([features], columns=feature_columns)
    x_imp = pd.DataFrame(imputer.transform(x_df), columns=feature_columns)

    # 3. Prepare input for GARNN
    # Assume single timepoint, shape (1, 1, num_features)
    x_tensor = torch.tensor(x_imp.values, dtype=torch.float32).unsqueeze(0)
    # Dummy patient and feature graph indices (replace with real if available)
    patient_edge_index = None
    feature_edge_index = garnn_model.feature_graph_edge_index if hasattr(garnn_model, 'feature_graph_edge_index') else None
    import logging
    with torch.no_grad():
        logits = garnn_model(x_tensor, patient_edge_index, feature_edge_index)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        try:
            subtype_idx = int(np.argmax(probs))
            subtype_proba = float(probs[subtype_idx])
            subtype_labels = ['none', 'limited', 'diffuse', 'overlap']  # Update if your label order differs
            predicted_subtype = subtype_labels[subtype_idx] if subtype_idx < len(subtype_labels) else f'unknown_{subtype_idx}'
        except Exception as e:
            logging.error(f"Subtype prediction error: {e}; probs={probs}")
            predicted_subtype = 'error'
            subtype_proba = float('nan')
        proba = float(np.sum(probs[1:]))  # Probability of any scleroderma (not 'none')
    # Use best threshold from training (0.2)
    prediction = int(proba > 0.2)

    # 4. Recommend tests (keep using RF for SHAP, or switch to GARNN if available)
    # For now, fallback to RF for SHAP/test rec if needed
    clf = joblib.load('scleroderma_rf_model.joblib')
    top1, top3, antibody_suggestions = recommend_tests(features, clf, imputer, feature_columns)

    # 5. SHAP explainability (using RF explainer for now)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_imp)
    shap_feature_importance = list(zip(feature_columns, np.abs(shap_values[1][0]) if isinstance(shap_values, list) else np.abs(shap_values[0])))
    shap_feature_importance.sort(key=lambda x: -x[1])
    top_shap_features = [f for f, v in shap_feature_importance[:5]]
    top_shap_values = [float(v) for _, v in shap_feature_importance[:5]]
    shap_feature_names = [f for f, v in shap_feature_importance]
    shap_abs = [float(v) for f, v in shap_feature_importance]

    return {
        'scleroderma_probability': float(proba),
        'prediction': 'Likely Scleroderma' if prediction else 'Unlikely Scleroderma',
        'predicted_subtype': predicted_subtype,
        'subtype_probability': subtype_proba,
        'top_1_recommended_test': top1,
        'top_3_recommended_tests': top3,
        'recommended_antibody_tests': antibody_suggestions,
        'shap_feature_names': top_shap_features,
        'shap_feature_values': top_shap_values,
        'shap_all_features': shap_feature_names,
        'shap_all_values': [float(v) for v in shap_abs]
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
