import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import shap

# 1. Load data
import pickle
from datetime import datetime

df = pd.read_csv('merged_patient_data.csv')
# Load demographic/clinical info from pats.pkl
with open('pats', 'rb') as f:
    pats_df = pickle.load(f)
if not isinstance(pats_df, pd.DataFrame):
    pats_df = pd.DataFrame(pats_df)

# 2. Select initial presentation features (as discussed)
initial_features = [
    'Dyspnea (NYHA-stage)',
    'Worsening of cardiopulmonary manifestations within the last month',
    'Muscle weakness',
    'Proximal muscle weakness not explainable by other causes',
    'Muscle atrophy',
    'Myalgia',
    'Stomach symptoms (early satiety, vomiting)',
    'Intestinal symptoms (diarrhea, bloating, constipation)',
    'Skin thickening of the fingers of both hands extending proximal to the MCP joints',
    'Skin thickening of the whole finger distal to MCP (Sclerodactyly)',
    'Joint synovitis',
    'Tendon friction rubs',
    'Joint polyarthritis',
    'Swollen joints',
    'DAS 28 (ESR, calculated)',
    'DAS 28 (CRP, calculated)',
    'Auricular Arrhythmias',
    'Cardiac arrhythmias',
    'Renal crisis',
    'Worsening of skin within the last month',
    'Extent of skin involvement',
    'Modified Rodnan Skin Score, only imported value',
    'Body weight (kg)',
    'BNP (pg/ml)',
    'NTproBNP (pg/ml)',
    'Left ventricular ejection fraction (%)',
    'Pericardial effusion on echo',
    'Conduction blocks',
]

# 2b. Add demographic/clinical features to use for training
extra_features = [
    'Sex',
    'Height',
    'Race_white',
    'Race_asian',
    'Race_black',
    'Hispanic',
    'Subset_limited',
    'Subset_diffuse',
    'Age_at_onset',
    'Disease_duration',
]

# 3. Encode target variable
y = df['Scleroderma present'].map({'Yes': 1, 'No': 0})

# 4. Prepare features (X)
X = df[initial_features].copy()

# 4b. Merge in demographic/clinical features from pats_df using patient id
# Assume df and pats_df are aligned by row order or share a patient id column
# If not, join on patient id if available
if 'Id Patient V2018' in df.columns and 'Id Patient V2018' in pats_df.columns:
    pats_df = pats_df.set_index('Id Patient V2018')
    df = df.set_index('Id Patient V2018')
    merged = X.join(pats_df, how='left')
else:
    merged = pd.concat([X, pats_df], axis=1)

# Feature engineering: create binary flags and engineered features
merged['Sex'] = merged['Sex'].str.lower().map({'female': 1, 'male': 0})
merged['Subset_limited'] = merged['Subsets of SSc according to LeRoy (1988)'].str.lower().str.contains('limited', na=False).astype(int)
merged['Subset_diffuse'] = merged['Subsets of SSc according to LeRoy (1988)'].str.lower().str.contains('diffuse', na=False).astype(int)
merged['Race_white'] = (merged['Race white'].str.strip().str.lower() == 'white').astype(int)
merged['Race_asian'] = merged['Race asian'].notnull().astype(int)
merged['Race_black'] = merged['Race black'].notnull().astype(int)
merged['Hispanic'] = merged['Hispanic'].notnull().astype(int)
# Height as float
merged['Height'] = pd.to_numeric(merged['Height'], errors='coerce')
# Age at onset and disease duration
merged['Date of birth'] = pd.to_datetime(merged['Date of birth'], errors='coerce')
merged['Onset of first non-Raynaud?s of the disease'] = pd.to_datetime(merged['Onset of first non-Raynaud?s of the disease'], errors='coerce')
merged['Age_at_onset'] = (merged['Onset of first non-Raynaud?s of the disease'] - merged['Date of birth']).dt.days // 365
merged['Disease_duration'] = (datetime.now() - merged['Onset of first non-Raynaud?s of the disease']).dt.days // 365

# Restrict to top 15 features from RF for feature selection comparison
rf_top15 = [
    'Skin thickening of the whole finger distal to MCP (Sclerodactyly)',
    'Skin thickening of the fingers of both hands extending proximal to the MCP joints',
    'Renal crisis',
    'Worsening of skin within the last month',
    'Pericardial effusion on echo',
    'Auricular Arrhythmias',
    'Cardiac arrhythmias',
    'Joint synovitis',
    'Worsening of cardiopulmonary manifestations within the last month',
    'Conduction blocks',
    'Tendon friction rubs',
    'Muscle weakness',
    'Intestinal symptoms (diarrhea, bloating, constipation)',
    'Joint polyarthritis',
    'Proximal muscle weakness not explainable by other causes',
]
X = merged[rf_top15]

# 5. Encode categorical variables (no imputation)
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Impute missing values (all features are numeric after label encoding)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

# --- ADVANCED FEATURE ENGINEERING ---
# 1. Interaction terms between top features (manually selected for demonstration)
top_interactions = [
    ('Skin thickening of the whole finger distal to MCP (Sclerodactyly)', 'Myalgia'),
    ('Muscle weakness', 'Swollen joints'),
    ('DAS 28 (ESR, calculated)', 'DAS 28 (CRP, calculated)'),
]
for f1, f2 in top_interactions:
    if f1 in X_imputed.columns and f2 in X_imputed.columns:
        name = f'{f1}_x_{f2}'
        X_imputed[name] = X_imputed[f1] * X_imputed[f2]

# 2. Polynomial features for select numeric labs
from sklearn.preprocessing import PolynomialFeatures
poly_labs = ['BNP (pg/ml)', 'NTproBNP (pg/ml)', 'Body weight (kg)', 'DAS 28 (ESR, calculated)', 'DAS 28 (CRP, calculated)', 'Forced Vital Capacity (FVC - % predicted)', 'DLCO/SB (% predicted)']
poly_labs_present = [col for col in poly_labs if col in X_imputed.columns]
if len(poly_labs_present) > 0:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_feats = poly.fit_transform(X_imputed[poly_labs_present])
    poly_feat_names = poly.get_feature_names_out(poly_labs_present)
    poly_df = pd.DataFrame(poly_feats, columns=poly_feat_names, index=X_imputed.index)
    # Remove original columns to avoid duplication
    X_imputed = X_imputed.drop(columns=[f for f in poly_feat_names if f in X_imputed.columns], errors='ignore')
    X_imputed = pd.concat([X_imputed, poly_df], axis=1)

# Save feature columns and imputer for API
joblib.dump(list(X_imputed.columns), 'scleroderma_feature_columns.joblib')
joblib.dump(imp, 'scleroderma_imputer.joblib')

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# 7. Address class imbalance with SMOTE (oversample minority class in training set only)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 8. Try multiple models and compare ROC AUC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import StackingClassifier

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=(y_train_res.value_counts()[0] / y_train_res.value_counts()[1]),
                             eval_metric='logloss', use_label_encoder=False, random_state=42)
}

# Add stacking ensemble
stacking = StackingClassifier(
    estimators=[
        ('rf', models['RandomForest']),
        ('lr', models['LogisticRegression']),
        ('xgb', models['XGBoost'])
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    passthrough=True,
    n_jobs=-1
)
models['StackingEnsemble'] = stacking

best_auc = 0
best_model = None
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probs)
    print(f'--- {name} ---')
    print(classification_report(y_test, preds))
    print('ROC AUC:', auc)
    if name == 'RandomForest':
        # Print top 15 feature importances
        importances = model.feature_importances_
        feat_names = X_train_res.columns if hasattr(X_train_res, 'columns') else [f'f{i}' for i in range(len(importances))]
        sorted_idx = np.argsort(importances)[::-1]
        print('\nTop 15 Random Forest Features:')
        for i in sorted_idx[:15]:
            print(f'{feat_names[i]}: {importances[i]:.4f}')
        # Misclassification analysis
        print('\nRandom Forest Misclassification Analysis:')
        false_pos_idx = (preds == 1) & (y_test == 0)
        false_neg_idx = (preds == 0) & (y_test == 1)
        print(f'False Positives: {false_pos_idx.sum()}')
        print(f'False Negatives: {false_neg_idx.sum()}')
        # Try to summarize Sex and Has_Raynaud if available
        if "Sex" in X_test.columns:
            print('False Positives by Sex:')
            print(X_test.loc[false_pos_idx, "Sex"].value_counts(dropna=False))
            print('False Negatives by Sex:')
            print(X_test.loc[false_neg_idx, "Sex"].value_counts(dropna=False))
        if "Has_Raynaud" in X_test.columns:
            print('False Positives by Has_Raynaud:')
            print(X_test.loc[false_pos_idx, "Has_Raynaud"].value_counts(dropna=False))
            print('False Negatives by Has_Raynaud:')
            print(X_test.loc[false_neg_idx, "Has_Raynaud"].value_counts(dropna=False))
    if auc > best_auc:
        best_auc = auc
        best_model = model

# 9. Save best model and feature columns ONLY (no imputer)
joblib.dump(best_model, 'scleroderma_rf_model.joblib')
joblib.dump(list(X.columns), 'scleroderma_feature_columns.joblib')
print(f'Training complete. Best model: {type(best_model).__name__} with ROC AUC: {best_auc}')

# 10. SHAP for test recommendation
def recommend_tests(patient_dict, model, feature_list, top_n=3):
    # patient_dict: dict of {feature: value} for available data
    x = []
    for f in feature_list:
        x.append(patient_dict.get(f, np.nan))
    x_df = pd.DataFrame([x], columns=feature_list)
    x_imp = pd.DataFrame(imputer.transform(x_df), columns=feature_list)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_imp)
    # Rank missing features by absolute SHAP value
    missing = [f for f in feature_list if pd.isna(patient_dict.get(f, np.nan))]
    shap_abs = np.abs(shap_values[1][0])
    ranked = sorted([(f, shap_abs[i]) for i, f in enumerate(feature_list) if f in missing], key=lambda x: x[1], reverse=True)
    top1 = ranked[0][0] if ranked else None
    top3 = [f for f, _ in ranked[:top_n]]
    return top1, top3

# Example usage (with partial patient data):
# patient = {'Dyspnea (NYHA-stage)': 2, 'Myalgia': 'Yes'}
# top1, top3 = recommend_tests(patient, clf, imputer, initial_features)
# print('Top 1 recommended test:', top1)
# print('Top 3 recommended tests:', top3)

print('Training complete. Model and recommendation logic are ready.')
