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
df = pd.read_csv('merged_patient_data.csv')

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

# 3. Encode target variable
y = df['Scleroderma present'].map({'Yes': 1, 'No': 0})

# 4. Prepare features (X)
X = df[initial_features].copy()

# 5. Encode categorical variables and handle missing values
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Impute missing values as a new category (for categorical) or median (for numeric)
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# 7. Train classifier (favoring sensitivity)
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 8. Evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, y_proba))

# 9. Save model, imputer, and feature list
joblib.dump(clf, 'scleroderma_rf_model.joblib')
joblib.dump(imputer, 'scleroderma_imputer.joblib')
joblib.dump(initial_features, 'scleroderma_feature_columns.joblib')

# 10. SHAP for test recommendation
def recommend_tests(patient_dict, model, imputer, feature_list, top_n=3):
    # patient_dict: dict of {feature: value} for available data
    # missing features will be imputed
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
