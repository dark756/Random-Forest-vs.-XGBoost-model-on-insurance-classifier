import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("--- PHASE 1: Loading Saved Assets ---")

# Load Models
rf_model = joblib.load('rf_model.joblib')
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('xgb_model.json')

# Load Preprocessing Artifacts
le = joblib.load('label_encoder.joblib')
trained_columns = joblib.load('trained_columns.joblib')
imputation_dict = joblib.load('imputation_dict.joblib')

print("--- PHASE 2: Processing Data & Extracting Truth ---")

# Load your testing data
new_data = pd.read_csv('new_insurance_data_to_predict.csv')

# Drop Customer ID if it exists
if 'Customer' in new_data.columns:
    new_data = new_data.drop(columns=['Customer'])

# --- NEW: Extract the True Labels for Evaluation ---
target_col = 'Response' # Or 'Claim_Status'
if target_col not in new_data.columns:
    raise ValueError(f"To calculate accuracy, the '{target_col}' column must be in your test data!")

# Save the true labels and encode them using our saved label encoder
y_true_raw = new_data[target_col]
y_true = le.transform(y_true_raw)

# Now drop the target column from the features so the models can't cheat
new_data = new_data.drop(columns=[target_col])
# ---------------------------------------------------

# Apply medians and modes safely
for col, fill_value in imputation_dict.items():
    if col in new_data.columns: 
        new_data[col] = new_data[col].fillna(fill_value)

# One-Hot Encode and Align
new_data_encoded = pd.get_dummies(new_data, drop_first=True)
new_data_encoded = new_data_encoded.reindex(columns=trained_columns, fill_value=0)


print("--- PHASE 3: Generating Predictions ---")

# Generate raw numeric predictions
raw_preds_rf = rf_model.predict(new_data_encoded)
raw_preds_xgb = xgb_model.predict(new_data_encoded)


print("\n--- PHASE 4: COMPARATIVE ANALYSIS ---")

# Function to calculate and print metrics
def evaluate_model(name, y_actual, y_pred):
    acc = accuracy_score(y_actual, y_pred)
    # Using 'weighted' to account for class imbalance automatically
    prec = precision_score(y_actual, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_actual, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_actual, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{name} Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return f1 # We will use F1-Score as the ultimate tie-breaker

# Evaluate both
rf_f1 = evaluate_model("Random Forest", y_true, raw_preds_rf)
xgb_f1 = evaluate_model("XGBoost", y_true, raw_preds_xgb)

print("\n--- THE VERDICT ---")
if xgb_f1 > rf_f1:
    print("🏆 XGBoost is the better model for this dataset based on overall F1-Score.")
elif rf_f1 > xgb_f1:
    print("🏆 Random Forest is the better model for this dataset based on overall F1-Score.")
else:
    print("🤝 It's a dead tie. Both models performed equally well.")
    
# Optional: detailed breakdown if you want to see class-by-class
print("\nDetailed XGBoost Classification Report:")
print(classification_report(y_true, raw_preds_xgb, target_names=le.classes_, zero_division=0))
print("======================================================")
print("\nDetailed Random Forest Classification Report:")
print(classification_report(y_true, raw_preds_rf, target_names=le.classes_, zero_division=0))