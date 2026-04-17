import pandas as pd
import numpy as np
import joblib

# Scikit-Learn & Imblearn & XGBoost
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

print("--- PHASE 1: Data Loading & Preprocessing ---")

# 1. Load Data
df = pd.read_csv('AutoInsuranceClaims2024.csv')

# Optional but recommended: Drop ID columns like 'Customer' as they have no predictive power
if 'Customer' in df.columns:
    df = df.drop(columns=['Customer'])

# 2. Smart Imputation Strategy (Numeric vs String)
# We will save these calculated values to apply the exact same logic to future test data.
imputation_dict = {}

numeric_cols = df.select_dtypes(include=[np.number]).columns
string_cols = df.select_dtypes(exclude=[np.number]).columns

# Apply Median to Numeric Columns
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)  
    imputation_dict[col] = median_val

# Apply Mode to String/Categorical Columns
for col in string_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)   
    imputation_dict[col] = mode_val

# 3. Target Encoding
# Assuming 'Response' or 'Claim_Status' is your target. Update if needed.
target_col = 'Response' if 'Response' in df.columns else 'Claim_Status'

le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

# 4. One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)
trained_columns = X.columns.tolist() # Save this specific column layout

# 5. Train/Test Split & SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("--- PHASE 2: Model Training ---")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_resampled, y_train_resampled)

# Train XGBoost (Simplified Tuning for brevity)
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    random_state=42, 
    eval_metric='logloss',
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_weight
)
xgb_model.fit(X_train, y_train)

print("--- PHASE 3: Exporting Assets ---")

# Save Models
joblib.dump(rf_model, 'rf_model.joblib')
xgb_model.save_model('xgb_model.json')

# Save Preprocessing Artifacts
joblib.dump(le, 'label_encoder.joblib')
joblib.dump(trained_columns, 'trained_columns.joblib')
joblib.dump(imputation_dict, 'imputation_dict.joblib')

print("Success! All models, encoders, layout, and imputation rules saved.")