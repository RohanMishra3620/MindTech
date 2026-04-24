import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

df = pd.read_csv(r"C:\Users\C9IN\Desktop\Mental_Health\02_Dataset\04_Final_dataset.csv")

features = [
    'Age', 'Education_Level', 'Number_of_Children',
    'Physical_Activity_Level', 'Employment_Status', 'Income',
    'Alcohol_Consumption', 'Dietary_Habits', 'Sleep_Patterns',
    'History_of_Mental_Illness', 'History_of_Substance_Abuse',
    'Family_History_of_Depression', 'Chronic_Medical_Conditions',
    'Marital_Status_Divorced', 'Marital_Status_Married',
    'Marital_Status_Single', 'Marital_Status_Widowed',
    'Smoking_Status_Current', 'Smoking_Status_Former',
    'Smoking_Status_Non-smoker'
]

target = 'Mental_Health_Label'

X = df[features]
y = df[target]

X = X.fillna(0)

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

weights = compute_sample_weight('balanced', y_train)

model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.5,
    max_depth=1,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=2,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train, sample_weight=weights)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

base_dir = os.getcwd()
model_dir = os.path.join(base_dir, "06_models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "mental_health_model.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(model_dir, "columns.pkl"))

print("Model + Columns Saved")