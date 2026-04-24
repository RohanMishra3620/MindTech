import os
import joblib
import pandas as pd

base_dir = os.path.dirname(__file__)

model = joblib.load(os.path.join(base_dir, "..", "06_models", "mental_health_model.pkl"))
columns = joblib.load(os.path.join(base_dir, "..", "06_models", "columns.pkl"))

def get_user_input():
    return {
        "Age": int(input("Age: ")),
        "Education_Level": input("Education (High School / Bachelor's Degree / Master's Degree / PhD): "),
        "Number_of_Children": int(input("Children: ")),
        "Physical_Activity_Level": input("Activity (Sedentary/Moderate/Active): "),
        "Employment_Status": input("Employment (Employed/Unemployed): "),
        "Income": float(input("Income: ")),
        "Alcohol_Consumption": input("Alcohol (Low/Moderate/High): "),
        "Dietary_Habits": input("Diet (Unhealthy/Moderate/Healthy): "),
        "Sleep_Patterns": input("Sleep (Poor/Fair/Good): "),
        "History_of_Mental_Illness": input("Mental Illness (Yes/No): "),
        "History_of_Substance_Abuse": input("Substance Abuse (Yes/No): "),
        "Family_History_of_Depression": input("Family Depression (Yes/No): "),
        "Chronic_Medical_Conditions": input("Chronic Disease (Yes/No): "),
        "Marital_Status_Divorced": 0,
        "Marital_Status_Married": 0,
        "Marital_Status_Single": 0,
        "Marital_Status_Widowed": 0,
        "Smoking_Status_Current": 0,
        "Smoking_Status_Former": 0,
        "Smoking_Status_Non-smoker": 0
    }

def encode_input(user):
    df = pd.DataFrame([user])

    yes_no_cols = [
        "History_of_Mental_Illness",
        "History_of_Substance_Abuse",
        "Family_History_of_Depression",
        "Chronic_Medical_Conditions"
    ]

    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    return df

def predict(df):
    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0]

    labels = ["Low Risk", "Moderate Risk", "High Risk"]

    print("\nPrediction:", labels[pred])
    print("Confidence:", [round(p * 100, 2) for p in probs])

user = get_user_input()
encoded = encode_input(user)
predict(encoded)