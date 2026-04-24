import os
import joblib
import pandas as pd
import random
from collections import Counter

base_dir = os.path.dirname(__file__)

model = joblib.load(os.path.join(base_dir, "..", "06_models", "mental_health_model.pkl"))
columns = joblib.load(os.path.join(base_dir, "..", "06_models", "columns.pkl"))

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

def generate_random_user():
    return {
        "Age": random.randint(18, 60),

        "Education_Level": random.choices(
            ["High School", "Bachelor's Degree", "Master's Degree"],
            weights=[0.3, 0.5, 0.2]
        )[0],

        "Physical_Activity_Level": random.choices(
            ["Sedentary", "Moderate", "Active"],
            weights=[0.2, 0.6, 0.2]
        )[0],

        "Employment_Status": random.choices(
            ["Employed", "Unemployed"],
            weights=[0.8, 0.2]
        )[0],

        "Alcohol_Consumption": random.choices(
            ["Low", "Moderate", "High"],
            weights=[0.6, 0.3, 0.1]
        )[0],

        "Dietary_Habits": random.choices(
            ["Healthy", "Moderate", "Unhealthy"],
            weights=[0.4, 0.4, 0.2]
        )[0],

        "Sleep_Patterns": random.choices(
            ["Good", "Fair", "Poor"],
            weights=[0.4, 0.4, 0.2]
        )[0],

        "History_of_Mental_Illness": random.choices(
            ["Yes", "No"], weights=[0.2, 0.8]
        )[0],

        "History_of_Substance_Abuse": random.choices(
            ["Yes", "No"], weights=[0.1, 0.9]
        )[0],

        "Family_History_of_Depression": random.choices(
            ["Yes", "No"], weights=[0.3, 0.7]
        )[0],

        "Chronic_Medical_Conditions": random.choices(
            ["Yes", "No"], weights=[0.2, 0.8]
        )[0],

        "Number_of_Children": random.randint(0, 3),
        "Income": random.randint(10000, 80000)
    }

def auto_test(n=100):
    print("\nRunning Fully Automated Test...\n")

    results = []

    for i in range(n):
        user = generate_random_user()
        encoded = encode_input(user)

        pred = model.predict(encoded)[0]
        probs = model.predict_proba(encoded)[0]

        labels = ["Low", "Moderate", "High"]

        results.append(pred)

        print(f"Test {i+1}: Age {user['Age']} -> {labels[pred]} | Confidence: {[round(p*100,2) for p in probs]}")

    count = Counter(results)

    print("\nFinal Distribution:")
    print("Low:", count[0])
    print("Moderate:", count[1])
    print("High:", count[2])

if __name__ == "__main__":
    print("\nMental Health Prediction System (AUTO MODE)\n")
    auto_test(100)