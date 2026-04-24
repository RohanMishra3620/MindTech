import pandas as pd

df = pd.read_csv(r"C:\Users\C9IN\Desktop\Mental_Health\02_Dataset\03_Feature_Eng.csv")

def calculate_mental_health(row):
    score = 0.0

    if row.get("History_of_Mental_Illness", 0) == 1:
        score += 3.0
    if row.get("History_of_Substance_Abuse", 0) == 1:
        score += 3.0
    if row.get("Family_History_of_Depression", 0) == 1:
        score += 2.0

    if "Work_Disability" in df.columns and row.get("Work_Disability", 0) == 1:
        score += 3.5
    elif "Employment_Status" in df.columns and row.get("Employment_Status") == "Unemployed":
        score += 2.5

    if pd.notnull(row.get("Income")):
        if row["Income"] < 20000:
            score += 1.5
        elif row["Income"] < 50000:
            score += 0.5

    if row.get("Smoking_Status_Current", 0) == 1:
        score += 2.0
    if row.get("Sleep_Patterns", 1) == 0:
        score += 2.0
    if row.get("Physical_Activity_Level", 1) == 0:
        score += 1.0
    if row.get("Alcohol_Consumption", 0) == 2:
        score += 2.0
    if row.get("Dietary_Habits", 1) == 0:
        score += 1.0

    if row.get("Physical_Activity_Level", 1) == 2:
        score -= 1.0
    if row.get("Sleep_Patterns", 1) == 2:
        score -= 1.0

    if row.get("Marital_Status_Divorced", 0) == 1 or row.get("Marital_Status_Widowed", 0) == 1:
        score += 1.0

    if pd.notnull(row.get("Age")):
        if row["Age"] < 25 or row["Age"] > 60:
            score += 1.0

    if row.get("Chronic_Medical_Conditions", 0) == 1:
        score += 1.0

    if score <= 4:
        return 0
    elif score <= 8:
        return 1
    else:
        return 2
df["Mental_Health_Label"] = df.apply(calculate_mental_health, axis=1)

df.to_csv(r"C:\Users\C9IN\Desktop\Mental_Health\02_Dataset\04_final_dataset.csv", index=False)

print("Processing Complete ✅")

print("\n📊 Training Data Distribution:")
print(df["Mental_Health_Label"].value_counts())