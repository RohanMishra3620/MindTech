import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv(r"C:\Users\C9IN\Desktop\Mental_Health\02_Dataset\04_Final_dataset.csv")

model = joblib.load("06_models/mental_health_model.pkl")
columns = joblib.load("06_models/columns.pkl")

X = df.drop("Mental_Health_Label", axis=1)
y = df["Mental_Health_Label"]

X = pd.get_dummies(X)
X = X.reindex(columns=columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = model.predict(X_test)

print("\n========== MODEL PERFORMANCE ==========\n")

test_acc = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, model.predict(X_train))

print(f"Train Accuracy : {round(train_acc * 100, 2)} %")
print(f"Test Accuracy  : {round(test_acc * 100, 2)} %")

labels = ["Low", "Moderate", "High"]

print("\n========== CLASSIFICATION REPORT ==========\n")
print(classification_report(y_test, y_pred, target_names=labels))

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{l}" for l in labels],
    columns=[f"Pred_{l}" for l in labels]
)

print("\n========== CONFUSION MATRIX ==========\n")
print(cm_df)

errors = X_test.copy()
errors["Actual"] = y_test.values
errors["Predicted"] = y_pred

errors = errors[errors["Actual"] != errors["Predicted"]]

print("\n========== ERROR ANALYSIS ==========\n")
print("Total Errors:", len(errors))
print("\nSample Errors:\n")
print(errors.head(10))

print("\n========== PREDICTION DISTRIBUTION ==========\n")
print(pd.Series(y_pred).value_counts())

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})