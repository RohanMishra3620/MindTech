import os
import joblib
import pandas as pd
import traceback
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# -------- Paths --------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model", "mental_health_model.pkl")
cols_path = os.path.join(base_dir, "model", "columns.pkl")


# -------- Load model function (IMPORTANT FIX) --------
def load_model():
    try:
        print("Loading model from:", model_path)
        model = joblib.load(model_path)

        print("Loading columns from:", cols_path)
        columns = joblib.load(cols_path)

        print("Model & Columns loaded successfully")
        return model, columns

    except Exception as e:
        print("MODEL LOAD ERROR ")
        traceback.print_exc()
        return None, None


# -------- Encode input --------
def encode_input(user_data, columns):
    df = pd.DataFrame([user_data])

    yes_no_cols = [
        "History_of_Mental_Illness",
        "History_of_Substance_Abuse",
        "Family_History_of_Depression",
        "Chronic_Medical_Conditions"
    ]

    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0)

    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    return df


# -------- Routes --------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():

    model, columns = load_model()

    if model is None or columns is None:
        return jsonify({
            "error": "Model failed to load. Check server logs."
        }), 500

    try:
        user = {
            "Age": int(request.form.get("Age", 0)),
            "Education_Level": request.form.get("Education", "Bachelor's Degree"),
            "Number_of_Children": int(request.form.get("Children", 0)),
            "Physical_Activity_Level": request.form.get("Activity", "Moderate"),
            "Employment_Status": "Employed",
            "Income": float(request.form.get("Income", 0)),
            "Alcohol_Consumption": "Low",
            "Dietary_Habits": "Moderate",
            "Sleep_Patterns": request.form.get("Sleep", "Good"),
            "History_of_Mental_Illness": request.form.get("MentalIllness", "No"),
            "History_of_Substance_Abuse": request.form.get("Substance", "No"),
            "Family_History_of_Depression": request.form.get("FamilyDepression", "No"),
            "Chronic_Medical_Conditions": request.form.get("Chronic", "No")
        }

        # Categorical encoding
        marital = request.form.get("Marital", "Single")
        user[f"Marital_Status_{marital}"] = 1

        smoking = request.form.get("Smoking", "Non-smoker")
        user[f"Smoking_Status_{smoking}"] = 1

        # Encode
        df_encoded = encode_input(user, columns)

        # Predict
        prediction_idx = model.predict(df_encoded)[0]
        probabilities = model.predict_proba(df_encoded)[0]

        labels = ["Low Risk", "Moderate Risk", "High Risk"]

        return jsonify({
            "result": labels[prediction_idx],
            "probs": {
                "Low": round(float(probabilities[0]) * 100, 1),
                "Moderate": round(float(probabilities[1]) * 100, 1),
                "High": round(float(probabilities[2]) * 100, 1)
            }
        })

    except Exception as e:
        print("Prediction Error:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


# -------- Run --------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)