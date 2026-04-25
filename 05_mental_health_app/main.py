import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# -------- Load Model with Error Handling -------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model", "mental_health_model.pkl")
cols_path = os.path.join(base_dir, "model", "columns.pkl")
try:
    model = joblib.load(model_path)
    columns = joblib.load(cols_path)
except Exception as e:
    print(f" Model Load Error: {e}")
    model = None
    columns = None

def encode_input(user_data):
    df = pd.DataFrame([user_data])

    # Convert binary features
    yes_no_cols = [
        "History_of_Mental_Illness",
        "History_of_Substance_Abuse",
        "Family_History_of_Depression",
        "Chronic_Medical_Conditions"
    ]

    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0)

    # One-hot encoding and aligning with training columns
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    return df

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model files missing on server."}), 500
        
    try:
        # Extract form data
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

        # Handle Categorical Encoding (Marital & Smoking)
        marital = request.form.get("Marital", "Single")
        user[f"Marital_Status_{marital}"] = 1
        
        smoking = request.form.get("Smoking", "Non-smoker")
        user[f"Smoking_Status_{smoking}"] = 1

        # Process and Predict
        df_encoded = encode_input(user)
        prediction_idx = model.predict(df_encoded)[0]
        probabilities = model.predict_proba(df_encoded)[0]

        labels = ["Low Risk", "Moderate Risk", "High Risk"]

        return jsonify({
               "result": labels[prediction_idx],
               "probs": {
        # Convert float32 to standard float using float()
        "Low": round(float(probabilities[0]) * 100, 1),
        "Moderate": round(float(probabilities[1]) * 100, 1),
        "High": round(float(probabilities[2]) * 100, 1)
          }
     })
       

    except Exception as e:
        print(f"🔥 Prediction Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)