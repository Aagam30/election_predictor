import os
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = pickle.load(open("election_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    print("Error loading model or scaler:", e)
    model = None
    scaler = None

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = ""
    if request.method == "POST":
        try:
            age = float(request.form["age"])
            income = float(request.form["income"])
            education = float(request.form["education"])
            gender = 1 if request.form["gender"] == "Male" else 0
            region = int(request.form["region"])

            features = np.array([[age, income, education, gender, region]])
            scaled = scaler.transform(features)
            result = model.predict(scaled)

            prediction = "Yes" if result[0] == 1 else "No"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT as env variable
    app.run(host="0.0.0.0", port=port, debug=False)
