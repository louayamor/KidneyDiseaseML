from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        try:
            data = {
                "age": float(request.form["age"]),
                "blood_pressure": float(request.form["blood_pressure"]),
                "specific_gravity": float(request.form["specific_gravity"]),
                "albumin": float(request.form["albumin"]),
                "sugar": float(request.form["sugar"]),
                "red_blood_cells": request.form["red_blood_cells"],
                "pus_cell": request.form["pus_cell"],
                "pus_cell_clumps": request.form["pus_cell_clumps"],
                "bacteria": request.form["bacteria"],
                "blood_glucose_random": float(request.form["blood_glucose_random"]),
                "blood_urea": float(request.form["blood_urea"]),
                "serum_creatinine": float(request.form["serum_creatinine"]),
                "sodium": float(request.form["sodium"]),
                "potassium": float(request.form["potassium"]),
                "haemoglobin": float(request.form["haemoglobin"]),
                "packed_cell_volume": float(request.form["packed_cell_volume"]),
                "white_blood_cell_count": float(request.form["white_blood_cell_count"]),
                "red_blood_cell_count": float(request.form["red_blood_cell_count"]),
                "hypertension": request.form["hypertension"],
                "diabetes_mellitus": request.form["diabetes_mellitus"],
                "coronary_artery_disease": request.form["coronary_artery_disease"],
                "appetite": request.form["appetite"],
                "peda_edema": request.form["peda_edema"],
                "aanemia": request.form["aanemia"],
            }

            input_df = pd.DataFrame([data])

            categorical_cols = [
                "red_blood_cells",
                "pus_cell",
                "pus_cell_clumps",
                "bacteria",
                "hypertension",
                "diabetes_mellitus",
                "coronary_artery_disease",
                "appetite",
                "peda_edema",
                "aanemia",
            ]

            label_encoders = {col: LabelEncoder().fit(input_df[col]) for col in categorical_cols}
            for col, le in label_encoders.items():
                input_df[col] = le.transform(input_df[col])

            numeric_cols = [
                "age",
                "blood_pressure",
                "specific_gravity",
                "albumin",
                "sugar",
                "blood_glucose_random",
                "blood_urea",
                "serum_creatinine",
                "sodium",
                "potassium",
                "haemoglobin",
                "packed_cell_volume",
                "white_blood_cell_count",
                "red_blood_cell_count",
            ]
            scaler = StandardScaler()
            input_df[numeric_cols] = scaler.fit_transform(input_df[numeric_cols])

            pipeline = PredictionPipeline()
            prediction = pipeline.predict(input_df)
            print("[DEBUGGER] The prediction is: ", prediction)

            label_map = {0: "This patient is safe from chronic diseases ", 1: "Warning this patient has chronic disease"}
            prediction_label = label_map[int(prediction[0])]

            return render_template("results.html", prediction=prediction_label)

        except Exception as e:
            print("The Exception message is:", e)
            return "Something went wrong, please check your inputs."

    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
