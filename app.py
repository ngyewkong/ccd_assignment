from flask import Flask, render_template, request
import joblib
from numpy import ndarray

app = Flask(__name__)
lr_model = joblib.load("CCD_lr_assignment")
dt_model = joblib.load("CCD_dt_assignment")
mlp_model = joblib.load("CCD_mlp_assignment")
rf_model = joblib.load("CCD_rf_assignment")
gb_model = joblib.load("CCD_xgb_assignment")

# @ is a function decorator
# must run the app.route first before running any function below


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = request.form.get('age')
        age = float(age)
        print(age)
        income = request.form.get('income')
        income = float(income)
        print(income)
        loan = request.form.get('loan')
        loan = float(loan)
        print(loan)

        input = [[income, age, loan]]
        print(input)
        lr_pred = lr_model.predict(input)
        print("lr is " + str(lr_pred[0]))

        dt_pred = dt_model.predict(input)
        print("dt is " + str(dt_pred[0]))

        mlp_pred = mlp_model.predict(input)
        print("mlp is " + str(mlp_pred[0]))

        rf_pred = rf_model.predict(input)
        print("rf is " + str(rf_pred[0]))

        gb_pred = gb_model.predict(input)
        print("gb is " + str(gb_pred[0]))

        if lr_pred[0] == 0:
            result1 = "No Default"
        elif lr_pred[0] == 1:
            result1 = "Default"

        if dt_pred[0] == 0:
            result2 = "No Default"
        elif dt_pred[0] == 1:
            result2 = "Default"

        if mlp_pred[0] == 0:
            result3 = "No Default"
        elif mlp_pred[0] == 1:
            result3 = "Default"

        if rf_pred[0] == 0:
            result4 = "No Default"
        elif rf_pred[0] == 1:
            result4 = "Default"

        if gb_pred[0] == 0:
            result5 = "No Default"
        elif gb_pred[0] == 1:
            result5 = "Default"

        return (render_template("index.html", result1='Logistic Regression model predicts: ' + result1, result2='Decision Tree model predicts: ' + result2, result3='Neural Network model predicts: ' + result3, result4='Random Forest model predicts: ' + result4, result5='Gradient Boosted Decision Tree model predicts: ' + result5))
    else:
        return (render_template("index.html", result1='No input submitted.', result2='No input submitted.', result3='No input submitted.', result4='No input submitted.', result5='No input submitted.'))
