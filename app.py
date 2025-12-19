#load libraries
from flask import Flask, render_template, request
import joblib
import pandas as pd

#init app
app = Flask(__name__)

#load model and error
model = joblib.load("model/homeai_model.joblib")
mae = joblib.load("model/mae.joblib")

#home route
@app.route("/")
def index():
    return render_template("index.html")

#predict route
@app.route("/predict", methods=["POST"])
def predict():
    #read zip
    zip_code = request.form["zip_code"]

    #read inputs
    sqft_living = float(request.form["sqft_living"])
    bedrooms = float(request.form["bedrooms"])
    bathrooms = float(request.form["bathrooms"])
    sqft_lot = float(request.form["sqft_lot"])
    yr_built = float(request.form["yr_built"])

    #build input
    X = pd.DataFrame([{
        "sqft_living": sqft_living,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_lot": sqft_lot,
        "yr_built": yr_built
    }])

    #predict
    pred = float(model.predict(X)[0])

    #range
    low = pred - float(mae)
    high = pred + float(mae)

    #format
    pred_s = f"${pred:,.0f}"
    low_s = f"${low:,.0f}"
    high_s = f"${high:,.0f}"

    return render_template(
        "result.html",
        zip_code=zip_code,
        pred=pred_s,
        low=low_s,
        high=high_s
    )


#run server
if __name__ == "__main__":
    app.run(debug=True)
