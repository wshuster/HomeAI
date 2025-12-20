# HomeAI: Intelligent House Price Estimator

HomeAI is a Flask-based machine learning web application that predicts house prices
using real home sales data from a single ZIP code.

## Project Overview
The model is trained on recent home sales from ZIP code 98103.
Users enter basic home characteristics and receive:
- A predicted home price
- An estimated price range based on model error (MAE)

## Tech Stack
- Python 3
- Pandas / NumPy
- Scikit-learn (RandomForestRegressor)
- Flask
- Joblib
- HTML / CSS (Bootstrap)

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Train the model:
   python train_model.py

3. Start the web app:
   python app.py

4. Open a browser and go to:
   http://127.0.0.1:5000
