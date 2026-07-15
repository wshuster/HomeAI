# HomeAI: Intelligent House Price Estimator

HomeAI is a Flask-based machine learning web application that predicts house prices using real home sales data from a single ZIP code.

## Project Overview

The application uses a Random Forest machine learning model trained on recent home sales from ZIP code 98103.

Users enter home characteristics including:
- Square footage
- Number of bedrooms
- Number of bathrooms
- Lot size
- Year built

The application returns:
- Predicted home price
- Estimated price range based on the model's Mean Absolute Error (MAE)

## Features

- Machine learning home price prediction
- Interactive web interface built with Flask
- User-friendly input form
- Real-time prediction results
- Trained model saved and loaded with Joblib

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- HTML
- CSS
- Bootstrap

## Machine Learning Workflow

1. Load and preprocess housing data.
2. Select relevant features for training.
3. Train a Random Forest Regressor model.
4. Save the trained model using Joblib.
5. Load the model in the Flask application.
6. Generate home price predictions from user input.

## Screenshots

### Home Page
*(Add a screenshot of your application's home page here.)*

### Prediction Result
*(Add a screenshot showing the prediction results here.)*

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/wshuster/HomeAI.git
cd HomeAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the machine learning model

```bash
python train_model.py
```

### 4. Start the Flask application

```bash
python app.py
```

### 5. Open your browser

Visit:

```
http://127.0.0.1:5000
```

## Future Improvements

- Support additional ZIP codes
- Improve prediction accuracy through feature engineering
- Add interactive data visualizations
- Deploy the application to a cloud hosting platform
- Enhance the user interface and user experience

## Author

**William Shuster**

Computer Science Graduate  
Python • Flask • SQL • Linux • Machine Learning
