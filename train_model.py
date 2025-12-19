#load libraries
from xml.parsers.expat import model
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#main entry
def main():
    #load cleaned dataset
    df = pd.read_csv("data/sales.csv")

    #split features and target
    X = df.drop(columns=["price"])
    y = df["price"]

    #train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #train model
    model = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)
    model.fit(X_train, y_train)


    #predict
    preds = model.predict(X_test)

    #error
    mae = mean_absolute_error(y_test, preds)

    #save model and error
    joblib.dump(model, "model/homeai_model.joblib")
    joblib.dump(mae, "model/mae.joblib")

    #confirm
    print("model trained for zip 98103")
    print("mae:", mae)

#run main
if __name__ == "__main__":
    main()
