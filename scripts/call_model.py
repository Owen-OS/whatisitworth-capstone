
import pandas as pd
from xgboost import DMatrix

import numpy as np

import logging
try:
    from scripts import prepare_data, create_model
except:
    import prepare_data, create_model

#with open("models/xgb_model.bin", 'rb') as f:
#    model = pickle.load(f)


def get_prediction(input_data: dict, model="xgboost") -> float:
    """
        Call the model and get a prediction for a car.

        :param dict input_data: The input data for the car. Should contain Title, variant, mileage, location, and year.

    """
    print("call_model.get_prediction...")

    print("Fetching model...")
    model = create_model.fetch_model("model_v2.bin", use_old=True)

    print("Model loaded.\n")

    print(f"Preparing DF...")
    df_base = pd.read_csv("data/autotrader.csv")
    df, _ = prepare_data.prepare_data(df_base, save_data=True, use_old=True)
    print(f"DF prepared.\n")

    print(f"Preparing Input DF...")
    df_input = pd.DataFrame([input_data])
    print(f"1. {df_input}")


    df_input, _ = prepare_data.prepare_data(df_input, use_old=False, save_data=False)
    df_input = prepare_data.fill_mean_values(df_input, df)

    print(f"Trying to convert df_input to a DMatrix...")
    dm_input = DMatrix(df_input, enable_categorical=True)
    print(f"Input converted to DMatrix.\n")


    print(dm_input)

    print(f"Getting a prediction from the model...")
    pred = model.predict(dm_input)[0]
    print(f"Got a prediction: {pred}\n")

    return round(pred, 2)

if __name__ == "__main__":
    input_data = {
          "title"       : "ford fiesta"
        , "variant"     : "ST"
        , "mileage"     : 20000
        , "year"        : 2020
        , "province"    : "Western Cape" 
    }

    print(get_prediction(input_data))


