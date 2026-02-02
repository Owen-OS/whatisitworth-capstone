# Script to train an XGBoost model and save it using pickle

import xgboost as xgb           #type:ignore
from xgboost import DMatrix     #type:ignore
import pickle
import pandas as pd             #type:ignore

import os
try:
    from scripts import prepare_data
except:
    import prepare_data


def fetch_model(model_name: str = "model.bin", use_old=False, save_model=False):
    """
    def fetch_model returns a machine learning model. It can either train a new model, or load an existing model depending on the parameters specified.
    
    :param str model_name: The name of the model to be fetched/saved. 
    :param bool use_old: Whether or not to load an existing model, found in ./data/
    :param bool save_model: Whether or not to save the current model, overwriting the model currently saved.
    """

    # ================================================================================================================
    # STEP 1. Fetch an existing model, if relevant
    # ================================================================================================================
    
    print(f"Fetching model...")
    print("\tChecking to see if model has been created before...")

    print(os.listdir("models"))
    if use_old and model_name in os.listdir("models"):
        print(f"\tLocated {model_name} in models dir")

        with open(f"models/{model_name}", 'rb') as f:
            model = pickle.load(f)
            print(f"\t Returning model.")
            print("===========================================\n\n")
            return model
        print("Model not found. Generating model...")
    
    # ================================================================================================================
    # STEP 2. Fetch new data.
    #   - We are not fetching an existing model.
    #   - Fetch data for a new model.
    # ================================================================================================================
    df = pd.read_csv("data/raw.csv")

    print("Loading in prepared data...")
    df, y = prepare_data.prepare_data(df, save_data=True, use_old=False, price_threshold=(0, 1500000))
    print("Prepared data loaded.\n")

    print("Creating a DMatrix...")
    dm = DMatrix(df, label=y, enable_categorical=True)
    print("Dmatrix created.\n")


    # ================================================================================================================
    # STEP 3. Train a model using parameters from the hyperparameter tuning.
    #   - We are not fetching an existing model.
    #   - Fetch data for a new model.
    # ================================================================================================================  
    squared_error_model_params = {
        "objective"           : "reg:squarederror"
        , "max_depth"           : 7
        , "subsample"           : 1.0
        , "colsample_bytree"    : 0.5
        , "colsample_bylevel"   : 0.9
        , "colsample_bynode"    : 0.9
        , "min_child_weight"    : 3
        , "lambda"              : 0
        , "eta"                 : 0.05
    }

    gamma_model_parameters = {
        "objective"             : "reg:gamma",

        "max_depth"             : 3,
        "subsample"             : 1.0,
        "colsample_bytree"      : 0.3,
        "colsample_bylevel"     : 0.7,
        "colsample_bynode"      : 0.9,
        "min_child_weight"      : 6,
        "lambda"                : 1.0,
        "eta"                   : 0.1
    }

    print(f"Training model on data...")
    model = xgb.train(squared_error_model_params, dm, num_boost_round=2001)
    print(f"Model trained.\n")

    # ================================================================================================================
    # STEP 4. Save the new model, if relevant.
    #   - We are not fetching an existing model.
    #   - Fetch data for a new model.
    # ================================================================================================================
    if save_model:
        print(f"Saving model as a binary file...")
        with open(f"models/{model_name}", 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved.")

    return model
    
if __name__ == "__main__":
    fetch_model("model_v2.bin", use_old=False, save_model=True)