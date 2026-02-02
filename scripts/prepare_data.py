
import pandas as pd
import numpy as np
import json
import os
import sys
from tqdm import tqdm

from time import time

import pickle

def transform_title(title: str) -> str:

    strs = title.split(" ")

    make = strs[0].lower()
    model = "_".join(strs[1:]).lower()

    return make, model

def enrich_data(df: pd.DataFrame, save_data:bool =True, use_old: bool=False) -> pd.DataFrame:
    """
    Function takes a DataFrame and uses the JSON files in 'car_results' to add additional features. 
    
    :param pd.DataFrame df: The DataFrame to be modified and receive additional features.
    :param bool save_data:  Bool to determine if data is saved as a csv file to 'data' directory.
    :param bool use_old:    Bool to determine if the function pulls in the old data saved in the 'data' directory.


    :return pd.DataFrame: The DataFrame with the addtional features. 
    """
    print(f"\n======================================================================================================")
    print(f"def enrich_data")
    print(f"    - Enriching DataFrame to include more features...")

    file_name = "data_enriched.csv"
    if use_old and file_name in os.listdir("data"):
        print(f"    - Found old CSV file to read from. Returning now.")
        df = pd.read_csv(os.path.join("data", file_name))
        print(f"======================================================================================================\n\n")
        return df
    
    # Additional features in the JSON files to be added to the DataFrame
    engine_features = [
        "Engine capacity (litre)"
        , "Engine detail"
        , "Cylinder layout and quantity"
        , "Fuel type"
        , "Fuel capacity"
        , "Fuel consumption (average)**"
        , "Power maximum (detail)"
        , "Acceleration 0-100 km/h"
        , "Maximum/top speed"
        , "CO2 emissions (average)"
        , "Torque maximum"
    ]

    handling_features = [
        "Driven wheels"
        , "Power steering"
        , "Stability control"
        , "Traction control"
        , "Front tyres"
        , "Rear tyres"
    ]

    comfort_features = [
        "Air conditioning"
        , "Electric windows"
        , "Seats (quantity)"
        , "Heated seats - front"
        , "No of doors"
    ]

    general_features = [
        "transmission"
    ]

    features = [engine_features, handling_features, comfort_features, general_features]

    # Populate the dataframe with additional data

    # Depending on which angle we are approaching this from, we might only want to loop through each file in the car_results directory
    # or each entry in the DataFrame. Whichever is less, should be less computationally expensive.
    
    print(f"Enriching data...")

    # New Dataframe to hold data that has already been processed.
    df_new = df.copy()

    start = time()

    with open("data/all_car_data.json") as f:
        all_car_data = json.load(f)
    

    if len(df) > len(all_car_data):
        
        print(f"Adding full_name column for ease of lookup")
        df["full_name"] = df["title"] + " " + df["variant"]

        df["full_name"] = df["full_name"].str.lower().str.replace(" ", "_").str.replace("/", "#")




        for variant, variant_data in tqdm(all_car_data.items()):


            specifications = variant_data.get("specifications", {})
            general_info = specifications.get("General", {})
            engine_info = specifications.get("Engine", {})
            handling_info = specifications.get("Handling", {})
            comfort_info = specifications.get("Comfort", {})

            make = general_info.get("Make", "")
            model = general_info.get("Model", "")
            variant = general_info.get("Variant", "")

            car_name = f"{make} {model} {variant}".lower().replace(" ", "_").replace("/", "#")

            subset_indexes = list(df[df["full_name"] == car_name].index)

            for feature_set in features:
                for feature in feature_set:
                    val = general_info.get(feature) or engine_info.get(feature) or handling_info.get(feature) or comfort_info.get(feature)

                    df_new.loc[subset_indexes, feature] = val

            # Drop the entries from the old DataFrame so that we can search through a smaller number of rows each time                
            df = df.drop(index=subset_indexes)


        print(f"Found cars: {df_new.shape[0]}")
        print(f"Not found: {df.shape[0]}")

        df = df_new
    else:

    

        not_found = 0
        for i in tqdm(range(df.shape[0])):
            row = df.loc[i]

            title = row["title"].lower().replace("land rover", "land_rover").replace(" ", "_").replace("/", "#")
            variant = row["variant"].lower().replace(" ", "_").replace("/", "_")

            # try:
            #     file_name = os.path.join("car_results", f"{title}_{variant}.json")
            #     with open(file_name) as f:
            #         car_data = json.load(f)
            # except FileNotFoundError:
            #     car_data = {}
            #     not_found += 1
            #     print(f"Could not find car: '{file_name}' ")
            car_data = all_car_data.get(f"{title}_{variant}", {})
    
            specifications = car_data.get("specifications", {})

            general_info = specifications.get("General", {})
            engine_info = specifications.get("Engine", {})
            handling_info = specifications.get("Handling", {})
            comfort_info = specifications.get("Comfort", {})

            for feature_set in features:
                for feature in feature_set:
                    df.loc[i, feature] = general_info.get(feature) or engine_info.get(feature) or handling_info.get(feature) or comfort_info.get(feature)

        print(f"Failed to find {not_found} cars")
    end = time()
    print(f"Data enriched. Total Time: {round(end - start, 2)}s")

    if save_data: 
        print(f"Saving data to {os.path.join("data", file_name)}")
        df.to_csv(os.path.join("data", file_name))

    print(f"    - Returning dataframe")
    print(f"======================================================================================================\n\n")

    return df

def prepare_data(df: pd.DataFrame, save_data=False, use_old=False, price_threshold=(0,np.inf), cat_cols = None, num_cols = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Function takes in a pandas DataFrame, and performs some data preparation so we can standardize 
    the data being used to train the model and the data being used for our predictions.

    :param pd.DataFrame df: The input dataframe to be cleaned and enriched.
    :param bool save_data: Whether or not to save the processed data as a CSV file.
    :param bool use_old: Bool to choose if previously prepared data can be read in instead of processing the full dataset.
    :param tuple price_threshold: Consists of a lower and upper limit, used to filter out the cars used for training. 

    """
    print(f"\n======================================================================================================")

    assert isinstance(df, pd.DataFrame)

    print("def prepare_data")

    # Measure the time the entire prepare_data function takes to run
    func_start = time()

    # -------------------------------------------------------
    # STEP 0. Pull in old data
    #   - If param is specified, we can look for previously processed data and return that instead
    #   - Look for 'data/prepared_data.csv' and 'data/prepared_y.csv'
    # -------------------------------------------------------
    if use_old:
        print(f"    - Looking for previously saved data...")
        if "prepared_data.bin" in os.listdir("data"):
            print(f"    - Found data. Opening file and loading data...")

            with open("data/prepared_data.bin", "rb") as f:
                (df, y) = pickle.load(f)
            
            print(f"    - Data loaded.")
            return (df, y)
        else:
            print(f"    - use_old was set to True, but we could not find the required bin files. Continue with preparing the data...")


    # -------------------------------------------------------
    # STEP 1. Filter rows based on price
    # -------------------------------------------------------
    print(f"    - Filtering data based on price threshold: {price_threshold}")

    lower_threshold = price_threshold[0]
    upper_threshold = price_threshold[1]
    try:
        print(f"    - Initial Size: {df.shape[0]}")
        df["price"] = convert_column_to_float(df, "price", parse_price)

        df = df[(df["price"] <= upper_threshold) & (df["price"] >= lower_threshold)]
        df = df.reset_index(drop=True)
        print(f"    - New Size: {df.shape[0]}")
    
    except:
        pass
    
    # -------------------------------------------------------
    # STEP 2. PULL IN ADDITIONAL FEATURES FROM THE JSON FILES
    #   - Either loop through files in `car_results` directory or through the items in the DataFrame (whichever is smaller)
    # -------------------------------------------------------
    df = enrich_data(df, save_data=save_data, use_old=use_old)

    # -------------------------------------------------------
    # STEP 3. Standardize column names, data 
    #   - Convert to lower case
    #   - Replace spaces
    #   - Select the columns we want
    # -------------------------------------------------------

    print(f"    - Standardizing data...")
    start = time()
    # Adjust the columns
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    df["title"] = df["title"].str.lower()

    print(f"    - Extracting make and model from Title...")
    # Extract Make and Model from Title column
    df["title"] = df["title"].str.replace("land rover", "land-rover")

    for i, title in tqdm(enumerate(df["title"])):
        make = title.split(" ")[0]
        model = " ".join(title.split(" ")[1:])
        df.loc[i, "make"] = make
        df.loc[i, "model"] = model


    print(f"    - Renaming columns...")
    # Rename inconvenient column names
    rename_columns = {
        "fuel_consumption_(average)**"  : "avg_fuel_consumption"
        , "engine_capacity_(litre)"     : "engine_capacity"
        , "power_maximum_(detail)"      : "max_power"
        , "acceleration_0-100_km/h"     : "acceleration"
        , "maximum/top_speed"           : "max_speed"
        , "seats_(quantity)"            : "num_seats"
        , "heated_seats_-_front"        : "heated_seats_front"
        , "co2_emissions_(average)"     : "co2_emissions"
        , "torque_maximum"              : "max_torque"
    }

    for orig_col, new_col in rename_columns.items():
        df.rename(columns={orig_col: new_col}, inplace=True)

    print(f"All Column Names: {df.columns}")

    print(f"    - Converting columns to floats...")
    # Convert fields that contain float values to be floats
    float_columns = {
        "fuel_capacity"             : parse_fuel_capacity
        , "acceleration"            : parse_acceleration
        , "max_power"               : parse_max_power
        , "avg_fuel_consumption"    : parse_avg_fuel_consumption
        , "mileage"                 : parse_mileage
        , "price"                   : parse_price
        , "engine_capacity"         : parse_engine_capacity
        , "max_speed"               : parse_max_speed
    }

    for f_col, func in float_columns.items():
        try:
            print(f"Tring to convert col: '{f_col}' to float")
            df[f_col] = convert_column_to_float(df, f_col, func)
        except:
            pass

    # Set the Province if it is not             
    
    if "province" not in list(df.columns):
    
        cities_provinces_map = pd.read_csv("data/sa_cities_provinces.csv")

        locations = df["location"]
        provinces = []
        for l in tqdm(locations):
            if isinstance(l, str):
                try:
                    city = l.split(",")[0].lower().strip()
                    province = cities_provinces_map[cities_provinces_map["area"] == city]["province"].values[0]
                    provinces.append(province.lower())
                except:
                    provinces.append("na")
            else:
                provinces.append("na")

        df["province"] = provinces

    end = time()
    print(f"Data standardized. Total Time: {round(end - start, 2)}s")

    # -------------------------------------------------------
    # STEP 4. Make sure the DataFrame contains only the features we want
    #   - Extract desired numerical and categorical features
    #   - Convert str columns to categorical. This is required for DMatrix manipulation
    #   - Extract the feature column. This may not be present
    # -------------------------------------------------------
    if cat_cols is None:
        cat_cols = [   
            "transmission"
            , "make"
            , "model"
            , "fuel_type"
            , "province"
            , "no_of_doors"
        ]
    if num_cols is None:
        num_cols = [
            "year"
            , "fuel_capacity"
            , "avg_fuel_consumption"
            , "mileage"
            , "max_power"
            , "acceleration"
            , "max_speed"
        ]

    target = "price"

    # Target may not be present if we are processing data for predictions
    try:
        y = df[target].values
    except:
        y = None

    df = df[cat_cols + num_cols]

    for col in cat_cols:
        print(col)
        df.loc[:, col] = df[col].astype(str).str.lower()
        df.loc[:, col] = df[col].str.replace("none", "na")

    # -------------------------------------------------------
    # STEP 6. Missing values
    #   - For numerical columns: fill with the mean
    #   - For categorical columns: Fill with the value "NA"
    # -------------------------------------------------------
    
    print(f"Filling missing values...")
    start = time()
    for col in list(df.select_dtypes(include=["category", "object"]).columns):
        df[col] = df[col].fillna("na")
    
    for col in list(df.select_dtypes(include=["float64", "int"]).columns):
        df[col] = df[col].fillna(df[col].mean())
    end = time()
    print(f"Missing values filled. Total time: {round(end-start, 3)}s \n")

    for col in list(df.select_dtypes(include=["object"]).columns):
        df[col] = df[col].astype("category")
    

    # -------------------------------------------------------
    # STEP 6. [OPTIONAL] SAVING THE DATA
    #   - If save_data parameter is specified, we need to save the data we have processed
    # -------------------------------------------------------

    if save_data:
        print(f"Saving data...")
        start = time()

        with open("data/prepared_data.bin", "wb") as f:
            pickle.dump((df, y), f)

        end = time()
        print(f"Saving complete. Total time: {round(end - start, 2)}s\n")

    func_end = time()
    print(f"Data Preparation Complete! Total Time = {round(func_end-func_start, 3)}s\n")
    return (df, y)




def fill_mean_values(df_small: pd.DataFrame, df_big: pd.DataFrame) -> pd.DataFrame:
    """
        Function to fill null values in small DataFrame with the mean values in the big DataFrame
    """

    print(f"Filling null values in prediction...")
    print("Prediction Now:")
    print(df_small)

    null_columns = set(df_small.columns[df_small.isna().any()])
    num_columns = set(df_small.select_dtypes(include=["float64", "int"]))

    null_num_cols = null_columns.intersection(num_columns)

    for col in null_columns:
        if col in null_num_cols: 
            df_small.loc[:, col] = df_small.loc[:, col].fillna(df_big[col].mean())
        else:
            df_small.loc[:, col] = "na"
 
    return df_small

def convert_column_to_float(df: pd.DataFrame, column_name: str, parse_function, null_value = np.nan) -> list:
    """
        Function to parse the columns of a dataframe to float values. 

        Parameters
        -------------
        :param pd.DataFrame df: The DataFrame containing the column to be parsed.
        :param str column_name: The name of the column to be parsed, as it appears in the DataFrame.
        :param function parse_function: A function to be executed on each value in the column, used to extract a float value from a string. I.e if the float is the first 3 characters of a string: return val[0:3]
        :param null_value: The default value to be used in case we are not able to extract a float value.

        
        Returns
        ------------
        :return list: A list containing the float values representing in the selected column.
    """


    column = df[column_name]
    new_values = []

    # Mainly used for debugging, seeing where the values failed
    failed_values = {}

    for val in column:
        try:
            if not isinstance(val, float):
                new_val = (float)(parse_function(val))
            else:
                new_val = val
            new_values.append(new_val)
        except Exception as e:
            error_key = str(e)
            failed_values[error_key] = failed_values.get(error_key, 0) + 1
            new_values.append(null_value)

    return new_values


# Following functions are for converting invidual columns to string values

def parse_tyres(val:str) -> list:

    # Example: 215/40 R18

    try:
        tyre, radius = val.split(" ")

        radius = (int)(radius[1:])
        width, aspect_ratio = tyre.split("/")

        width = (int)(width)
        aspect_ratio = (int)(aspect_ratio)

    except:
        print(f"Failed to parse tyre value: {val}")
        return [0, 0, "na"]

    return [width, aspect_ratio, radius]


def parse_acceleration(val: str) -> str:
    return val.split(" ")[0].replace(",", ".")

def parse_engine_capacity(val: str) -> str:
    return val.replace("L", "").replace(",", ".")

def parse_max_speed(val: str) -> str:
    return val.split(" ")[0].strip()

def parse_fuel_capacity(val: str) -> str:
    """
        Function to extract float section of fuel capacity fields.  
        Different styles:
            - '50'
            - '87 + 63 (total 150)'
            - '70 litre + 34.5 kWh'
            - '75 (opt 85)'
            - '81.2 kWh'

        Parameters
        ------------
        :param str val: The value to be parsed as a float
    """

    try:
        # Check if entire string can easily be converted to an int or float
        (int)(val)
        (float)(val)
        return val
    except Exception:
        pass

    if len(val.split("litre")) > 1:
        return val.split("litre")[0].strip()

    if len(val.split(" ")) == 2 and val.split(" ")[1] == "kwh":
        return None

    if "opt" in val:
        return val.split(" ")[0]
    
    if "total" in val:
        return val.split("total")[-1].strip(" ()")

    return None


def parse_max_power(val: str) -> str:

    try:
        return float(val.split(" ")[0])
    except:
        return None


def parse_avg_fuel_consumption(val: str) -> str:
    """
        Function to parse the data in the average_fuel_consumption column. 
    """

    if isinstance(val, float):
        return val

    try:
        if "/100km" in val:
            return val.split(" ")[0].replace(",", ".")
    except Exception as e:
        pass

    val_split = val.split(" ")
    if len(val_split) > 1 and "-" in val_split[0]:

        [num1, num2] = val_split[0].split("-")

        print(f"num1: {num1}, num2: {num2}")

        return (float(num1) + float(num2)) / 2

    try:
        [num1, num2] = val.split("-")

        return (float(num1) + float(num2)) / 2
    except:
        return None


def parse_price(val):
    if isinstance(val, float) or isinstance(val, int):
        return val
    return val.replace(" ", "")


def parse_mileage(val):
    return float(val)


def extract_province(location: str) -> str:
    city = location.split(",")[1].strip()

    with open("data/sa_cities_provinces.csv") as f:
        city_to_province_map = json.load(f)
    
    province = city_to_province_map.get(city, "na")

    return province


if __name__ == "__main__":
    df = pd.read_csv("data/raw.csv")
    
    from time import time

    start = time()

    df, y = prepare_data(df, use_old=False, save_data=True, price_threshold=(0, 1500000))
    end = time()

    print(df.head(10))
    print(f"Processing Time: {end - start}s")

