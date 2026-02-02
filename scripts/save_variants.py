import pandas as pd         #type:ignore
import json
import sys
from time import time


def create_make_model_json():

    print(f"===================================================================")
    print(f"Running create_make_model_json()...")

    print(f"    - Reading in the data...")
    df = pd.read_csv("data/raw.csv")
    print(f"    - Data read.\n")

    print(f"    - Isolating the relevant data...")
    df = df[["title", "variant"]]
    df["title"] = df["title"].str.replace("Land Rover", "Land-Rover")
    print(f"    - Data isolated.\n")

    print(f"    - Extracting make and model from title feature...")
    start = time()
    for i, title in enumerate(df["title"]):
            make = title.split(" ")[0]
            model = " ".join(title.split(" ")[1:])
            df.loc[i, "make"] = make
            df.loc[i, "model"] = model

    del df["title"]
    end = time()
    print(f"    - Make and model extracted. Time: {round(end - start, 2)}s\n")

    df = df.drop_duplicates()

    make_model_variants = {}

    print(f"")
    for row in df.itertuples():
        variant = row[1]
        make = row[2]
        model = row[3]    

        if make not in make_model_variants:
            make_model_variants[make] = {model: [variant]}
        elif model not in make_model_variants[make]:
            make_model_variants[make][model] = [variant]
        else:
            make_model_variants[make][model].append(variant)
        
    print(f"    - Sorting the variants...")
    start = time()
    for make, models in make_model_variants.items():
        for model, variants in models.items():
            try:
                models[model] = sorted(list(map(capitalise, variants)))
            except:
                print(variants)
    end = time()
    print(f"    - Sorting done. Time: {round(end - start, 2)}s\n")
    
    with open("data/make_model_variants.json", "w") as f:
        json.dump(make_model_variants, f, indent=4)


def capitalise(string: str) -> str:
    """
    Docstring for capitalise
    
    :param string: Description
    :type string: str
    :return: Description
    :rtype: str
    """

    try:
        string = string[0].upper() + string[1:]
    except:
        pass
    return string

if __name__ == "__main__":
     create_make_model_json()
