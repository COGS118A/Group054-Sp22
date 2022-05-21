"""Data Cleaning pipeline for Used Car Value classifier

By @gbhand for COGS118A
"""
import re
import subprocess

import numpy as np
import pandas as pd

# pylint: disable=invalid-name


def append_msrp(cleaned_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Adds MSRP data and expected depreciation to a dataframe obtained from cars.com

    Args:
        cleaned_df: dataframe containing Year, Make, and Model
        verbose: verbose logging (default false)
    Returns:
        copy of dataframe with MSRP and depreciated value appended
    """
    full_dataframe = cleaned_df.copy()
    dataframe = full_dataframe.copy()

    # select only neccessary columns and rows
    dataframe = dataframe[["Year", "Make", "Model"]]
    dataframe = dataframe.drop_duplicates()

    # `get_msrp_list` expects a n-length list of 3-dicts
    car_list = dataframe.to_dict(orient="records")
    msrp_list = get_msrp_list(car_list, verbose=verbose)

    # add column to dataframe then join with original
    dataframe["MSRP"] = msrp_list
    dataframe["MSRP"] = pd.to_numeric(dataframe["MSRP"], errors="coerce")

    dataframe["expected_value"] = get_expected_value(dataframe)

    index = full_dataframe.index
    full_dataframe = pd.merge(full_dataframe, dataframe, how="left")
    full_dataframe.index = index

    return full_dataframe


def get_expected_value(
    df: pd.DataFrame, last_year: int = 2017, method: str = "carfax"
) -> pd.DataFrame:
    """Dataframe apply wrapper for `standard_depreciation

    Args:
        df: input dataframe
        last_year: year data was collected
    Returns:
        Series of expected values
    """
    return df.apply(
        lambda x: standard_depreciation(x["MSRP"], int(last_year - x["Year"]), method),
        axis=1,
    )


def retains_value(price: float, expected_value: float) -> int:
    """Boolean threshold for vehicle retaining value

    Args:
        price: last sold price
        expected_value: value predicted via standard depreciation
    Returns:
        True if value is retained
    """
    return price >= expected_value


def which_grand(make: str, model: str) -> str:
    """Maps `Grand` model to correct name based on make

    Args:
        make: brand of the vehicle
        model: model of the vehicle
    Returns:
        corrected model of the vehicle, if applicable
    """
    GRAND_MAP = {
        "Jeep": "Grand_cherokee",
        "Dodge": "Grand_caravan",
        "Pontiac": "Grand_prix",
        "Mercury": "Grand_marquis",
        "Suzuki": "Grand_vitara",
    }
    return GRAND_MAP.get(make, model)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies data cleaning process to dataframe

    Args:
        df: dirty dataframe
    Returns:
        cleaned dataframe
    """
    df = df.copy()

    # Clean model
    df["Model"] = df["Model"].apply(parse_model)

    # there are 5 brands with a `Grand` model, none of which actually use that name
    df["Model"] = df.apply(lambda x: which_grand(x["Make"], x["Model"]), axis=1)

    return df


def append_retains_value(df: pd.DataFrame) -> pd.DataFrame:
    """Returns copy of dataframe with `retains_value` boolean column appended"""
    df = df.copy()

    df["retains_value"] = df.apply(
        lambda x: retains_value(x["Price"], x["expected_value"]), axis=1
    )

    return df


def standard_depreciation(initial_value: float, years_old: int, method: str) -> float:
    """Calculates expected depreciation solely from initial value and age
    Based on https://www.carfax.com/blog/car-depreciation

    Args:
        initial_value: price vehicle was first sold at (MSRP)
        years_old: years since purchase
    Returns:
        expected value after depreciation
    """
    if years_old == -1:
        years_old = 0

    # print(f"car is {years_old} years old")
    if years_old < 0 or years_old > 100:
        print(f"ayo chief what's this? car is {years_old} years old")
        raise RuntimeError

    # method = "ramsey"

    if method == "carfax":
        # drops 10% in value the moment it is purchased
        if years_old == 0:
            return initial_value * 0.90

        # drops 20% first year
        final_value = initial_value * 0.8

        # each addtional year drops 15%
        for _ in range(1, years_old):
            final_value -= final_value * 0.15

        return round(final_value, 2)
    elif method == "ramsey":
        if years_old == 0:
            # print(years_old)
            return initial_value * 0.90

        return initial_value * (1.4 - 0.59 * np.log(0.61 * years_old + 2.1))

    elif method == "linear":
        if years_old == 0:
            return initial_value * 0.90

        return initial_value * (1 - 0.1 * years_old)

    elif method == "experian":
        # adapted from https://www.experian.com/blogs/ask-experian/what-is-depreciation-on-a-car/

        # drops 5% in value the moment it is purchased
        if years_old == 0:
            return initial_value * 0.95

        # drops 10% first year
        final_value = initial_value * 0.9

        # each addtional year drops 10% in first 5 years
        for _ in range(1, min(years_old, 5)):
            final_value -= final_value * 0.10

        for _ in range(5, min(years_old, 10)):
            final_value -= final_value * 0.1

        return final_value


# remove compile from the loop
UPPERCASE_REGEX = re.compile(r"[A-Z]")


def upper_count_regex(text):
    "Returns count of uppercase letters in string"
    return len(UPPERCASE_REGEX.findall(text))


def get_trailing_digit_index(text):
    "Returns index of first digit following alphabetical characters"
    match = re.search(r"(?<=[a-z])\d", text)
    if match:
        return match.start()
    else:
        return 0


def get_regex_index(regex, text) -> int:
    """Gets the index of the first regex match, if any

    Args:
        regex: pattern to use
        text: string to match against
    Returns:
        index of first match in `text` if any, else -1
    """
    match = re.search(regex, text)
    if match:
        return match.start()
    else:
        return -1


def parse_model(text: str) -> str:
    """Attempts to match input string to cars.com-compliant model name

    TODO: reliable with Toyota, mixed results with other makes

    Args:
        text: text to match
    Returns:
        parsed model name]
    """
    text = text.replace(",", "").replace('"', "").replace("-", "_")

    if text[:2] == "FJ":
        return "FJ_cruiser"

    # Assume short names are already clean
    if len(text) < 4:
        return text

    # Special cases
    if text[:2] == "86":
        return "86"

    if text[:4] == "T100":
        return "T100"

    if text[:4] == "RAV4":
        return "RAV4"

    if text[:4] == "C_HR":
        return "C_HR"

    if text[:4] == "Land":
        return "Land_cruiser"

    # Model
    if text.isalpha() and upper_count_regex(text) == 1:
        return text

    # ModelTrim
    if text.isalpha() and upper_count_regex(text) != 1:
        regex = r"(?<=[a-z])[A-Z]"
        idx = get_regex_index(regex, text)
        return text[:idx]

    # ModelTrim2
    regex = r"(?<=[a-z])[A-Z]"
    idx = get_regex_index(regex, text)
    if idx > 1:
        return text[:idx]

    # Model4
    if get_trailing_digit_index(text):
        idx = get_trailing_digit_index(text)
        return text[:idx]

    # MODEL4WD
    regex = r"(?i)([2-6]|f|r|a)wd"
    idx = get_regex_index(regex, text)
    if idx > 1:
        return text[:idx]

    # MODEL4dr
    regex = r"(?i)[2-6]dr"
    idx = get_regex_index(regex, text)
    if idx > 1:
        return text[:idx]

    # 1500 (RAM)
    regex = r"^[1-9]500"
    idx = get_regex_index(regex, text)
    if idx == 0:
        return text[:4]

    # Try using any leftovers
    return text


def get_msrp_list(cars: list, verbose: bool = False) -> list:
    """Fetches MSRP value for a list of cars using `msrp_scraper.py`

    Args:
        cars: list of dicts for each car containing `Make`, `Model`, and `Year`
    Returns:
        MSRP values as a list of floats
    """
    batch = []
    for car in cars:
        url = (
            f"https://www.cars.com/research/{car['Make']}-{car['Model']}-{car['Year']}/"
        )
        batch.append(url.lower())

    with open("test_in", mode="w", encoding="utf-8") as input_file:
        input_file.writelines(line + "\n" for line in batch)

    process = subprocess.run(
        "python msrp_scraper.py test_in test_out",
        check=True,
        shell=True,
        capture_output=True,
        encoding="utf-8",
    )
    if verbose:
        print(str(process.stdout))

    with open("test_out", mode="r", encoding="utf-8") as result_file:
        msrps = result_file.read().splitlines()

    return msrps


if __name__ == "__main__":
    # for i in range(6):
    #     print(standard_depreciation(40000, i))

    # urls = generate_test_urls()[:10]
    # print("hi")

    # print(batch_scrape_msrp(urls))

    df1 = pd.read_csv("true_car_cleaned.csv", index_col=0)
    df1 = df1[["Price", "Year", "MSRP"]]
    # for i in range(7, 9):\
    i = 2017
    methods = ["carfax", "ramsey", "linear", "experian"]
    methods = ["experian"]
    for method in methods:
        # print(f"last_year={i}", end="\t")
        print(f"method={method}", end="\t")
        df = df1.copy()
        df["expected"] = get_expected_value(df, last_year=i, method=method)
        print(f"expected: {df['expected'].mean()}", end="\t")
        print(f"actual: {df['Price'].mean()}", end="\t")
        print(f"retain: {(df['Price'] > df['expected']).mean()}")
        # print(df)

    # print("Testing utilities...")
    # l = [{"Year": 2020, "Make": "Homda", "Model": "Civic"}]
    # print(get_msrp_list(l))
    # exit()

    # with open("all_toyota.csv") as f:
    #     lines = f.readlines()[1:]
    #     lines = [line.rstrip() for line in lines]
    #     models = set(lines)

    # # print(len(models))
    # # print(models)
    # unclean = models.copy()
    # cleaned = set()
    # for model in models:
    #     clean_model = parse_model(model)
    #     if clean_model:
    #         unclean.remove(model)
    #         print(f"{model} -> {clean_model}")
    #         cleaned.add(clean_model)

    # print(unclean)
    # print(cleaned)
