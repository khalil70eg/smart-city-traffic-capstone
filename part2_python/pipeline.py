import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
#-------------------------------------------------------------------------
# Logging configuration module logger setup
#-------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
file_handler = logging.FileHandler("pipeline.log", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

#-------------------------------------------------------------------------
# Configuration for the pipeline
#-------------------------------------------------------------------------
EXPECTED_COLUMNS = ["holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main",
                     "weather_description", "date_time", "traffic_volume"]
DATE_COLUMN = "date_time"
CATEGORICAL_COLUMNS = ["holiday", "weather_main", "weather_description"]
NUMERICAL_COLUMNS = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]

#-------------------------------------------------------------------------
# Task 1: Data Pipeline construction ( with logging and initial inspection)
# 1.1 Load the raw CSV data
#-------------------------------------------------------------------------
"""load the raw csv file, use eception handling with try-except block 
to handle file I/O and parsing errors and do not use bare except: statements."""
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file with exception handling."""
    logger.info("Loading data from %s", file_path)
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error("File not found: %s", file_path, exc_info=True)
        raise
    except PermissionError:
        logger.error("Permission denied reading: %s", file_path, exc_info=True)
        raise
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty: %s", file_path, exc_info=True)
        raise
    except pd.errors.ParserError:
        logger.error("Error parsing CSV file: %s", file_path, exc_info=True)
        raise
    except UnicodeDecodeError:
        logger.error("File is not valid UTF-8 text: %s", file_path, exc_info=True)
        raise

    logger.info("Data loaded successfully: %d rows, %d columns.", df.shape[0], df.shape[1])
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info("Missing values per column:\n%s", missing.to_string())
    else:
        logger.info("No missing values found.")
    return df

#-------------------------------------------------------------------------
# End  1.1 Load the raw CSV data
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# 1.2 Validate the data schema 
#-------------------------------------------------------------------------

# o	Validate the schema before performing any other data-processing operations. 
# o	Check that all expected columns are present in the dataset.

def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the DataFrame contains all expected columns."""
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        logger.error("Missing expected columns: %s", missing_columns)
        raise ValueError(f"Missing expected columns: {missing_columns}")
    logger.info("All expected columns are present.")
#-------------------------------------------------------------------------
# End1.2 Validate the data schema 
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# 1.3 clean the data
#-------------------------------------------------------------------------
# o	Standardise inconsistent categorical values. 
# o	Parse and validate date/time fields. 
# o	Identify and remove duplicate rows. 
# o	Detect and handle outliers or impossible values. 
# o	For example, identify sensor readings showing 0 Kelvin or rainfall values exceeding the physically plausible range of 9K mm. 
# o	Impute identified outliers using the median where appropriate.
# o	logger should be used to log warnings and errors during the cleaning process.



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by standardising categorical values, parsing dates, removing duplicates, and handling outliers."""
    # Standardise categorical values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()

    # Parse and validate date/time fields
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except (ValueError, TypeError):
            logger.warning("Column %s could not be parsed as datetime.", col)

    # Remove duplicate rows and log the number of duplicates removed
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        logger.warning("Removing %d duplicate rows.", num_duplicates)   
    df = df.drop_duplicates()

    # Detect and handle outliers and log any replacements
    for col in df.select_dtypes(include='number').columns:
        if col == 'temperature':
            num_outliers = (df[col] <= 0).sum()
            if num_outliers > 0:
                logger.warning("Replacing %d outliers in column %s with median.", num_outliers, col)
            df.loc[df[col] <= 0, col] = df[col].median()
        if col == 'rainfall':
            num_outliers = (df[col] > 9000).sum()
            if num_outliers > 0:
                logger.warning("Replacing %d outliers in column %s with median.", num_outliers, col)
            df.loc[df[col] > 9000, col] = df[col].median()

    return df
#-------------------------------------------------------------------------
# End 1.3 clean the data
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# 1.4 Use conditionals and loops where appropriate
#-------------------------------------------------------------------------
# o	Use conditionals and loops explicitly when row-by-row or group-by-group processing is required. 
# o	For example, when imputing values by month, calculate and apply the appropriate monthly value rather than using a single global average for the entire dataset.
# o logger should be used to log warnings and errors during the imputation process.

def impute_missing_by_month(df: pd.DataFrame, date_column: str = DATE_COLUMN,
                             columns: list = None) -> pd.DataFrame:
    """Impute missing numerical values using each column's monthly mean,
    rather than a single global average for the entire dataset."""
    if columns is None:
        columns = NUMERICAL_COLUMNS

    if date_column not in df.columns:
        logger.error("Date column %s not found; cannot perform monthly imputation.", date_column)
        return df

    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        logger.warning("Column %s is not datetime type; converting before imputation.", date_column)
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    months = df[date_column].dt.month

    # Explicit loop: process one numerical column at a time
    for col in columns:
        if col not in df.columns:
            logger.warning("Column %s not found in DataFrame; skipping imputation.", col)
            continue

        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue

        logger.info("Column %s has %d missing values; imputing by month.", col, missing_count)

        # Explicit loop: process one month-group at a time (group-by-group)
        for month in range(1, 13):
            month_mask = months == month
            if not month_mask.any():
                continue

            month_missing_mask = month_mask & df[col].isnull()
            if not month_missing_mask.any():
                continue

            month_values = df.loc[month_mask, col].dropna()

            # Conditional: guard against a month with no valid values at all
            if month_values.empty:
                logger.error(
                    "No valid values available for column %s in month %d; "
                    "cannot calculate a monthly mean for imputation.", col, month
                )
                continue

            month_mean = month_values.mean()
            num_to_fill = int(month_missing_mask.sum())

            df.loc[month_missing_mask, col] = month_mean

            logger.warning(
                "Imputed %d missing value(s) in column %s for month %d using monthly mean %.2f.",
                num_to_fill, col, month, month_mean
            )

    return df
#-------------------------------------------------------------------------
# End 1.4 Use conditionals and loops where appropriate
#-------------------------------------------------------------------------





