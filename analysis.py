"""Analyze wine quality using Pandas and linear regression."""

import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def load_data(filename="wine_quality_merged.csv"):
    """Load the wine dataset from a CSV file."""
    return pd.read_csv(filename)


def get_high_quality_wines(df):
    """Return wines with a quality score of at least 7."""
    return df.loc[df["quality"] >= 7].copy()


def get_average_quality(df):
    """Calculate the mean quality score for each wine type."""
    return df.groupby("type")["quality"].mean()


def clean_data(df):
    """Validate required fields and remove exact duplicate rows."""
    required_columns = {"alcohol", "quality", "type"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {names}")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df.isna().any().any():
        raise ValueError("The dataset contains missing values.")

    for column in ["alcohol", "quality"]:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

    if not df["alcohol"].between(0, 100, inclusive="neither").all():
        raise ValueError("Alcohol must be greater than 0 and less than 100.")

    valid_quality = df["quality"].between(0, 10) & (
        df["quality"] == df["quality"].round()
    )
    if not valid_quality.all():
        raise ValueError("Quality must contain integer scores from 0 to 10.")

    if not df["type"].isin(["red", "white"]).all():
        raise ValueError("Wine type must be 'red' or 'white'.")

    return df.drop_duplicates().reset_index(drop=True)


def summarize_outliers(df):
    """Count IQR outlier flags by wine type without removing observations."""
    chemical_columns = df.select_dtypes(include="number").columns.drop("quality")
    records = []

    for wine_type, group in df.groupby("type"):
        for column in chemical_columns:
            first_quartile = group[column].quantile(0.25)
            third_quartile = group[column].quantile(0.75)
            iqr = third_quartile - first_quartile

            lower_bound = first_quartile - 1.5 * iqr
            upper_bound = third_quartile + 1.5 * iqr

            flagged = (group[column] < lower_bound) | (group[column] > upper_bound)

            records.append(
                {
                    "type": wine_type,
                    "feature": column,
                    "flagged_count": int(flagged.sum()),
                }
            )

    return pd.DataFrame(records)


def inspect_data(df):
    """Print dataset structure and basic data-quality checks."""
    print("\nFirst Five Rows:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isna().sum())

    print("\nNumber of Duplicate Rows:")
    print(df.duplicated().sum())


def plot_quality_distribution(df, filename="wine_quality_distribution.png"):
    """Save a histogram of wine quality scores."""
    fig, ax = plt.subplots()

    ax.hist(df["quality"], bins=6, edgecolor="black")
    ax.set_xlabel("Wine Quality")
    ax.set_ylabel("Number of Wines")
    ax.set_title("Distribution of Wine Quality")

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def train_model(df):
    """Fit an alcohol-only regression and return the model and test MSE."""
    features = df[["alcohol"]]
    target = df["quality"]

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(features_train, target_train)

    predictions = model.predict(features_test)
    mse = mean_squared_error(target_test, predictions)

    return model, mse


def main():
    """Run data inspection, summary analysis, plotting, and modeling."""
    start = time.perf_counter()
    df = load_data()
    load_time = time.perf_counter() - start

    print(f"\nPandas CSV Load Time: {load_time:.6f} seconds")
    inspect_data(df)
    original_rows = len(df)
    df = clean_data(df)

    print("\nData Cleaning:")
    print(f"Original rows: {original_rows}")
    print(f"Duplicate rows removed: {original_rows - len(df)}")
    print(f"Cleaned rows: {len(df)}")

    print("\nOutlier Flags by Wine Type:")
    print(summarize_outliers(df).to_string(index=False))
    print("Flagged observations are retained in the analysis.")
    start = time.perf_counter()
    high_quality = get_high_quality_wines(df)
    average_quality = get_average_quality(df)
    analysis_time = time.perf_counter() - start

    print(f"\nPandas Analysis Time: {analysis_time:.6f} seconds")

    print("\nNumber of High Quality Wines:")
    print(len(high_quality))

    print("\nAverage Quality by Wine Type:")
    print(average_quality)

    plot_quality_distribution(df)

    _, mse = train_model(df)

    print("\nMachine Learning - Linear Regression")
    print("Input: Alcohol")
    print("Output: Quality")
    print(f"Mean Squared Error: {mse:.4f}")


if __name__ == "__main__":
    main()
