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
