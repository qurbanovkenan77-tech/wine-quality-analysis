# importing necessary libraries
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# functions for testing

def load_data(filename="wine_quality_merged.csv"):
    return pd.read_csv(filename)


def get_high_quality_wines(df):
    return df[df["quality"] >= 7]


def get_average_quality(df):
    return df.groupby("type")["quality"].mean()


def train_model(df):
    # input and output
    X = df[["alcohol"]]
    y = df["quality"]

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # make predictions
    predictions = model.predict(X_test)

    # evaluate the model
    mse = mean_squared_error(y_test, predictions)

    return model, mse


if __name__ == "__main__":

    start = time.perf_counter()

    df = load_data("wine_quality_merged.csv")

    end = time.perf_counter()

    print("\nPandas CSV Load Time:")
    print(end - start)

    print(df.head())

    # inspect the data
    print("\nDataset Information:")
    df.info()

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nNumber of Duplicates:")
    print(df.duplicated().sum())

    # basic filtering and grouping
    start = time.perf_counter()

    high_quality = get_high_quality_wines(df)
    average_quality = get_average_quality(df)

    end = time.perf_counter()

    print("\nPandas Analysis Time:")
    print(end - start)

    # visualization
    plt.hist(df["quality"], bins=6, edgecolor="black")

    plt.xlabel("Wine Quality")
    plt.ylabel("Number of Wines")
    plt.title("Distribution of Wine Quality")

    plt.savefig("wine_quality_distribution.png")
    plt.close()

    # machine learning
    model, mse = train_model(df)

    print("\nMachine Learning - Linear Regression")
    print("Input: Alcohol")
    print("Output: Quality")
    print("Mean Squared Error:", mse)