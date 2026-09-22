# importing necessary libraries
import os
import pandas as pd

from analysis import (
    load_data,
    get_high_quality_wines,
    get_average_quality,
    train_model,
)


# Test 1: Data loading
def test_load_data():
    df = load_data("wine_quality_merged.csv")

    assert len(df) == 6497
    assert "quality" in df.columns
    assert "alcohol" in df.columns
    assert "type" in df.columns


# Test 2: Filtering high-quality wines
def test_high_quality_filter():
    df = load_data("wine_quality_merged.csv")

    high_quality = get_high_quality_wines(df)

    assert len(high_quality) == 1277
    assert (high_quality["quality"] >= 7).all()


# Test 3: Grouping by wine type
def test_average_quality():
    df = load_data("wine_quality_merged.csv")

    averages = get_average_quality(df)

    assert "red" in averages.index
    assert "white" in averages.index
    assert averages["red"] > 0
    assert averages["white"] > 0


# Test 4: Machine learning model
def test_machine_learning():
    df = load_data("wine_quality_merged.csv")

    model, mse = train_model(df)

    assert model is not None
    assert mse >= 0
    assert mse < 1


# System Test: Run the main parts of the workflow together
def test_complete_workflow():
    df = load_data("wine_quality_merged.csv")

    high_quality = get_high_quality_wines(df)
    averages = get_average_quality(df)
    model, mse = train_model(df)

    assert len(df) > 0
    assert len(high_quality) > 0
    assert len(averages) == 2
    assert model is not None
    assert mse >= 0