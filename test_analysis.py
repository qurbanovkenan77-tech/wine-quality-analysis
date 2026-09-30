"""Tests for wine data preparation and analysis."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from analysis import (
    clean_data,
    evaluate_models,
    get_average_quality,
    get_high_quality_wines,
    load_data,
    summarize_outliers,
    train_model,
)


@pytest.fixture
def sample_data():
    """Small dataset with known values for testing."""
    return pd.DataFrame(
        {
            "alcohol": [10.0, 11.0, 12.0, 13.0],
            "quality": [5, 6, 7, 8],
            "type": ["red", "red", "white", "white"],
        }
    )


def test_load_real_data():
    df = load_data()

    assert df.shape == (6497, 13)
    assert {"quality", "alcohol", "type"}.issubset(df.columns)


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")


def test_clean_real_data():
    cleaned = clean_data(load_data())

    assert len(cleaned) == 5320
    assert not cleaned.duplicated().any()
    assert not cleaned.isna().any().any()
    assert len(get_high_quality_wines(cleaned)) == 1009


def test_clean_removes_duplicates_without_changing_input(sample_data):
    duplicated = pd.concat(
        [sample_data, sample_data.iloc[[0]]],
        ignore_index=True,
    )
    original = duplicated.copy(deep=True)

    cleaned = clean_data(duplicated)

    assert_frame_equal(cleaned, sample_data)
    assert_frame_equal(duplicated, original)


def test_clean_is_repeatable(sample_data):
    cleaned = clean_data(sample_data)

    assert_frame_equal(clean_data(cleaned), cleaned)


def test_clean_rejects_empty_data(sample_data):
    with pytest.raises(ValueError, match="empty"):
        clean_data(sample_data.iloc[:0])


@pytest.mark.parametrize("column", ["alcohol", "quality", "type"])
def test_clean_rejects_missing_column(sample_data, column):
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_data(sample_data.drop(columns=column))


def test_clean_rejects_missing_values(sample_data):
    sample_data.loc[0, "alcohol"] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        clean_data(sample_data)


@pytest.mark.parametrize("column", ["alcohol", "quality"])
def test_clean_rejects_text_in_numeric_columns(sample_data, column):
    sample_data[column] = sample_data[column].astype(str)

    with pytest.raises(ValueError, match="must be numeric"):
        clean_data(sample_data)


@pytest.mark.parametrize("value", [0, -1, 100, float("inf")])
def test_clean_rejects_invalid_alcohol(sample_data, value):
    sample_data.loc[0, "alcohol"] = value

    with pytest.raises(ValueError, match="Alcohol must"):
        clean_data(sample_data)


@pytest.mark.parametrize("value", [-1, 11, 6.5, float("inf")])
def test_clean_rejects_invalid_quality(sample_data, value):
    sample_data["quality"] = sample_data["quality"].astype(float)
    sample_data.loc[0, "quality"] = value

    with pytest.raises(ValueError, match="Quality must"):
        clean_data(sample_data)


def test_clean_rejects_unknown_wine_type(sample_data):
    sample_data.loc[0, "type"] = "rose"

    with pytest.raises(ValueError, match="Wine type must"):
        clean_data(sample_data)


def test_high_quality_includes_threshold(sample_data):
    result = get_high_quality_wines(sample_data)

    assert result["quality"].tolist() == [7, 8]


def test_high_quality_returns_empty_when_none_qualify(sample_data):
    result = get_high_quality_wines(sample_data.iloc[:2])

    assert result.empty
    assert result.columns.tolist() == sample_data.columns.tolist()


def test_high_quality_handles_empty_input(sample_data):
    result = get_high_quality_wines(sample_data.iloc[:0])

    assert result.empty


def test_average_quality_matches_known_values(sample_data):
    averages = get_average_quality(sample_data)

    assert averages["red"] == pytest.approx(5.5)
    assert averages["white"] == pytest.approx(7.5)


def test_outliers_are_checked_separately_by_type():
    df = pd.DataFrame(
        {
            "alcohol": [10.0] * 8 + [20.0] * 8,
            "quality": [6] * 16,
            "type": ["red"] * 8 + ["white"] * 8,
        }
    )

    report = summarize_outliers(df)

    assert len(report) == 2
    assert report["flagged_count"].eq(0).all()
    assert set(report["feature"]) == {"alcohol"}


def test_outlier_report_flags_extreme_value_without_removing_it():
    df = pd.DataFrame(
        {
            "alcohol": [10.0, 10.1, 10.2, 10.3, 20.0],
            "quality": [5, 6, 5, 6, 7],
            "type": ["red"] * 5,
        }
    )
    original = df.copy(deep=True)

    report = summarize_outliers(df)

    assert report.iloc[0]["flagged_count"] == 1
    assert_frame_equal(df, original)


def test_model_learns_known_linear_relationship():
    alcohol = list(range(8, 18))
    df = pd.DataFrame(
        {
            "alcohol": alcohol,
            "quality": [0.5 * value for value in alcohol],
        }
    )

    model, mse = train_model(df)

    assert model.coef_[0] == pytest.approx(0.5)
    assert model.intercept_ == pytest.approx(0.0, abs=1e-10)
    assert mse < 1e-10


def test_complete_cleaned_workflow():
    cleaned = clean_data(load_data())

    high_quality = get_high_quality_wines(cleaned)
    averages = get_average_quality(cleaned)
    outliers = summarize_outliers(cleaned)
    model, mse = train_model(cleaned)

    assert len(high_quality) == 1009
    assert set(averages.index) == {"red", "white"}
    assert len(outliers) == 22
    assert model.n_features_in_ == 1
    assert 0 <= mse < 1


def test_baseline_uses_training_mean():
    """Verify baseline metrics against a manually calculated example."""
    df = pd.DataFrame(
        {
            "alcohol": list(range(8, 18)),
            "quality": list(range(10)),
        }
    )

    _, results = evaluate_models(df)

    # With 10 rows, test_size=0.2, and random_state=42:
    # test rows are 8 and 1; all remaining rows are training data.
    training_quality = [0, 2, 3, 4, 5, 6, 7, 9]
    test_quality = [8, 1]
    training_mean = sum(training_quality) / len(training_quality)

    expected_mse = sum((actual - training_mean) ** 2 for actual in test_quality) / len(
        test_quality
    )

    expected_mae = sum(abs(actual - training_mean) for actual in test_quality) / len(
        test_quality
    )

    baseline = results.loc["Mean baseline"]

    assert baseline["MSE"] == pytest.approx(expected_mse)
    assert baseline["RMSE"] == pytest.approx(expected_mse**0.5)
    assert baseline["MAE"] == pytest.approx(expected_mae)

    # The synthetic data have an exact linear relationship.
    assert results.loc["Alcohol regression", "MSE"] < 1e-10
    assert results.loc["Alcohol regression", "R2"] == pytest.approx(1.0)
