import os

import pytest

from components.config import ProjectConfig
from components.data_reader import DataReader
from components.utils import is_databricks


@pytest.fixture
def create_test_csv():
    if not os.path.exists("tests_data"):
        os.makedirs("tests_data")

    file_path = "tests_data/test.csv"
    with open(file_path, "w") as f:
        f.write("A,B\n")
        f.write("1,10\n")
        f.write("2,20\n")
        f.write("3,30\n")
        f.write("4,40\n")
        f.write("5,50\n")

    yield file_path


@pytest.fixture
def mock_config(create_test_csv):
    """Create a mock ProjectConfig with all required fields using dictionary-based features."""
    return ProjectConfig(
        input_data=create_test_csv,
        test_size=0.2,
        random_state=42,
        num_features=[
            {"name": "avg_price_per_room", "alias": "avg_price_per_room", "dtype": "float64"},
            {"name": "no_of_special_requests", "alias": "no_of_special_requests", "dtype": "int16"},
        ],
        cat_features=[
            {"name": "type_of_meal_plan", "alias": "type_of_meal_plan", "dtype": "category"},
            {"name": "room_type_reserved", "alias": "room_type_reserved", "dtype": "category"},
            {"name": "market_segment_type", "alias": "market_segment_type", "dtype": "category"},
        ],
        target={
            "name": "booking_status",
            "alias": "booking_status",
            "dtype": "int16",
            "mapping": {"Canceled": 1, "Not_Canceled": 0},
        },
        parameters={"random_state": 42, "n_estimators": 300, "max_depth": 6, "learning_rate": 0.01},
        endpoint_name="test-endpoint",
        id_columns=["id"],
        catalog_name="uc_test",
        schema_name="hotel_reservation",
        experiment_name="hotel_booking_test",
    )


@pytest.mark.skipif(is_databricks(), reason="Only Local test")
def test_read_pandas_csv_local(mock_config):
    reader = DataReader(mock_config)
    df = reader.read_pandas_csv(filepath="tests_data/test.csv")  # Fixed file path
    assert df.shape == (5, 2)
    assert df.columns.tolist() == ["A", "B"]
    assert df["A"].tolist() == [1, 2, 3, 4, 5]
    assert df["B"].tolist() == [10, 20, 30, 40, 50]


@pytest.mark.skipif(not is_databricks(), reason="Only Databricks test")
def test_read_spark_csv_databricks(mock_config):
    reader = DataReader(mock_config)
    df = reader.read_spark_csv(filepath="tests_data/test.csv")
    assert df.shape == (5, 2)
    assert df.columns.tolist() == ["A", "B"]
    assert df["A"].tolist() == [1, 2, 3, 4, 5]
    assert df["B"].tolist() == [10, 20, 30, 40, 50]
