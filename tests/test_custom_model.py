import pandas as pd
import pytest

from components.config import ProjectConfig, Tags
from components.models.custom_model import CustomModel  # Replace 'your_script' with the actual script name


@pytest.fixture
def mock_config():
    return ProjectConfig(
        input_data="dummy_path.csv",
        num_features=[
            {"name": "num_feature1", "alias": "num1", "dtype": "int32"},
            {"name": "num_feature2", "alias": "num2", "dtype": "int32"}
        ],
        cat_features=[
            {"name": "cat_feature1", "alias": "cat_feature1", "dtype": "category"},
            {"name": "cat_feature2", "alias": "cat_feature2", "dtype": "category"}
        ],
        target={
            "name": "booking_status",
            "alias": "booking_status",
            "dtype": "int32",
            "mapping": {"Canceled": 1, "Not_Canceled": 0}
        },
        test_size=0.2,
        random_state=42,
        parameters={"n_estimators": 100},
        endpoint_name="test-endpoint",
        id_columns=["id"],
        catalog_name="uc_test",
        schema_name="hotel_reservation",
        experiment_name="hotel_booking_test"
    )


@pytest.fixture
def mock_tags():
    return Tags(git_sha="123", branch="test", job_run_id = "test")


@pytest.fixture
def mock_model(mock_config, mock_tags):
    return CustomModel(mock_config, mock_tags, None, code_paths=["test_path.whl"])  # No Spark dependency


@pytest.fixture
def fake_data():
    return pd.DataFrame(
        {
            "booking_status": [0, 1, 0],
            "num1": ["10", "20", "30"],
            "num2": ["100", "200", "300"],
            "cat_feature1": ["A", "B", "C"],
            "cat_feature2": [1, 2, 3],  # Intentionally numeric to test conversion
        }
    )


def test_load_data(mock_model, fake_data):
    mock_model.train_set = fake_data.copy()
    mock_model.test_set = fake_data.copy()

    features = [feat['alias'] for feat in mock_model.config.num_features + mock_model.config.cat_features]
    target = mock_model.config.target['alias']

    mock_model.X_train = mock_model.train_set[features]
    mock_model.y_train = mock_model.train_set[target]

    assert not mock_model.train_set.empty
    assert "num1" in mock_model.X_train.columns
    assert "num2" in mock_model.X_train.columns
    assert "cat_feature1" in mock_model.X_train.columns
    assert "cat_feature2" in mock_model.X_train.columns
    assert "booking_status" in mock_model.y_train.name


def test_prepare_features(mock_model):
    mock_model.prepare_features()

    assert mock_model.preprocessor is not None
    assert mock_model.pipeline is not None
