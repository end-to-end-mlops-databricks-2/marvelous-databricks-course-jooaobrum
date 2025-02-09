import pandas as pd
import pytest

from components.config import ProjectConfig, Tags
from components.models.custom_model import CustomModel  # Replace 'your_script' with the actual script name


@pytest.fixture
def mock_config():
    return ProjectConfig(
        input_data="dummy_path.csv",
        num_features=["num_feature1", "num_feature2"],
        cat_features=["cat_feature1", "cat_feature2"],
        target="booking_status",
        test_size=0.2,
        random_state=42,
        parameters={"n_estimators": 100},
        catalog_name="test_catalog",
        schema_name="test_schema",
        experiment_name="test_experiment",
    )


@pytest.fixture
def mock_tags():
    return Tags(git_sha="123", branch="test")


@pytest.fixture
def mock_model(mock_config, mock_tags):
    return CustomModel(mock_config, mock_tags, None, code_paths=["test_path.whl"])  # No Spark dependency


@pytest.fixture
def fake_data():
    return pd.DataFrame(
        {
            "booking_status": ["Not_Canceled", "Canceled", "Not_Canceled"],
            "num_feature1": ["10", "20", "30"],
            "num_feature2": ["100", "200", "300"],
            "cat_feature1": ["A", "B", "C"],
            "cat_feature2": [1, 2, 3],  # Intentionally numeric to test conversion
        }
    )


def test_load_data(mock_model, fake_data):
    mock_model.train_set = fake_data.copy()
    mock_model.test_set = fake_data.copy()

    mock_model.X_train = mock_model.train_set[mock_model.config.num_features + mock_model.config.cat_features]
    mock_model.y_train = mock_model.train_set[mock_model.config.target]

    assert not mock_model.train_set.empty
    assert "num_feature1" in mock_model.X_train.columns
    assert "num_feature2" in mock_model.X_train.columns
    assert "cat_feature1" in mock_model.X_train.columns
    assert "cat_feature2" in mock_model.X_train.columns
    assert "booking_status" in mock_model.y_train.name


def test_prepare_features(mock_model):
    mock_model.prepare_features()

    assert mock_model.preprocessor is not None
    assert mock_model.pipeline is not None
