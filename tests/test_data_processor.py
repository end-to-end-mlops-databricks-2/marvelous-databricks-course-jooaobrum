import numpy as np
import pandas as pd
import pytest
from components.config import ProjectConfig
from components.data_processor import DataProcessor

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "booking_status": ["Not_Canceled", "Canceled", "Not_Canceled"],
            "num_feature1": ["10", "20", "30"],
            "num_feature2": ["100", "200", "300"],
            "cat_feature1": ["A", "B", "C"],
            "cat_feature2": [1, 2, 3],  # Intentionally numeric to test conversion
        }
    )

@pytest.fixture
def config() -> ProjectConfig:
    """Create a ProjectConfig instance with dictionary-based features for testing."""
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
        parameters={"param1": "value1"},
        test_size=0.2,
        random_state=42,
        endpoint_name="test-endpoint",
        id_columns=["id"],
        catalog_name="uc_test",
        schema_name="hotel_reservation",
        experiment_name="hotel_booking_test"
    )

@pytest.fixture
def processor(sample_data, config) -> DataProcessor:
    """Create a DataProcessor instance for testing."""
    return DataProcessor(sample_data.copy(), config)

def test_validate_columns(processor):
    """Test column validation."""
    # This should pass without raising an exception
    processor.validate_columns()
    
    # Test with missing column
    processor.data = processor.data.drop(columns=["num_feature1"])
    with pytest.raises(ValueError):
        processor.validate_columns()

def test_rename_columns(processor, sample_data):
    """Test column renaming when aliases differ from names."""
    # First verify original column names
    assert "num_feature1" in processor.data.columns
    assert "num_feature2" in processor.data.columns
    
    # Apply renaming
    processor.rename_columns()
    
    # Check if columns were renamed
    assert "num1" in processor.data.columns
    assert "num2" in processor.data.columns
    assert "cat_feature1" in processor.data.columns
    assert "cat_feature2" in processor.data.columns
    assert "booking_status" in processor.data.columns
    
    # Check that original column names are gone (if your implementation drops them)
    assert "num_feature1" not in processor.data.columns
    assert "num_feature2" not in processor.data.columns

def test_convert_datatypes(processor):
    """Test data type conversion."""
    # First rename columns
    processor.rename_columns()
    # Then convert data types
    processor.convert_datatypes()
    
    # Check numeric features are converted to integers (using aliases)
    assert pd.api.types.is_integer_dtype(processor.data["num1"])
    assert pd.api.types.is_integer_dtype(processor.data["num2"])
    
    # Check categorical features are converted to category type
    assert pd.api.types.is_categorical_dtype(processor.data["cat_feature1"])
    assert pd.api.types.is_categorical_dtype(processor.data["cat_feature2"])
    
    # Check if values were correctly converted
    assert processor.data["num1"].tolist() == [10, 20, 30]
    assert processor.data["num2"].tolist() == [100, 200, 300]
    
    # Check if target was mapped correctly
    assert processor.data["booking_status"].tolist() == [0, 1, 0]

def test_check_null_values(processor):
    """Test null value detection."""
    # Test with no nulls
    processor.check_null_values(raise_error=False)
    
    # Test with nulls
    processor.data.loc[0, "num_feature1"] = None
    processor.data.loc[1, "cat_feature1"] = None
    
    # Should not raise error because less than 25% nulls
    processor.check_null_values(raise_error=False)
    
    # Now introduce enough nulls to cross the threshold
    for i in range(3):
        processor.data.loc[i, "num_feature2"] = None
    
    # This should raise an error due to high percentage of nulls
    with pytest.raises(ValueError):
        processor.check_null_values(raise_error=True)

def test_split_data(processor):
    """Test data splitting functionality."""
    processor.pre_processing()  # Preprocess before splitting
    train, test = processor.split_data()
    
    # Check if splits are correct
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    
    # Check split sizes
    assert len(train) == 2  # With 3 rows and 0.2 test_size, train should have 2 rows
    assert len(test) == 1   # With 3 rows and 0.2 test_size, test should have 1 row
    
    # Check if the data was correctly split
    assert set(train.index).isdisjoint(set(test.index))