import pytest
import pandas as pd
import numpy as np
from components.config import ProjectConfig
from components.data_processor import DataProcessor


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'booking_status': ['Not_Canceled', 'Canceled', 'Not_Canceled'],
        'num_feature1': ['10', '20', '30'],
        'num_feature2': ['100', '200', '300'],
        'cat_feature1': ['A', 'B', 'C'],
        'cat_feature2': [1, 2, 3]  # Intentionally numeric to test conversion
    })

@pytest.fixture
def config() -> ProjectConfig:
    """Create a ProjectConfig instance for testing."""
    return ProjectConfig(
        input_data='dummy_path.csv',
        num_features=['num_feature1', 'num_feature2'],
        cat_features=['cat_feature1', 'cat_feature2'],
        target='booking_status',
        parameters={'param1': 'value1'},
        test_size=0.2,
        random_state=42
    )

@pytest.fixture
def processor(sample_data, config) -> DataProcessor:
    """Create a DataProcessor instance for testing."""
    return DataProcessor(sample_data.copy(), config)


def test_treat_target(processor):
    """Test target variable transformation."""
    row_not_canceled = pd.Series({'booking_status': 'Not_Canceled'})
    row_canceled = pd.Series({'booking_status': 'Canceled'})

    assert processor._treat_target(row_not_canceled) == 0
    assert processor._treat_target(row_canceled) == 1

def test_treat_num_features(processor):
    """Test numeric features conversion."""
    processor._treat_num_features()
    
    for feature in processor.cfg.num_features:
        # Check if dtype is numeric
        assert pd.api.types.is_numeric_dtype(processor.data[feature])
        # Check if individual values are numpy numeric types
        assert isinstance(processor.data[feature].iloc[0], (np.integer, np.floating))
        # Verify actual values
        if feature == 'num_feature1':
            assert processor.data[feature].tolist() == [10, 20, 30]
        elif feature == 'num_feature2':
            assert processor.data[feature].tolist() == [100, 200, 300]

def test_treat_cat_features(processor):
    """Test categorical features conversion."""
    processor._treat_cat_features()
    
    for feature in processor.cfg.cat_features:
        assert processor.data[feature].dtype == object
        assert isinstance(processor.data[feature].iloc[0], str)

def test_pre_processing(processor):
    """Test complete preprocessing pipeline."""
    processed_data = processor.pre_processing()
    
    # Check if target was transformed
    assert set(processed_data[processor.cfg.target].unique()) == {0, 1}
    
    # Check numeric features
    for feature in processor.cfg.num_features:
        assert pd.api.types.is_numeric_dtype(processed_data[feature])
    
    # Check categorical features
    for feature in processor.cfg.cat_features:
        assert processed_data[feature].dtype == object


def test_split_data(processor):
    """Test data splitting functionality."""
    processor.pre_processing()  # Preprocess before splitting
    train, test = processor.split_data()
    
    # Check if splits are correct
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    
    # Check split sizes
    assert len(test) < len(train)
