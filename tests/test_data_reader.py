import pytest
import os
from components.config import ProjectConfig
from components.data_reader import DataReader


@pytest.fixture
def create_test_csv():
    file_path = 'test.csv'
    with open(file_path, 'w') as f:
        f.write("A,B\n")
        f.write("1,10\n")
        f.write("2,20\n")
        f.write("3,30\n")
        f.write("4,40\n")
        f.write("5,50\n")
    yield file_path
    
@pytest.fixture
def mock_config(create_test_csv):
    """Create a mock ProjectConfig with all required fields."""
    return ProjectConfig(
        input_data=create_test_csv,
        num_features=['column1'],  # Numeric features
        cat_features=['column2'],  # Categorical features
        target='column1',         # Target variable
        parameters={              # Required parameters
            'some_param': 'value'
        }
    )





def test_read_csv(mock_config):

    reader = DataReader(mock_config)

    df = reader.read_csv()

    assert df.shape == (5, 2)
    assert df.columns.tolist() == ["A", "B"]
    assert df["A"].tolist() == [1, 2, 3, 4, 5]
    assert df["B"].tolist() == [10, 20, 30, 40, 50]

    os.remove(mock_config.input_data)