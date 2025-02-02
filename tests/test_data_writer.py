from unittest.mock import Mock, patch
import pytest
import pandas as pd
from pyspark.sql import SparkSession
from components.config import ProjectConfig
from components.data_writer import DataWriter


@pytest.fixture
def mock_spark():
    """Fixture to create a mock SparkSession."""
    return Mock(spec=SparkSession)


@pytest.fixture
def mock_config():
    """Fixture to create a mock ProjectConfig."""
    return ProjectConfig(
        input_data="/dummy/path.csv",
        test_size=0.2,
        random_state=42,
        num_features=["feature1", "feature2"],
        cat_features=["category1"],
        target="target",
        catalog_name="test_catalog",
        schema_name="test_schema"
    )


@pytest.fixture
def data_writer(mock_config):
    """Fixture to create a DataWriter instance."""
    return DataWriter(mock_config)


def test_save_to_catalog(data_writer, mock_spark):
    """Test the save_to_catalog method of DataWriter."""

    # Create a dummy pandas DataFrame
    df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})

    # Mock Spark DataFrame creation and chained calls
    mock_spark_df = Mock()
    mock_spark.createDataFrame.return_value = mock_spark_df

    # ✅ Explicitly mock write.mode() -> saveAsTable()
    mock_spark_df.write.mode.return_value = mock_spark_df.write
    mock_spark_df.write.mode.return_value.saveAsTable.return_value = None

    # Call the function
    data_writer.save_to_catalog(df, "test_table", mock_spark)

    # ✅ Ensure `createDataFrame()` was called once
    mock_spark.createDataFrame.assert_called_once_with(df)

    # ✅ Verify that `write.mode("append")` was called
    mock_spark_df.write.mode.assert_called_once_with("append")

    # ✅ Ensure that the DataFrame was saved to the correct table
    mock_spark_df.write.mode.return_value.saveAsTable.assert_called_once_with(
        "test_catalog.test_schema.test_table"
    )

    # ✅ Check if the ALTER TABLE command was executed
    mock_spark.sql.assert_called_with(
        "ALTER TABLE test_catalog.test_schema.test_table SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
    )
