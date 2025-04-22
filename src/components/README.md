# Configuration and Data Management Components

The Hotel Reservation ML project includes several powerful components for data handling, processing, and model management. Here's an overview of the key modules:

## Configuration Management (`components/config.py`)

The `ProjectConfig` class provides a robust configuration system built on Pydantic:

- **Environment Support**: Load different configurations for dev, test, or production
- **YAML Integration**: Read settings from structured YAML files
- **Type Validation**: Enforce proper data types for all configuration values
- **Feature Definition**: Support for detailed feature specifications with name/alias mappings

```python
# Example usage
config = ProjectConfig.from_yaml("path/to/config.yml", env="dev")
```

### Key Configuration Elements

- **Data Sources**: Configure input data locations
- **Feature Definitions**: Define numeric and categorical features with mappings
- **Model Parameters**: Set hyperparameters for training
- **Databricks Integration**: Configure catalog, schema, and endpoint names
- **Deployment Settings**: Manage endpoint and pipeline IDs

### Tags Support

The `Tags` class tracks important metadata for MLOps traceability:

- Git commit SHA for version control
- Branch name for deployment tracking
- Job run ID for workflow integration

## Data Processing Pipeline

The project includes comprehensive data handling capabilities:

1. **DataReader**: Flexible data ingestion from multiple sources
   - CSV files (Spark and pandas-based)
   - Databricks tables and SQL queries
   - Support for custom parsing options

2. **DataProcessor**: Standardized preprocessing workflows
   - Column validation and renaming
   - Data type conversion
   - Null value handling
   - Custom transformation support

3. **DataSplitter**: Multiple strategies for creating train/test sets
   - Random splitting
   - Stratified splitting for classification tasks
   - Supports validation set creation

4. **DataWriter**: Efficient data persistence to Databricks
   - Delta table integration
   - Automatic timestamp addition
   - Change Data Feed support
   - Table management (create, update, truncate)

## Integration Points

These modules integrate seamlessly with:

- **FeatureLookupModel**: For feature engineering and model training
- **FeatureLookupServing**: For model deployment and online inference
- **MLflow Toolkit**: For experiment tracking and model registry
