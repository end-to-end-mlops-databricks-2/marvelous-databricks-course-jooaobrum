# Hotel Reservation Model Components

This project contains several key modules for building ML models with feature lookup capabilities in Databricks.

## Core Components

### `FeatureLookUpModel`

The central class for creating, training, and registering feature-based ML models:

- Creates and manages feature tables in the Databricks Feature Store
- Loads training/testing data from Delta tables
- Retrieves features from Feature Store using lookups
- Trains classification models with customizable preprocessing
- Evaluates model performance and registers improved models

### `MLflowToolkit`

Simplifies MLflow operations:

- Manages experiment tracking with minimal boilerplate
- Handles model registration in Unity Catalog
- Provides convenient methods for logging metrics and parameters
- Supports feature store integration for model logging
- Offers batch scoring capabilities

### `FeatureStoreWrapper`

Streamlines Feature Store interactions:

- Creates and updates feature tables with simplified interface
- Handles feature lookups for model training
- Manages table schemas and primary keys
- Supports point-in-time lookups for time series data
- Enables Change Data Feed (CDF) for feature table updates

### `ModelFactory` and `MLPipeline`

Tools for building standardized ML pipelines:

- `Preprocessor`: Configurable preprocessing with support for numeric/categorical features
- `ModelFactory`: Creates various ML models with sensible defaults
- `MLPipeline`: Combines preprocessing, resampling, and models into complete pipelines

## Implementation Details

### Data Workflow

1. **Data Preparation**: 
   - Reads data from Databricks tables
   - Supports feature creation and preprocessing
   - Creates and updates feature tables in the Feature Store

2. **Feature Retrieval**:
   - Uses Feature Store lookups for consistent feature access
   - Preserves feature lineage and definitions
   - Handles feature versioning automatically

3. **Model Training**:
   - Supports configurable preprocessing strategies:
     - Numeric features: standard, minmax, or robust scaling
     - Categorical features: one-hot, ordinal, or binary encoding
     - Missing values: mean, median, most frequent, or constant
   - Uses LightGBM as the default classifier
   - Tracks all parameters and metrics in MLflow

## Usage Example

```python
# Initialize model with configuration
model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Create feature table (only needed once)
model.create_feature_table()

# Load data and retrieve features
model.load_data()
model.retrieve_features()

# Train the model
model.train()

# Register if improved
if model.model_improved():
    version = model.register_model()
```

## Customization Options

- **Model Selection**: Change model type via the `ModelFactory`
- **Preprocessing**: Configure strategies in the project config
- **Feature Engineering**: Update the feature table schema as needed
- **Monitoring**: Track model performance over time

## Technical Requirements

- Databricks Runtime 15.4 LTS or higher
- Python 3.11+
- Required packages: mlflow, scikit-learn, pandas, lightgbm, databricks-sdk
- Databricks Feature Store and Unity Catalog access

## Best Practices

- Keep feature definitions consistent across training and inference
- Register improved models only after thorough evaluation
- Use feature lookup for real-time serving
- Monitor model performance with automated refresh jobs
- Leverage Change Data Feed for efficient feature table updates