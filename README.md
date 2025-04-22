# Hotel Reservation ML Project

## Overview

An end-to-end MLOps pipeline for hotel reservation cancellation prediction, including data preprocessing, feature engineering, model training, and deployment with online feature lookup.

## Setup

### Prerequisites

- Python 3.11+
- Databricks workspace with 15.4 LTS runtime
- UV package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/end-to-end-mlops-databricks-2/marvelous-databricks-course-jooaobrum.git
cd marvelous-databricks-course-jooaobrum

# Create environment
uv venv -p 3.11 venv
source venv/bin/activate
uv pip install -r pyproject.toml --all-extras
uv lock
```

## Project Structure

```
marvelous-databricks-course-jooaobrum/
├── files/                        # Configuration files
├── notebooks/                    # Databricks notebooks for development
├── scripts/                      # Automation scripts
├── tests/                        # Unit and integration tests
├── src/components/               # Reusable components
│   ├── models/                   # Model implementations
│   ├── serving/                  # Model serving utilities
│   └── [utility modules]         # Various utility modules
├── Makefile                      # Build and release automation
├── project_config.yml            # Main project configuration
├── pyproject.toml                # Python package dependencies
├── databricks.yml                # Databricks workflow configuration
└── bundle_monitoring.yml         # Monitoring job configuration
```

## Workflow

### 1. Development in Notebooks

Start by working with the notebook templates in Databricks:

- `01_data_processing_template.py`: Data processing and feature engineering
- `02_train_and_register_model_template.py`: Model training and registration
- `03_fe_model_online_serving_template.py`: Feature lookup serving setup
- `04_create_monitoring_table.py`: Monitoring configuration
- `05_create_alerts.py`: Alert setup

**Note:** Some functionality (feature store, DBUtils) only works in Databricks, not in local IDEs.

### 2. Package and Deploy

Use the Makefile to build and deploy your code to Databricks:

```bash
# Full release process
make release

# Or individual steps
make clean
make update-version
make build
make update-whl-dbx
```

After deployment, install in Databricks:

```python
%pip install /Volumes/uc_dev/hotel_reservation/samples/packages/hotel_reservation-latest-py3-none-any.whl
dbutils.library.restartPython()
```

### 3. Automation Scripts

Run the automation scripts to execute the pipeline:

```bash
# Data processing
python scripts/01_data_processing.py --root_path /path/to/project --env dev

# Model training
python scripts/02_train_and_register_fe_model.py --root_path /path/to/project \
  --env dev --job_run_id 12345 --branch main --git-sha abc123

# Deploy model endpoint
python scripts/03_deploy_fe_model_serving_endpoint.py --root_path /path/to/project --env dev

# Refresh monitoring
python scripts/04_refresh_monitoring.py --root_path /path/to/project --env dev
```

### 4. Job Scheduling

Schedule jobs with Databricks Workspace bundles:

```bash
# Deploy to default environment
databricks bundle deploy

# Deploy to specific environment
databricks bundle deploy --target <environment>
```

## Configuration

Configure your project in `files/project_config.yml`:

```yaml
common:
  catalog_name: "uc_dev"
  schema_name: "hotel_reservation"
  model_name: "hotel_reservation_model_fe"
  # Additional configuration...

dev:
  experiment_name: "hotel_reservation_dev"

prod:
  experiment_name: "hotel_reservation_prod"
```

## Troubleshooting

- **Configuration issues**: Verify `project_config.yml` exists and contains all required fields
- **Feature table errors**: Create the feature table if missing
- **Deployment failures**: Check model registration and endpoint configuration
- **Databricks-specific features**: Test DBUtils and feature store operations directly in notebooks, not local IDE

## Best Practices

- Develop in notebooks, productionize in scripts
- Keep configuration in version control
- Test thoroughly before deployment
- Monitor model performance in production
- Add CI with commit checks & CD to deploy bundle in different environments
