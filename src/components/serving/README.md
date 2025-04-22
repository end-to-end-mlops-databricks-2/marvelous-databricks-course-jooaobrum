# Feature Lookup Serving

A specialized module for deploying ML models with feature store integration in Databricks.

## Overview

The `FeatureLookupServing` class provides a streamlined interface for deploying machine learning models with online feature lookup capabilities. This component enables real-time inference by automatically synchronizing feature data from offline to online tables and managing model serving endpoints.

## Key Features

- **Online Table Management**: Create and update online feature tables for low-latency inference
- **Model Endpoint Deployment**: Deploy models to serving endpoints with configurable settings
- **Auto-Capture Configuration**: Set up inference logging for monitoring
- **Pipeline Updates**: Trigger and monitor pipeline updates for online tables
- **Status Monitoring**: Track endpoint health and performance metrics

## Usage Example

```python
# Initialize the serving manager
serving_manager = FeatureLookupServing(
    model_name="hotel_reservation_model_fe",
    endpoint_name="hotel-reservation-endpoint",
    catalog_name="uc_dev",
    schema_name="hotel_reservation",
    feature_table_name="hotel_reservation_features",
    primary_keys=["Booking_ID"],
    alias="latest-model"
)

# Create online feature table (one-time setup)
serving_manager.create_online_table()

# Deploy or update the model serving endpoint
serving_manager.deploy_or_update_serving_endpoint(
    workload_size="Small",
    scale_to_zero=True,
    enable_inference_tables=True
)

# Check endpoint status
status = serving_manager.get_endpoint_status()
print(f"Endpoint status: {status['state']}")
```

## Deployment Options

- **Workload Size**: Configure compute resources (Small, Medium, Large)
- **Scale-to-Zero**: Automatically scale down inactive endpoints to reduce costs
- **Version Control**: Deploy specific model versions or use aliases (e.g., "latest-model")
- **Inference Tables**: Enable automatic logging of inference requests and responses

## Monitoring

The serving manager provides monitoring capabilities to track endpoint performance:

- **Request Count**: Number of inference requests
- **Response Time**: Average and p99 latency metrics
- **Endpoint State**: Current operational status
- **Update History**: Creation and modification timestamps

## Best Practices

- Create the online table before deploying the model endpoint
- Enable scale-to-zero for cost efficiency in development environments
- Use inference tables to monitor data drift and performance
- Check endpoint status periodically to ensure availability
- Update online tables when feature data changes
