import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple, Callable

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

import mlflow
from loguru import logger


def build_preprocessor(
    numeric_features: List[str] = None,
    categorical_features: List[str] = None,
    numeric_strategy: str = "standard",  # "standard", "minmax", "robust", "none"
    categorical_strategy: str = "onehot",  # "onehot", "ordinal", "binary", "none"
    missing_strategy: str = "mean",  # "mean", "median", "most_frequent", "constant"
    custom_transformers: List[Tuple[str, Any, List[str]]] = None,
    **kwargs
) -> ColumnTransformer:
    """
    Build a preprocessing pipeline with common transformations.
    
    Parameters
    ----------
    numeric_features : List[str]
        List of numeric feature column names
    categorical_features : List[str]
        List of categorical feature column names
    numeric_strategy : str
        Strategy for scaling numeric features:
        - "standard": StandardScaler (z-score normalization)
        - "minmax": MinMaxScaler (scale to 0-1 range)
        - "robust": RobustScaler (scale using median and IQR)
        - "none": No scaling, pass through values
    categorical_strategy : str
        Strategy for encoding categorical features:
        - "onehot": One-hot encoding
        - "ordinal": Ordinal encoding (convert to integers)
        - "binary": Binary encoding for binary features
        - "none": No encoding, pass through values
    missing_strategy : str
        Strategy for handling missing values:
        - "mean": Replace with feature mean
        - "median": Replace with feature median
        - "most_frequent": Replace with most frequent value
        - "constant": Replace with constant (specify with missing_fill_value)
        - "none": Don't handle missing values
    custom_transformers : List[Tuple[str, Any, List[str]]]
        List of custom transformers in the format (name, transformer, columns)
    **kwargs : 
        Additional options:
        - missing_fill_value: Value to use with "constant" strategy
        - remainder: How to handle columns not specified ("drop" or "passthrough")
        
    Returns
    -------
    ColumnTransformer
        Configured preprocessing pipeline
    """
    transformers = []
    numeric_features = numeric_features or []
    categorical_features = categorical_features or []
    remainder = kwargs.get('remainder', 'drop')
    
    # Process numeric features
    if numeric_features:
        if numeric_strategy == "standard":
            num_transformer = StandardScaler()
        elif numeric_strategy == "minmax":
            num_transformer = MinMaxScaler()
        elif numeric_strategy == "robust":
            from sklearn.preprocessing import RobustScaler
            num_transformer = RobustScaler()
        elif numeric_strategy == "none":
            num_transformer = "passthrough"
        else:
            raise ValueError(f"Unknown numeric strategy: {numeric_strategy}")
            
        # Handle missing values for numeric features
        if missing_strategy != "none" and numeric_strategy != "none":
            imputer_kwargs = {}
            if missing_strategy == "constant":
                imputer_kwargs["fill_value"] = kwargs.get("missing_fill_value", 0)
                
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy=missing_strategy, **imputer_kwargs)),
                ("scaler", num_transformer)
            ])
            transformers.append(("num", num_pipeline, numeric_features))
        else:
            transformers.append(("num", num_transformer, numeric_features))
    
    # Process categorical features
    if categorical_features:
        if categorical_strategy == "onehot":
            cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        elif categorical_strategy == "ordinal":
            cat_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        elif categorical_strategy == "binary":
            cat_transformer = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
        elif categorical_strategy == "none":
            cat_transformer = "passthrough"
        else:
            raise ValueError(f"Unknown categorical strategy: {categorical_strategy}")
            
        # Handle missing values for categorical features
        if missing_strategy != "none" and categorical_strategy != "none":
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", cat_transformer)
            ])
            transformers.append(("cat", cat_pipeline, categorical_features))
        else:
            transformers.append(("cat", cat_transformer, categorical_features))
    
    # Add custom transformers
    if custom_transformers:
        for name, transformer, columns in custom_transformers:
            transformers.append((name, transformer, columns))
    
    # Build column transformer
    if not transformers:
        logger.warning("No transformers configured. Returning 'passthrough'")
        return "passthrough"
        
    return ColumnTransformer(transformers=transformers, remainder=remainder)


def create_pipeline(
    preprocessor: Any,
    model_type: str,
    task: str = "classification",
    model_params: Dict[str, Any] = None,
    resampling_strategy: Optional[str] = None,  # Add resampling option
    resampling_params: Dict[str, Any] = None,  # Add resampling parameters
    **kwargs
) -> Pipeline:
    """
    Create a complete ML pipeline with preprocessing and model.
    
    Parameters
    ----------
    preprocessor : Any
        Preprocessing transformer (e.g., from build_preprocessor)
    model_type : str
        Type of model to create, options include:
        - Classification: "random_forest", "logistic_regression", "gradient_boosting", 
                          "decision_tree", "svm", "lightgbm", "xgboost", "catboost"
        - Regression: "random_forest", "linear_regression", "gradient_boosting",  
                      "decision_tree", "svr", "lightgbm", "xgboost", "catboost"
    task : str
        "classification" or "regression"
    model_params : Dict[str, Any]
        Parameters for the model
    resampling_strategy : str, optional
        Resampling strategy to use:
        - "smote": Synthetic Minority Over-sampling Technique
        - "random_over": Random over-sampling
        - "random_under": Random under-sampling
        - "adasyn": Adaptive Synthetic sampling
        - None: No resampling (default)
    resampling_params : Dict[str, Any], optional
        Parameters for the resampling technique
    **kwargs :
        Additional options
        
    Returns
    -------
    Pipeline
        Complete scikit-learn pipeline ready for training
    """
    model_params = model_params or {}
    
    # Create model based on type and task
    if task == "classification":
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**model_params)
        elif model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**model_params)
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(**model_params)
        elif model_type == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(**model_params)
        elif model_type == "svm":
            from sklearn.svm import SVC
            model = SVC(**model_params)
        elif model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(**model_params)
            except ImportError:
                raise ImportError("LightGBM is not installed. Install with 'pip install lightgbm'")
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(**model_params)
            except ImportError:
                raise ImportError("XGBoost is not installed. Install with 'pip install xgboost'")
        elif model_type == "catboost":
            try:
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(**model_params)
            except ImportError:
                raise ImportError("CatBoost is not installed. Install with 'pip install catboost'")
        else:
            raise ValueError(f"Unknown classification model type: {model_type}")
    
    elif task == "regression":
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**model_params)
        elif model_type == "linear_regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression(**model_params)
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(**model_params)
        elif model_type == "decision_tree":
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor(**model_params)
        elif model_type == "svr":
            from sklearn.svm import SVR
            model = SVR(**model_params)
        elif model_type == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(**model_params)
            except ImportError:
                raise ImportError("LightGBM is not installed. Install with 'pip install lightgbm'")
        elif model_type == "xgboost":
            try:
                from xgboost import XGBRegressor
                model = XGBRegressor(**model_params)
            except ImportError:
                raise ImportError("XGBoost is not installed. Install with 'pip install xgboost'")
        elif model_type == "catboost":
            try:
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(**model_params)
            except ImportError:
                raise ImportError("CatBoost is not installed. Install with 'pip install catboost'")
        else:
            raise ValueError(f"Unknown regression model type: {model_type}")
    
    else:
        raise ValueError(f"Unknown task type: {task}. Must be 'classification' or 'regression'")
    
    # Create pipeline steps
    steps = [("preprocessor", preprocessor)]

    # Add resampling step if requested
    if resampling_strategy and task == "classification":
        try:
            from imblearn.pipeline import Pipeline as ImbPipeline
            
            # Select the appropriate resampling technique
            if resampling_strategy == "smote":
                from imblearn.over_sampling import SMOTE
                resampler = SMOTE(**resampling_params)
            elif resampling_strategy == "random_over":
                from imblearn.over_sampling import RandomOverSampler
                resampler = RandomOverSampler(**resampling_params)
            elif resampling_strategy == "random_under":
                from imblearn.under_sampling import RandomUnderSampler
                resampler = RandomUnderSampler(**resampling_params)
            elif resampling_strategy == "adasyn":
                from imblearn.over_sampling import ADASYN
                resampler = ADASYN(**resampling_params)
            else:
                raise ValueError(f"Unknown resampling strategy: {resampling_strategy}")
                
            # Add resampling step
            steps.append(("resampler", resampler))
            
            # Add model step
            steps.append(("model", model))
            
            # Return imbalanced-learn pipeline 
            return ImbPipeline(steps=steps)
    
        except ImportError:
                logger.warning("imbalanced-learn not installed. Skipping resampling step.")
                logger.warning("Install with: pip install imbalanced-learn")
                steps.append(("model", model))
                return Pipeline(steps=steps)
    
    # Add model step (for no resampling or regression tasks)
    steps.append(("model", model))
    
    # Return standard sklearn pipeline
    return Pipeline(steps=steps)


def train_model(
    pipeline: Pipeline,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    cv: Optional[int] = None,
    scoring: Optional[str] = None,
    eval_metric: Optional[str] = None,
    track_with_mlflow: bool = False,
    experiment_name: Optional[str] = None,
    log_params: bool = True,
    **kwargs
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a model pipeline with options for cross-validation and early stopping.
    
    Parameters
    ----------
    pipeline : Pipeline
        Scikit-learn pipeline to train
    X_train : DataFrame or array
        Training features
    y_train : Series or array
        Training target
    X_val : DataFrame or array, optional
        Validation features (used for early stopping)
    y_val : Series or array, optional
        Validation target (used for early stopping)
    cv : int, optional
        Number of cross-validation folds. If None, no CV is performed.
    scoring : str, optional
        Scoring metric for cross-validation
    eval_metric : str, optional
        Metric to use for early stopping
    track_with_mlflow : bool
        Whether to log training results with MLflow
    experiment_name : str, optional
        MLflow experiment name
    log_params : bool
        Whether to log model parameters to MLflow
    **kwargs :
        Additional parameters to pass to fit method
        
    Returns
    -------
    Tuple[Pipeline, Dict[str, float]]
        Trained pipeline and dictionary of performance metrics
    """
    # Prepare result metrics
    metrics = {}
    
    # Setup MLflow if requested
    if track_with_mlflow:
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            
        mlflow.start_run()
        
        # Log pipeline components
        if log_params:
            # Log model parameters
            for step_name, step in pipeline.steps:
                if step_name == "model" and hasattr(step, "get_params"):
                    params = step.get_params()
                    mlflow.log_params(params)
    
    try:
        # Cross-validation if requested
        if cv is not None and cv > 1:
            logger.info(f"Performing {cv}-fold cross-validation")
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
            avg_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            
            logger.info(f"CV {scoring or 'score'}: {avg_cv_score:.4f} (±{std_cv_score:.4f})")
            metrics["cv_score"] = avg_cv_score
            metrics["cv_score_std"] = std_cv_score
            
            if track_with_mlflow:
                mlflow.log_metric(f"cv_{scoring or 'score'}", avg_cv_score)
                mlflow.log_metric(f"cv_{scoring or 'score'}_std", std_cv_score)
        
        # Prepare fit parameters
        fit_params = kwargs.copy()
        
        # Train the model
        logger.info("Training model...")
        pipeline.fit(X_train, y_train, **fit_params)
        logger.info("Model training complete")
        
        # Calculate post-training metrics
        if hasattr(pipeline, "score"):
            train_score = pipeline.score(X_train, y_train)
            metrics["train_score"] = train_score
            logger.info(f"Training score: {train_score:.4f}")
            
            if track_with_mlflow:
                mlflow.log_metric("train_score", train_score)
            
            if X_val is not None and y_val is not None:
                val_score = pipeline.score(X_val, y_val)
                metrics["val_score"] = val_score
                logger.info(f"Validation score: {val_score:.4f}")
                
                if track_with_mlflow:
                    mlflow.log_metric("val_score", val_score)
        
        # Additional metrics for classification
        task = "classification" if hasattr(pipeline, "predict_proba") else "regression"
        
        if task == "classification" and X_val is not None and y_val is not None:
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                y_pred = pipeline.predict(X_val)
                
                metrics["accuracy"] = accuracy_score(y_val, y_pred)
                metrics["precision"] = precision_score(y_val, y_pred)
                metrics["recall"] = recall_score(y_val, y_pred)
                metrics["f1_score"] = f1_score(y_val, y_pred)
                
                # Log classification metrics
                logger.info(f"Validation metrics: Accuracy={metrics['accuracy']:.4f}, "
                          f"Precision={metrics['precision']:.4f}, "
                          f"Recall={metrics['recall']:.4f}, "
                          f"F1={metrics['f1']:.4f}")
                
                # Try to get ROC AUC if applicable (binary classification)
                if len(np.unique(y_val)) == 2 and hasattr(pipeline, "predict_proba"):
                    y_proba = pipeline.predict_proba(X_val)[:, 1]
                    metrics["roc_auc"] = roc_auc_score(y_val, y_proba)
                    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
                
                if track_with_mlflow:
                    for metric_name, metric_value in metrics.items():
                        if metric_name not in ["train_score", "val_score", "cv_score", "cv_score_std"]:
                            mlflow.log_metric(metric_name, metric_value)
            except Exception as e:
                logger.warning(f"Error calculating classification metrics: {str(e)}")
        
        # Additional metrics for regression
        elif task == "regression" and X_val is not None and y_val is not None:
            try:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                y_pred = pipeline.predict(X_val)
                
                metrics["mse"] = mean_squared_error(y_val, y_pred)
                metrics["rmse"] = np.sqrt(metrics["mse"])
                metrics["mae"] = mean_absolute_error(y_val, y_pred)
                metrics["r2"] = r2_score(y_val, y_pred)
                
                # Log regression metrics
                logger.info(f"Validation metrics: MSE={metrics['mse']:.4f}, "
                          f"RMSE={metrics['rmse']:.4f}, "
                          f"MAE={metrics['mae']:.4f}, "
                          f"R²={metrics['r2']:.4f}")
                
                if track_with_mlflow:
                    for metric_name, metric_value in metrics.items():
                        if metric_name not in ["train_score", "val_score", "cv_score", "cv_score_std"]:
                            mlflow.log_metric(metric_name, metric_value)
            except Exception as e:
                logger.warning(f"Error calculating regression metrics: {str(e)}")
    
    finally:
        # End MLflow run if started
        if track_with_mlflow:
            mlflow.end_run()
    
    return pipeline, metrics



def register_mlflow_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
    alias: str = "latest-model",
    tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Register a trained model in the MLflow Model Registry.
    
    Parameters
    ----------
    run_id : str
        MLflow run ID where the model is logged
    model_name : str
        Name to register the model under (can include catalog.schema.table format)
    artifact_path : str, optional
        Path within the run where the model is stored
    alias : str, optional
        Alias to assign to the new model version
    tags : dict, optional
        Tags to attach to the registered model
    
    Returns
    -------
    dict
        Information about the registered model
    """
    logger.info(f"Registering model {model_name} from run {run_id}...")
    
    # Register the model
    registered_model = mlflow.register_model(
        model_uri=f"runs:/{run_id}/{artifact_path}",
        name=model_name,
        tags=tags
    )
    
    # Get the version
    latest_version = registered_model.version
    
    # Set alias
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=latest_version
    )
    
    logger.info(f"Model registered successfully as {model_name} version {latest_version} with alias {alias}")
    
    # Return information about the registered model
    return {
        "name": model_name,
        "version": latest_version,
        "alias": alias,
        "run_id": run_id
    }


def batch_score_model(
    spark,
    model_uri: str,
    data: Union[pd.DataFrame, "SparkDataFrame"],
    feature_engineering: bool = True,
    **kwargs
) -> Union[pd.DataFrame, "SparkDataFrame"]:
    """
    Score a batch of data using a registered model, with support for Feature Store.
    
    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    model_uri : str
        URI of the model to use for scoring (e.g., "models:/my_model@latest-model")
    data : DataFrame
        Data to score
    feature_engineering : bool
        Whether to use Feature Engineering Client for scoring
    **kwargs : 
        Additional parameters for scoring
    
    Returns
    -------
    DataFrame
        Predictions DataFrame
    """
    logger.info(f"Scoring batch data with model {model_uri}...")
    
    # Handle pandas DataFrame input
    is_pandas = isinstance(data, pd.DataFrame)
    if is_pandas:
        spark_df = spark.createDataFrame(data)
    else:
        spark_df = data
    
    # Score using Feature Engineering Client if requested
    if feature_engineering:
        try:
            # Import Feature Engineering Client
            from databricks import feature_engineering
            fe_client = feature_engineering.FeatureEngineeringClient()
            
            # Score batch
            predictions = fe_client.score_batch(
                model_uri=model_uri,
                df=spark_df,
                **kwargs
            )
            
        except ImportError:
            logger.warning("Feature Engineering not available. Falling back to MLflow.")
            predictions = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)(spark_df)
            
    else:
        # Use standard MLflow scoring
        predictions = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)(spark_df)
    
    # Convert back to pandas if input was pandas
    if is_pandas:
        predictions = predictions.toPandas()
    
    logger.info(f"Batch scoring completed with {predictions.shape[0] if is_pandas else predictions.count()} predictions")
    return predictions


def compare_models(
    spark,
    champion_model_uri: str,
    challenger_model_uri: str,
    test_data: Union[pd.DataFrame, "SparkDataFrame"],
    target_column: str,
    metric: str = "roc_auc",
    improvement_threshold: float = 0.0,
    feature_engineering: bool = True
) -> Dict[str, Any]:
    """
    Compare performance between champion and challenger models.
    
    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    champion_model_uri : str
        URI of the champion model
    challenger_model_uri : str
        URI of the challenger model
    test_data : DataFrame
        Test data with features and target
    target_column : str
        Name of the target column
    metric : str
        Metric to use for comparison ('roc_auc', 'accuracy', 'f1', etc.)
    improvement_threshold : float
        Minimum improvement required to select challenger
    feature_engineering : bool
        Whether to use Feature Engineering for scoring
        
    Returns
    -------
    dict
        Comparison results including metrics and improvement status
    """
    logger.info(f"Comparing models: Champion={champion_model_uri}, Challenger={challenger_model_uri}")
    
    # Get predictions from both models
    champion_preds = batch_score_model(
        spark=spark,
        model_uri=champion_model_uri,
        data=test_data,
        feature_engineering=feature_engineering
    )
    
    challenger_preds = batch_score_model(
        spark=spark,
        model_uri=challenger_model_uri,
        data=test_data,
        feature_engineering=feature_engineering
    )
    
    # Get target values
    if isinstance(test_data, pd.DataFrame):
        y_true = test_data[target_column].values
    else:
        y_true = test_data.select(target_column).toPandas()[target_column].values
    
    # Extract prediction column
    if isinstance(champion_preds, pd.DataFrame):
        if 'probability' in champion_preds.columns:
            champion_pred_values = champion_preds['probability'].values
            challenger_pred_values = challenger_preds['probability'].values
        else:
            # Find prediction column
            pred_col = [col for col in champion_preds.columns if 'prediction' in col.lower()][0]
            champion_pred_values = champion_preds[pred_col].values
            challenger_pred_values = challenger_preds[pred_col].values
    else:
        # For Spark DataFrames
        if 'probability' in champion_preds.columns:
            champion_pred_values = champion_preds.select('probability').toPandas()['probability'].values
            challenger_pred_values = challenger_preds.select('probability').toPandas()['probability'].values
        else:
            # Find prediction column
            pred_cols = [col for col in champion_preds.columns if 'prediction' in col.lower()]
            pred_col = pred_cols[0] if pred_cols else champion_preds.columns[0]  # Fallback to first column
            champion_pred_values = champion_preds.select(pred_col).toPandas()[pred_col].values
            challenger_pred_values = challenger_preds.select(pred_col).toPandas()[pred_col].values
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
    
    # Handle binary classification case for roc_auc
    if metric == 'roc_auc':
        champion_score = roc_auc_score(y_true, champion_pred_values)
        challenger_score = roc_auc_score(y_true, challenger_pred_values)
    elif metric in ['accuracy', 'f1', 'precision', 'recall']:
        # Convert probabilities to class predictions if needed
        if champion_pred_values.min() >= 0 and champion_pred_values.max() <= 1:
            champion_class_preds = (champion_pred_values > 0.5).astype(int)
            challenger_class_preds = (challenger_pred_values > 0.5).astype(int)
        else:
            champion_class_preds = champion_pred_values
            challenger_class_preds = challenger_pred_values
        
        # Calculate specific metric
        if metric == 'accuracy':
            champion_score = accuracy_score(y_true, champion_class_preds)
            challenger_score = accuracy_score(y_true, challenger_class_preds)
        elif metric == 'f1':
            champion_score = f1_score(y_true, champion_class_preds, average='weighted')
            challenger_score = f1_score(y_true, challenger_class_preds, average='weighted')
        elif metric == 'precision':
            champion_score = precision_score(y_true, champion_class_preds, average='weighted')
            challenger_score = precision_score(y_true, challenger_class_preds, average='weighted')
        elif metric == 'recall':
            champion_score = recall_score(y_true, champion_class_preds, average='weighted')
            challenger_score = recall_score(y_true, challenger_class_preds, average='weighted')
    else:
        raise ValueError(f"Unsupported metric: {metric}. Supported metrics: roc_auc, accuracy, f1, precision, recall")
    
    # Determine if challenger is better
    improvement = challenger_score - champion_score
    is_better = improvement > improvement_threshold
    
    # Create result dictionary
    results = {
        "champion_model": champion_model_uri,
        "challenger_model": challenger_model_uri,
        "metric": metric,
        "champion_score": champion_score,
        "challenger_score": challenger_score,
        "improvement": improvement,
        "improvement_threshold": improvement_threshold,
        "challenger_is_better": is_better
    }
    
    logger.info(f"Model comparison completed: Champion {metric}={champion_score:.4f}, "
              f"Challenger {metric}={challenger_score:.4f}, Improvement={improvement:.4f}")
    
    if is_better:
        logger.info("Challenger model performed better than champion")
    else:
        logger.info("Champion model retained performance edge")
    
    return results


def evaluate_model_improvement(
    spark,
    current_model,
    latest_model_uri: str = None,
    test_data: Union[pd.DataFrame, "SparkDataFrame"] = None,
    features: List[str] = None,
    target_column: str = None,
    metric: str = "roc_auc",
    threshold: float = 0.0,
    feature_engineering_client = None
) -> bool:
    """
    Evaluate if a current model performs better than the latest registered model.
    
    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    current_model : 
        The current model to evaluate
    latest_model_uri : str, optional
        URI of the latest registered model (e.g., "models:/name@latest")
    test_data : DataFrame
        Test data containing features and target
    features : List[str]
        Feature columns to use for prediction
    target_column : str
        Target column name
    metric : str, default="roc_auc"
        Metric to use for comparison
    threshold : float, default=0.0
        Improvement threshold required to consider the model better
    feature_engineering_client :
        Feature Engineering Client instance (optional)
        
    Returns
    -------
    bool
        True if current model performs better, False otherwise
    """
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score, 
        precision_score, recall_score, mean_squared_error
    )
    
    try:
        logger.info(f"Evaluating model improvement using {metric} metric...")
        
        # Get predictions from the current model
        if hasattr(current_model, 'predict_proba') and metric in ['roc_auc']:
            preds_current = current_model.predict_proba(test_data[features])[:, 1]
        else:
            preds_current = current_model.predict(test_data[features])
        
        # Try to get predictions from the latest model
        try:
            if feature_engineering_client:
                # Use Feature Engineering Client for prediction
                fe = feature_engineering_client
                preds_latest = fe.score_batch(model_uri=latest_model_uri, df=test_data[features])
            else:
                # Use MLflow for prediction
                latest_model = mlflow.pyfunc.load_model(latest_model_uri)
                
                # Handle different prediction formats
                latest_preds = latest_model.predict(test_data[features])
                
                # Extract predictions from result
                if isinstance(latest_preds, dict) and 'probability' in latest_preds:
                    preds_latest = latest_preds['probability']
                elif isinstance(latest_preds, pd.DataFrame) and 'probability' in latest_preds.columns:
                    preds_latest = latest_preds['probability'].values
                else:
                    preds_latest = latest_preds
            
            # Calculate metrics based on the specified metric
            if metric == 'roc_auc':
                latest_score = roc_auc_score(test_data[target_column], preds_latest)
                current_score = roc_auc_score(test_data[target_column], preds_current)
            elif metric == 'accuracy':
                # Convert probabilities to class predictions if needed
                if hasattr(current_model, 'predict_proba'):
                    preds_current = (preds_current > 0.5).astype(int)
                if isinstance(preds_latest[0], float) and 0 <= preds_latest[0] <= 1:
                    preds_latest = (preds_latest > 0.5).astype(int)
                    
                latest_score = accuracy_score(test_data[target_column], preds_latest)
                current_score = accuracy_score(test_data[target_column], preds_current)
            elif metric == 'f1':
                # Convert probabilities to class predictions if needed
                if hasattr(current_model, 'predict_proba'):
                    preds_current = (preds_current > 0.5).astype(int)
                if isinstance(preds_latest[0], float) and 0 <= preds_latest[0] <= 1:
                    preds_latest = (preds_latest > 0.5).astype(int)
                    
                latest_score = f1_score(test_data[target_column], preds_latest)
                current_score = f1_score(test_data[target_column], preds_current)
            elif metric == 'rmse':
                latest_score = -1 * np.sqrt(mean_squared_error(test_data[target_column], preds_latest))
                current_score = -1 * np.sqrt(mean_squared_error(test_data[target_column], preds_current))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            # Compare performance
            model_improved = current_score > (latest_score + threshold)
            
            # Log results
            logger.info(f"Model comparison: Current model {metric}={current_score:.4f}, "
                      f"Latest model {metric}={latest_score:.4f}")
            
            if model_improved:
                logger.info(f"Current model performs better (improvement: {current_score - latest_score:.4f}). "
                          f"Register the challenger.")
            else:
                logger.info(f"Current model does not show sufficient improvement. "
                          f"Keep the champion model.")
            
            return model_improved
            
        except Exception as e:
            logger.warning(f"Error comparing with latest model: {str(e)}")
            logger.info("No valid latest model found for comparison. Register the current model.")
            return True
            
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        logger.info("Error during model evaluation. Defaulting to registering the current model.")
        return True