from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import mlflow
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from components.models import model_metrics

class Preprocessor:
    """A simplified interface for building scikit-learn preprocessing pipelines."""
    
    @staticmethod
    def create(
        numeric_features: List[str] = None,
        categorical_features: List[str] = None,
        numeric_strategy: str = "standard",  # "standard", "minmax", "robust", "none"
        categorical_strategy: str = "onehot",  # "onehot", "ordinal", "binary", "none"
        missing_strategy: str = "mean",  # "mean", "median", "most_frequent", "constant"
        missing_fill_value: Any = None,
        passthrough_features: List[str] = None,
        drop_features: List[str] = None,
    ) -> ColumnTransformer:
        """
        Create a preprocessing pipeline with common transformations.
        
        Parameters
        ----------
        numeric_features : List[str], optional
            List of numeric feature columns
        categorical_features : List[str], optional
            List of categorical feature columns
        numeric_strategy : str, default="standard"
            Strategy for scaling numeric features: "standard", "minmax", "robust", "none"
        categorical_strategy : str, default="onehot"
            Strategy for encoding categorical features: "onehot", "ordinal", "binary", "none"
        missing_strategy : str, default="mean"
            Strategy for handling missing values: "mean", "median", "most_frequent", "constant", "none"
        missing_fill_value : Any, optional
            Value to use with "constant" strategy
        passthrough_features : List[str], optional
            Features to pass through unchanged
        drop_features : List[str], optional
            Features to drop
            
        Returns
        -------
        ColumnTransformer
            Configured preprocessing pipeline
        """
        transformers = []
        numeric_features = numeric_features or []
        categorical_features = categorical_features or []
        passthrough_features = passthrough_features or []
        
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
                if missing_strategy == "constant" and missing_fill_value is not None:
                    imputer_kwargs["fill_value"] = missing_fill_value

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
        
        # Add passthrough features
        if passthrough_features:
            transformers.append(("pass", "passthrough", passthrough_features))
        
        # Determine remainder handling
        remainder = "drop" if drop_features else "passthrough"
        
        # Build column transformer
        if not transformers:
            return "passthrough"
            
        return ColumnTransformer(transformers=transformers, remainder=remainder)
    
class ModelFactory:
    """Factory class for creating common ML models with sensible defaults."""

    _classification_models = {
        'random_forest': ('sklearn.ensemble', 'RandomForestClassifier'),
        'logistic': ('sklearn.linear_model', 'LogisticRegression'),
        'gradient_boosting': ('sklearn.ensemble', 'GradientBoostingClassifier'),
        'decision_tree': ('sklearn.tree', 'DecisionTreeClassifier'),
        'svm': ('sklearn.svm', 'SVC'),
        'knn': ('sklearn.neighbors', 'KNeighborsClassifier'),
        'naive_bayes': ('sklearn.naive_bayes', 'GaussianNB'),
        'lightgbm': ('lightgbm', 'LGBMClassifier'),
        'xgboost': ('xgboost', 'XGBClassifier'),
        'catboost': ('catboost', 'CatBoostClassifier'),
    }


    @staticmethod
    def create(model_name: str, task: str = "classification", **params) -> BaseEstimator:
        """
        Create a new model instance.
        
        Parameters
        ----------
        model_name : str
            Name of the model to create
        task : str, default="classification"
            Task type: "classification", "regression", or "clustering"
        **params :
            Parameters to pass to the model constructor
            
        Returns
        -------
        BaseEstimator
            The initialized model instance
        """
        if task == "classification":
            model_dict = ModelFactory._classification_models
        else:
            raise ValueError(f"Unknown task type: {task}")
            
        if model_name not in model_dict:
            raise ValueError(f"Unknown model: {model_name} for task {task}")
            
        module_name, class_name = model _dict[model_name]
        
        try:
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name)
            return model_class(**params)
        except ImportError:
            raise ImportError(f"Could not import {model_name}. Please make sure it's installed.")


class MLPipeline:
    """
    Create and manage machine learning pipelines with preprocessing and models.
    """
    
    @staticmethod
    def create(
        preprocessor: Any,
        model: BaseEstimator,
        resampling: str = None,
        resampling_params: Dict[str, Any] = None,
    ) -> Pipeline:
        """
        Create a complete ML pipeline with preprocessing and model.
        
        Parameters
        ----------
        preprocessor : Any
            Preprocessing transformer
        model : BaseEstimator
            ML model instance
        resampling : str, optional
            Resampling strategy: "smote", "random_over", "random_under", "adasyn"
        resampling_params : Dict[str, Any], optional
            Parameters for the resampling technique
            
        Returns
        -------
        Pipeline
            Complete scikit-learn pipeline
        """
        resampling_params = resampling_params or {}
        steps = [("preprocessor", preprocessor)]
        
        # Add resampling step if requested
        if resampling
            if resampling == "smote":
                resampler = SMOTE(**resampling_params)
            elif resampling == "random_over":
                resampler = RandomOverSampler(**resampling_params)
            elif resampling == "random_under":
                resampler = RandomUnderSampler(**resampling_params)
            else:
                raise ValueError(f"Unknown resampling strategy: {resampling}")
                
            steps.append(("resampler", resampler))
            
            # Add model step
            steps.append(("model", model))
            
            # Return imbalanced-learn pipeline
            return ImbPipeline(steps=steps)
        
        # Add model step (for no resampling)
        steps.append(("model", model))
        
        # Return standard sklearn pipeline
        return Pipeline(steps=steps)
    

    @staticmethod
    def create_pipeline(
        model_name: str,
        task: str = "classification",
        numeric_features: List[str] = None,
        categorical_features: List[str] = None, 
        numeric_strategy: str = "standard",
        categorical_strategy: str = "onehot",
        missing_strategy: str = "mean",
        resampling: str = None,
        model_params: Dict[str, Any] = None,
    ) -> Pipeline:
        """
        Create a complete pipeline with one call using sensible defaults.
        
        Parameters
        ----------
        model_name : str
            Name of the model to create
        task : str, default="classification" 
            "classification", "regression", or "clustering"
        numeric_features : List[str], optional
            List of numeric features
        categorical_features : List[str], optional 
            List of categorical features
        numeric_strategy : str, default="standard"
            Strategy for scaling numeric features
        categorical_strategy : str, default="onehot"
            Strategy for encoding categorical features
        missing_strategy : str, default="mean"
            Strategy for handling missing values
        resampling : str, optional
            Resampling strategy for imbalanced data
        model_params : Dict[str, Any], optional
            Parameters for the model
            
        Returns
        -------
        Pipeline
            Complete scikit-learn pipeline
        """
        model_params = model_params or {}
        
        # Create preprocessor
        preprocessor = Preprocessor.create(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            missing_strategy=missing_strategy
        )
        
        # Create model
        model = ModelFactory.create(model_name, task=task, **model_params)
        
        # Create and return pipeline
        return MLPipeline.create(preprocessor, model, resampling)
    