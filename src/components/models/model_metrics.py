from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score


class ModelEvaluation:
    """Class for evaluating machine learning model performance."""

    @staticmethod
    def classification_metrics(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_prob: Optional[Union[pd.Series, np.ndarray]] = None,
        average: str = "weighted",
    ) -> Dict[str, float]:
        """
        Calculate common classification metrics.

        Parameters
        ----------
        y_true : Series or array
            True target values
        y_pred : Series or array
            Predicted target values
        y_prob : Series or array, optional
            Predicted probabilities (for binary classification)
        average : str, default='weighted'
            Averaging strategy for multiclass metrics

        Returns
        -------
        Dict[str, float]
            Dictionary of classification metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        }

        # Add ROC AUC for binary classification if probabilities are provided
        unique_classes = np.unique(y_true)
        if y_prob is not None and len(unique_classes) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["average_precision"] = average_precision_score(y_true, y_prob)

        return metrics

    @staticmethod
    def cross_validate(
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        metrics: List[str] = None,
        task: str = "classification",
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a model using cross-validation with multiple metrics.

        Parameters
        ----------
        model : Any
            Model or pipeline to evaluate
        X : DataFrame or array
            Feature data
        y : Series or array
            Target data
        cv : int, default=5
            Number of cross-validation folds
        metrics : List[str], optional
            Metrics to compute
        task : str, default="classification"
            "classification" or "regression"

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of evaluation metrics with mean and std
        """
        if task == "classification":
            metrics = metrics or ["accuracy", "precision", "recall", "f1", "roc_auc"]
        else:  # regression
            metrics = metrics or ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

        results = {}

        for metric in metrics:
            # Skip ROC AUC for non-binary classification
            if metric == "roc_auc" and task == "classification" and len(np.unique(y)) != 2:
                continue

            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                results[metric] = {"mean": scores.mean(), "std": scores.std(), "values": scores.tolist()}

                # Calculate RMSE from MSE for regression
                if metric == "neg_mean_squared_error":
                    rmse_scores = np.sqrt(-scores)
                    results["rmse"] = {
                        "mean": rmse_scores.mean(),
                        "std": rmse_scores.std(),
                        "values": rmse_scores.tolist(),
                    }
            except Exception as e:
                print(f"Warning: Could not calculate {metric}: {e}")

        return results

    @staticmethod
    def compare_models(
        models: List[Tuple[str, Any]],
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        task: str = "classification",
        metrics: List[str] = None,
        plot: bool = True,
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.

        Parameters
        ----------
        models : List[Tuple[str, Any]]
            List of (name, model) tuples to compare
        X : DataFrame or array
            Feature data
        y : Series or array
            Target data
        cv : int, default=5
            Number of cross-validation folds
        task : str, default="classification"
            "classification" or "regression"
        metrics : List[str], optional
            Metrics to use for comparison
        plot : bool, default=True
            Whether to generate comparison plots

        Returns
        -------
        DataFrame
            Comparison of model performances
        """
        all_results = {}

        for name, model in models:
            # Perform cross-validation
            results = ModelEvaluation.cross_validate(model, X, y, cv, metrics, task)
            model_metrics = {}

            # Extract mean values for each metric
            for metric, values in results.items():
                model_metrics[f"{metric}_mean"] = values["mean"]
                model_metrics[f"{metric}_std"] = values["std"]

            all_results[name] = model_metrics

        # Create DataFrame from results
        results_df = pd.DataFrame.from_dict(all_results, orient="index")

        # Generate plots if requested
        if plot:
            ModelEvaluation.plot_model_comparison(results_df, task)

        return results_df

    @staticmethod
    def plot_model_comparison(results_df: pd.DataFrame, task: str = "classification"):
        """
        Plot comparison of models based on evaluation metrics.

        Parameters
        ----------
        results_df : DataFrame
            DataFrame containing model comparison results
        task : str, default="classification"
            "classification" or "regression"
        """
        plt.figure(figsize=(12, 8))

        # Determine which metrics to plot based on task
        if task == "classification":
            metrics_to_plot = ["accuracy_mean", "f1_mean", "precision_mean", "recall_mean"]
            if "roc_auc_mean" in results_df.columns:
                metrics_to_plot.append("roc_auc_mean")
        else:  # regression
            metrics_to_plot = ["r2_mean", "rmse_mean", "neg_mean_absolute_error_mean"]

        # Filter only metrics that exist in the results
        metrics_to_plot = [m for m in metrics_to_plot if m in results_df.columns]

        # Prepare data for plotting
        plot_data = results_df[metrics_to_plot].copy()

        # Plot
        plot_data.plot(
            kind="bar", yerr=[results_df[m.replace("mean", "std")] for m in metrics_to_plot], capsize=5, figsize=(12, 8)
        )

        plt.title(f"Model Comparison for {task.capitalize()}", fontsize=16)
        plt.ylabel("Score", fontsize=14)
        plt.xlabel("Model", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend(title="Metrics")
        plt.tight_layout()

        # Return the plot for display
        return plt.gcf()

    @staticmethod
    def plot_confusion_matrix(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        normalize: bool = True,
        class_names: List[str] = None,
    ):
        """
        Plot confusion matrix for classification results.

        Parameters
        ----------
        y_true : Series or array
            True target values
        y_pred : Series or array
            Predicted target values
        normalize : bool, default=True
            Whether to normalize the confusion matrix
        class_names : List[str], optional
            Names of classes

        Returns
        -------
        matplotlib.figure.Figure
            Confusion matrix plot
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Normalize if requested
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Set up plot
        plt.figure(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        plt.title("Confusion Matrix", fontsize=16)
        plt.ylabel("True Label", fontsize=14)
        plt.xlabel("Predicted Label", fontsize=14)
        plt.tight_layout()

        return plt.gcf()

    @staticmethod
    def plot_roc_curve(
        y_true: Union[pd.Series, np.ndarray], y_prob: Union[pd.Series, np.ndarray], model_name: str = "Model"
    ):
        """
        Plot ROC curve for binary classification.

        Parameters
        ----------
        y_true : Series or array
            True target values
        y_prob : Series or array
            Predicted probabilities
        model_name : str, default='Model'
            Name of the model

        Returns
        -------
        matplotlib.figure.Figure
            ROC curve plot
        """
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)  # Random guessing line

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        return plt.gcf()

    @staticmethod
    def plot_precision_recall_curve(
        y_true: Union[pd.Series, np.ndarray], y_prob: Union[pd.Series, np.ndarray], model_name: str = "Model"
    ):
        """
        Plot precision-recall curve for binary classification.

        Parameters
        ----------
        y_true : Series or array
            True target values
        y_prob : Series or array
            Predicted probabilities
        model_name : str, default='Model'
            Name of the model

        Returns
        -------
        matplotlib.figure.Figure
            Precision-recall curve plot
        """
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f"{model_name} (AP = {avg_precision:.3f})", linewidth=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=14)
        plt.ylabel("Precision", fontsize=14)
        plt.title("Precision-Recall Curve", fontsize=16)
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)

        return plt.gcf()

    @staticmethod
    def optimize_threshold(
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        metric: str = "f1",
        thresholds: np.ndarray = None,
        plot: bool = True,
    ) -> Dict[str, float]:
        """
        Find the optimal probability threshold for binary classification.

        Parameters
        ----------
        y_true : Series or array
            True target values
        y_prob : Series or array
            Predicted probabilities
        metric : str, default='f1'
            Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
        thresholds : array, optional
            Array of thresholds to try (default: 100 values from 0.01 to 0.99)
        plot : bool, default=True
            Whether to plot the threshold optimization results

        Returns
        -------
        Dict[str, float]
            Dictionary with optimal threshold and corresponding metric value
        """
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 100)

        metric_values = []

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            if metric == "f1":
                score = f1_score(y_true, y_pred)
            elif metric == "accuracy":
                score = accuracy_score(y_true, y_pred)
            elif metric == "precision":
                score = precision_score(y_true, y_pred)
            elif metric == "recall":
                score = recall_score(y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            metric_values.append(score)

        # Find optimal threshold
        best_idx = np.argmax(metric_values)
        best_threshold = thresholds[best_idx]
        best_value = metric_values[best_idx]

        # Plot if requested
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, metric_values, linewidth=2)
            plt.axvline(x=best_threshold, color="r", linestyle="--", label=f"Optimal threshold: {best_threshold:.3f}")

            plt.title(f"Threshold Optimization for {metric.capitalize()}", fontsize=16)
            plt.xlabel("Threshold", fontsize=14)
            plt.ylabel(f"{metric.capitalize()} Score", fontsize=14)
            plt.grid(alpha=0.3)
            plt.legend()

        return {"threshold": best_threshold, f"{metric}_score": best_value}

    @staticmethod
    def plot_learning_curve(
        estimator: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        train_sizes: np.ndarray = None,
        scoring: str = None,
    ):
        """
        Plot learning curves to diagnose overfitting/underfitting.

        Parameters
        ----------
        estimator : Any
            Model or pipeline to evaluate
        X : DataFrame or array
            Feature data
        y : Series or array
            Target data
        cv : int, default=5
            Number of cross-validation folds
        train_sizes : array, optional
            Relative or absolute training set sizes
        scoring : str, optional
            Scoring metric

        Returns
        -------
        matplotlib.figure.Figure
            Learning curve plot
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring
        )

        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot
        plt.figure(figsize=(10, 6))

        # Plot mean
        plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
        plt.plot(train_sizes, test_mean, "o-", color="green", label="Cross-validation score")

        # Plot std
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="green")

        plt.title("Learning Curve", fontsize=16)
        plt.xlabel("Training Set Size", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()

        return plt.gcf()

    @staticmethod
    def feature_importance(model: Any, feature_names: List[str], top_n: int = None, plot: bool = True) -> pd.DataFrame:
        """
        Extract and optionally plot feature importance from a model.

        Parameters
        ----------
        model : Any
            Trained model
        feature_names : List[str]
            Names of features
        top_n : int, optional
            Number of top features to show
        plot : bool, default=True
            Whether to plot feature importance

        Returns
        -------
        DataFrame
            Feature importance sorted by importance
        """
        # Extract feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = importances[0]  # Take first class for multiclass
        elif hasattr(model, "steps") and hasattr(model.steps[-1][1], "feature_importances_"):
            importances = model.steps[-1][1].feature_importances_
        elif hasattr(model, "steps") and hasattr(model.steps[-1][1], "coef_"):
            importances = np.abs(model.steps[-1][1].coef_)
            if importances.ndim > 1:
                importances = importances[0]
        else:
            raise ValueError("Model does not have feature_importances_ or coef_ attribute")

        # Create DataFrame
        feature_importance = pd.DataFrame({"feature": feature_names[: len(importances)], "importance": importances})

        # Sort by importance
        feature_importance = feature_importance.sort_values("importance", ascending=False)

        # Limit to top N if specified
        if top_n is not None:
            feature_importance = feature_importance.head(top_n)

        # Plot if requested
        if plot:
            plt.figure(figsize=(10, max(6, len(feature_importance) * 0.3)))

            sns.barplot(x="importance", y="feature", data=feature_importance)

            plt.title("Feature Importance", fontsize=16)
            plt.xlabel("Importance", fontsize=14)
            plt.ylabel("Feature", fontsize=14)
            plt.tight_layout()

        return feature_importance
