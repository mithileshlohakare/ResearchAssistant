from typing import Dict, Any


class ModelSelectionAgent:
    def run(self, user_goal: str, plan: Dict[str, Any], dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        problem_type = plan.get("problem_type", "classification")
        rows = dataset_info.get("rows", 0)
        columns = dataset_info.get("columns", 0)

        if problem_type == "classification":
            return {
                "status": "completed",
                "problem_type": problem_type,
                "model_name": "RandomForestClassifier",
                "model_family": "tree_based",
                "library": "scikit-learn",
                "reason": (
                    "RandomForestClassifier is robust for tabular classification, "
                    "handles non-linear patterns well, and works strongly on small-to-medium datasets."
                ),
                "recommended_params": {
                    "n_estimators": 200,
                    "max_depth": 10,
                    "random_state": 42
                },
                "dataset_summary_used": {
                    "rows": rows,
                    "columns": columns
                }
            }

        if problem_type == "regression":
            return {
                "status": "completed",
                "problem_type": problem_type,
                "model_name": "RandomForestRegressor",
                "model_family": "tree_based",
                "library": "scikit-learn",
                "reason": (
                    "RandomForestRegressor is a strong baseline for tabular regression, "
                    "captures non-linear relationships, and requires minimal preprocessing."
                ),
                "recommended_params": {
                    "n_estimators": 200,
                    "max_depth": 12,
                    "random_state": 42
                },
                "dataset_summary_used": {
                    "rows": rows,
                    "columns": columns
                }
            }

        if problem_type == "time_series":
            return {
                "status": "completed",
                "problem_type": problem_type,
                "model_name": "XGBoostRegressor",
                "model_family": "boosting",
                "library": "xgboost",
                "reason": (
                    "XGBoost is a strong practical baseline for engineered-feature time series prediction "
                    "and performs well when sequential data is converted into supervised tabular form."
                ),
                "recommended_params": {
                    "n_estimators": 300,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "random_state": 42
                },
                "dataset_summary_used": {
                    "rows": rows,
                    "columns": columns
                }
            }

        if problem_type == "detection":
            return {
                "status": "completed",
                "problem_type": problem_type,
                "model_name": "ResNet50",
                "model_family": "deep_learning",
                "library": "pytorch",
                "reason": (
                    "ResNet50 is a standard strong vision backbone and a good starting point "
                    "for image-based detection/classification pipelines."
                ),
                "recommended_params": {
                    "epochs": 10,
                    "batch_size": 16,
                    "learning_rate": 0.001
                },
                "dataset_summary_used": {
                    "rows": rows,
                    "columns": columns
                }
            }

        return {
            "status": "failed",
            "problem_type": problem_type,
            "message": f"No model strategy found for problem type: {problem_type}"
        }