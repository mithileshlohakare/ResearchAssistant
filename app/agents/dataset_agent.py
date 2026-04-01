from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd


class DatasetAgent:
    def run(self, user_goal: str, base_dataset_dir: str = "../datasets") -> Dict[str, Any]:
        dataset_config = self._select_dataset(user_goal=user_goal, base_dataset_dir=base_dataset_dir)
        dataset_path = dataset_config["path"]

        if not os.path.exists(dataset_path):
            return {
                "status": "failed",
                "message": f"Dataset file not found: {dataset_path}",
                "dataset_name": dataset_config["dataset_name"],
                "dataset_path": dataset_path,
            }

        df = pd.read_csv(dataset_path)

        return {
            "status": "completed",
            "dataset_name": dataset_config["dataset_name"],
            "dataset_path": dataset_path,
            "target_column": dataset_config["target_column"],
            "problem_type": dataset_config["problem_type"],
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "feature_columns": [col for col in df.columns if col != dataset_config["target_column"]],
            "preview": df.head(5).to_dict(orient="records"),
            "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        }

    def _select_dataset(self, user_goal: str, base_dataset_dir: str) -> Dict[str, Any]:
        goal = user_goal.lower()

        if "heart" in goal:
            return {
                "dataset_name": "Heart Disease Dataset",
                "path": os.path.join(base_dataset_dir, "heart_disease.csv"),
                "target_column": "target",
                "problem_type": "classification",
            }

        if "diabetes" in goal:
            return {
                "dataset_name": "Diabetes Dataset",
                "path": os.path.join(base_dataset_dir, "diabetes.csv"),
                "target_column": "Outcome",
                "problem_type": "classification",
            }

        if "house" in goal or "price" in goal:
            return {
                "dataset_name": "House Prices Dataset",
                "path": os.path.join(base_dataset_dir, "house_prices.csv"),
                "target_column": "SalePrice",
                "problem_type": "regression",
            }

        return {
            "dataset_name": "Heart Disease Dataset",
            "path": os.path.join(base_dataset_dir, "heart_disease.csv"),
            "target_column": "target",
            "problem_type": "classification",
        }