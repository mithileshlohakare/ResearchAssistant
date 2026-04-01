import os
from datetime import datetime
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


class TrainingAgent:
    def run(
        self,
        dataset_info: Dict[str, Any],
        model_info: Dict[str, Any],
        model_dir: str = "../models"
    ) -> Dict[str, Any]:
        try:
            dataset_path = dataset_info["dataset_path"]
            target_column = dataset_info["target_column"]
            model_name = model_info["model_name"]
            params = model_info.get("recommended_params", {})

            os.makedirs(model_dir, exist_ok=True)

            df = pd.read_csv(dataset_path)

            if target_column not in df.columns:
                return {
                    "status": "failed",
                    "message": f"Target column '{target_column}' not found in dataset"
                }

            df = df.dropna()

            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if model_info.get("problem_type") == "classification" else None
            )

            model = self._build_model(model_name=model_name, params=params)

            if model is None:
                return {
                    "status": "failed",
                    "message": f"Unsupported model for training: {model_name}"
                }

            model.fit(X_train, y_train)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = model_name.lower()
            model_filename = f"{safe_model_name}_{timestamp}.pkl"
            model_path = os.path.join(model_dir, model_filename)

            joblib.dump(model, model_path)

            return {
                "status": "completed",
                "model_name": model_name,
                "model_path": model_path,
                "train_rows": int(X_train.shape[0]),
                "test_rows": int(X_test.shape[0]),
                "feature_count": int(X.shape[1]),
                "target_column": target_column,
                "features": list(X.columns),
                "saved_model_filename": model_filename
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": str(e)
            }

    def _build_model(self, model_name: str, params: Dict[str, Any]):
        if model_name == "RandomForestClassifier":
            return RandomForestClassifier(**params)

        if model_name == "RandomForestRegressor":
            return RandomForestRegressor(**params)

        return None