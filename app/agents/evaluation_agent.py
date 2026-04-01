import joblib
import pandas as pd
from typing import Any, Dict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


class EvaluationAgent:
    def run(
        self,
        dataset_info: Dict[str, Any],
        model_info: Dict[str, Any],
        training_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            dataset_path = dataset_info["dataset_path"]
            target_column = dataset_info["target_column"]
            model_path = training_info["model_path"]
            problem_type = model_info.get("problem_type", "classification")

            df = pd.read_csv(dataset_path).dropna()

            if target_column not in df.columns:
                return {
                    "status": "failed",
                    "message": f"Target column '{target_column}' not found in dataset"
                }

            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if problem_type == "classification" else None
            )

            model = joblib.load(model_path)
            y_pred = model.predict(X_test)

            if problem_type == "classification":
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
                rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
                cm = confusion_matrix(y_test, y_pred)

                return {
                    "status": "completed",
                    "problem_type": "classification",
                    "metrics": {
                        "accuracy": round(float(acc), 4),
                        "precision": round(float(prec), 4),
                        "recall": round(float(rec), 4),
                        "f1_score": round(float(f1), 4),
                    },
                    "confusion_matrix": cm.tolist(),
                    "test_samples": int(len(y_test)),
                    "predictions_preview": list(map(int, y_pred[:10]))
                }

            return {
                "status": "failed",
                "message": f"Evaluation not yet implemented for problem type: {problem_type}"
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": str(e)
            }