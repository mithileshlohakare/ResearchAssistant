import os
from datetime import datetime
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


class VisualizationAgent:
    def run(
        self,
        dataset_info: Dict[str, Any],
        model_info: Dict[str, Any],
        training_info: Dict[str, Any],
        evaluation_info: Dict[str, Any],
        report_dir: str = "../reports"
    ) -> Dict[str, Any]:
        try:
            plots_dir = os.path.join(report_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            dataset_path = dataset_info["dataset_path"]
            target_column = dataset_info["target_column"]
            model_path = training_info["model_path"]
            problem_type = model_info.get("problem_type", "classification")

            df = pd.read_csv(dataset_path).dropna()
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

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            saved_files = []

            if problem_type == "classification" and "confusion_matrix" in evaluation_info:
                cm = evaluation_info["confusion_matrix"]

                plt.figure(figsize=(6, 5))
                plt.imshow(cm, interpolation="nearest")
                plt.title("Confusion Matrix")
                plt.colorbar()
                plt.xticks([0, 1], ["Pred 0", "Pred 1"])
                plt.yticks([0, 1], ["Actual 0", "Actual 1"])
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")

                for i in range(len(cm)):
                    for j in range(len(cm[i])):
                        plt.text(j, i, str(cm[i][j]), ha="center", va="center")

                cm_filename = f"confusion_matrix_{timestamp}.png"
                cm_path = os.path.join(plots_dir, cm_filename)
                plt.tight_layout()
                plt.savefig(cm_path, dpi=200, bbox_inches="tight")
                plt.close()

                saved_files.append({
                    "type": "confusion_matrix",
                    "path": cm_path,
                    "filename": cm_filename
                })

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_names = list(X.columns)

                pairs = sorted(
                    zip(feature_names, importances),
                    key=lambda item: item[1],
                    reverse=True
                )[:10]

                top_features = [p[0] for p in pairs]
                top_scores = [float(p[1]) for p in pairs]

                plt.figure(figsize=(8, 5))
                plt.barh(top_features[::-1], top_scores[::-1])
                plt.title("Top Feature Importances")
                plt.xlabel("Importance Score")
                plt.ylabel("Feature")

                fi_filename = f"feature_importance_{timestamp}.png"
                fi_path = os.path.join(plots_dir, fi_filename)
                plt.tight_layout()
                plt.savefig(fi_path, dpi=200, bbox_inches="tight")
                plt.close()

                saved_files.append({
                    "type": "feature_importance",
                    "path": fi_path,
                    "filename": fi_filename,
                    "top_features": pairs
                })
            else:
                result = permutation_importance(
                    model, X_test, y_test, n_repeats=5, random_state=42
                )

                pairs = sorted(
                    zip(X.columns, result.importances_mean),
                    key=lambda item: item[1],
                    reverse=True
                )[:10]

                top_features = [p[0] for p in pairs]
                top_scores = [float(p[1]) for p in pairs]

                plt.figure(figsize=(8, 5))
                plt.barh(top_features[::-1], top_scores[::-1])
                plt.title("Top Permutation Importances")
                plt.xlabel("Importance Score")
                plt.ylabel("Feature")

                fi_filename = f"feature_importance_{timestamp}.png"
                fi_path = os.path.join(plots_dir, fi_filename)
                plt.tight_layout()
                plt.savefig(fi_path, dpi=200, bbox_inches="tight")
                plt.close()

                saved_files.append({
                    "type": "feature_importance",
                    "path": fi_path,
                    "filename": fi_filename,
                    "top_features": pairs
                })

            return {
                "status": "completed",
                "plots_dir": plots_dir,
                "files": saved_files
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": str(e)
            }