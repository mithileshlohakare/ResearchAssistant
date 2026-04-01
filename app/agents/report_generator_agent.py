import os
from datetime import datetime
from typing import Any, Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


class ReportGeneratorAgent:
    def run(
        self,
        user_goal: str,
        dataset_info: Dict[str, Any],
        model_info: Dict[str, Any],
        training_info: Dict[str, Any],
        evaluation_info: Dict[str, Any],
        visualization_info: Dict[str, Any],
        report_dir: str = "../reports"
    ) -> Dict[str, Any]:
        try:
            os.makedirs(report_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            md_filename = f"research_report_{timestamp}.md"
            pdf_filename = f"research_report_{timestamp}.pdf"

            md_path = os.path.join(report_dir, md_filename)
            pdf_path = os.path.join(report_dir, pdf_filename)

            markdown_content = self._build_markdown(
                user_goal=user_goal,
                dataset_info=dataset_info,
                model_info=model_info,
                training_info=training_info,
                evaluation_info=evaluation_info,
                visualization_info=visualization_info
            )

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            self._build_pdf(
                pdf_path=pdf_path,
                user_goal=user_goal,
                dataset_info=dataset_info,
                model_info=model_info,
                training_info=training_info,
                evaluation_info=evaluation_info,
                visualization_info=visualization_info
            )

            return {
                "status": "completed",
                "report_dir": report_dir,
                "markdown_report": {
                    "filename": md_filename,
                    "path": md_path
                },
                "pdf_report": {
                    "filename": pdf_filename,
                    "path": pdf_path
                }
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": str(e)
            }

    def _build_markdown(
        self,
        user_goal: str,
        dataset_info: Dict[str, Any],
        model_info: Dict[str, Any],
        training_info: Dict[str, Any],
        evaluation_info: Dict[str, Any],
        visualization_info: Dict[str, Any]
    ) -> str:
        metrics = evaluation_info.get("metrics", {})
        plot_files = visualization_info.get("files", [])

        lines: List[str] = [
            "# Autonomous Research Report",
            "",
            f"## User Goal",
            f"{user_goal}",
            "",
            "## Dataset Information",
            f"- Dataset Name: {dataset_info.get('dataset_name', 'N/A')}",
            f"- Dataset Path: {dataset_info.get('dataset_path', 'N/A')}",
            f"- Rows: {dataset_info.get('rows', 'N/A')}",
            f"- Columns: {dataset_info.get('columns', 'N/A')}",
            f"- Target Column: {dataset_info.get('target_column', 'N/A')}",
            "",
            "## Model Selection",
            f"- Model Name: {model_info.get('model_name', 'N/A')}",
            f"- Model Family: {model_info.get('model_family', 'N/A')}",
            f"- Library: {model_info.get('library', 'N/A')}",
            f"- Reason: {model_info.get('reason', 'N/A')}",
            "",
            "## Training Information",
            f"- Train Rows: {training_info.get('train_rows', 'N/A')}",
            f"- Test Rows: {training_info.get('test_rows', 'N/A')}",
            f"- Feature Count: {training_info.get('feature_count', 'N/A')}",
            f"- Saved Model: {training_info.get('saved_model_filename', 'N/A')}",
            "",
            "## Evaluation Metrics",
            f"- Accuracy: {metrics.get('accuracy', 'N/A')}",
            f"- Precision: {metrics.get('precision', 'N/A')}",
            f"- Recall: {metrics.get('recall', 'N/A')}",
            f"- F1 Score: {metrics.get('f1_score', 'N/A')}",
            "",
            "## Visualizations",
        ]

        for plot in plot_files:
            lines.append(f"- {plot.get('type', 'plot')}: {plot.get('path', 'N/A')}")

        lines.extend([
            "",
            "## Conclusion",
            "The system successfully completed planning, dataset preparation, model selection, training, evaluation, visualization, and report generation for the requested research goal.",
            ""
        ])

        return "\n".join(lines)

    def _build_pdf(
        self,
        pdf_path: str,
        user_goal: str,
        dataset_info: Dict[str, Any],
        model_info: Dict[str, Any],
        training_info: Dict[str, Any],
        evaluation_info: Dict[str, Any],
        visualization_info: Dict[str, Any]
    ) -> None:
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4

        y = height - 50
        line_height = 18

        def write_line(text: str, font="Helvetica", size=11):
            nonlocal y
            if y < 60:
                c.showPage()
                y = height - 50
            c.setFont(font, size)
            c.drawString(40, y, text[:110])
            y -= line_height

        c.setTitle("Autonomous Research Report")

        write_line("Autonomous Research Report", font="Helvetica-Bold", size=16)
        write_line("")

        write_line("User Goal", font="Helvetica-Bold", size=13)
        write_line(user_goal)
        write_line("")

        write_line("Dataset Information", font="Helvetica-Bold", size=13)
        write_line(f"Dataset Name: {dataset_info.get('dataset_name', 'N/A')}")
        write_line(f"Rows: {dataset_info.get('rows', 'N/A')}")
        write_line(f"Columns: {dataset_info.get('columns', 'N/A')}")
        write_line(f"Target Column: {dataset_info.get('target_column', 'N/A')}")
        write_line("")

        write_line("Model Selection", font="Helvetica-Bold", size=13)
        write_line(f"Model Name: {model_info.get('model_name', 'N/A')}")
        write_line(f"Library: {model_info.get('library', 'N/A')}")
        write_line(f"Problem Type: {model_info.get('problem_type', 'N/A')}")
        write_line("")

        write_line("Training Information", font="Helvetica-Bold", size=13)
        write_line(f"Train Rows: {training_info.get('train_rows', 'N/A')}")
        write_line(f"Test Rows: {training_info.get('test_rows', 'N/A')}")
        write_line(f"Saved Model: {training_info.get('saved_model_filename', 'N/A')}")
        write_line("")

        metrics = evaluation_info.get("metrics", {})
        write_line("Evaluation Metrics", font="Helvetica-Bold", size=13)
        write_line(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
        write_line(f"Precision: {metrics.get('precision', 'N/A')}")
        write_line(f"Recall: {metrics.get('recall', 'N/A')}")
        write_line(f"F1 Score: {metrics.get('f1_score', 'N/A')}")
        write_line("")

        write_line("Generated Visualizations", font="Helvetica-Bold", size=13)
        for plot in visualization_info.get("files", []):
            write_line(f"{plot.get('type', 'plot')}: {plot.get('filename', 'N/A')}")

        write_line("")
        write_line("Conclusion", font="Helvetica-Bold", size=13)
        write_line("The requested research pipeline completed successfully.")

        c.save()