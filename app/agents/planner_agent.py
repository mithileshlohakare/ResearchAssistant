from typing import Dict, Any, List


class PlannerAgent:
    def run(self, user_goal: str, problem_type: str = "auto") -> Dict[str, Any]:
        detected_problem_type = self._detect_problem_type(user_goal, problem_type)
        workflow_steps = self._build_workflow_steps(detected_problem_type)

        return {
            "goal": user_goal,
            "problem_type": detected_problem_type,
            "workflow": workflow_steps,
            "assigned_agents": [
                "planner_agent",
                "web_research_agent",
                "dataset_agent",
                "model_selection_agent",
                "training_agent",
                "evaluation_agent",
                "visualization_agent",
                "report_generator_agent"
            ]
        }

    def _detect_problem_type(self, user_goal: str, problem_type: str) -> str:
        if problem_type != "auto":
            return problem_type

        goal = user_goal.lower()

        if any(k in goal for k in ["stock", "forecast", "time series", "trend", "price prediction over time"]):
            return "time_series"

        if any(k in goal for k in ["fracture", "x-ray", "image", "detect", "object detection", "vision"]):
            return "detection"

        if any(k in goal for k in ["house price", "sales", "revenue", "score prediction", "predict price"]):
            return "regression"

        if any(k in goal for k in ["classify", "disease", "heart", "diabetes", "fraud", "spam", "sentiment"]):
            return "classification"

        return "classification"

    def _build_workflow_steps(self, problem_type: str) -> List[Dict[str, Any]]:
        steps = [
            {"step": 1, "agent": "web_research_agent", "task": "discover_web_sources"},
            {"step": 2, "agent": "dataset_agent", "task": "prepare_dataset"},
            {"step": 3, "agent": "model_selection_agent", "task": "select_best_model"},
            {"step": 4, "agent": "training_agent", "task": "train_model"},
            {"step": 5, "agent": "evaluation_agent", "task": "evaluate_model"},
            {"step": 6, "agent": "visualization_agent", "task": "generate_visualizations"},
            {"step": 7, "agent": "report_generator_agent", "task": "generate_report"}
        ]

        if problem_type == "detection":
            steps.insert(
                4,
                {"step": 4.5, "agent": "training_agent", "task": "save_prediction_samples"}
            )

        return steps