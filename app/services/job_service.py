import uuid
from typing import Dict, Any


class JobService:
    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, user_goal: str) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())

        job_data = {
            "job_id": job_id,
            "user_goal": user_goal,
            "status": "queued",
            "current_step": "job_created",
            "result": None,
            "error": None,
            "progress": {
                "planner": "pending",
                "web_research": "pending",
                "dataset": "pending",
                "model_selection": "pending",
                "training": "pending",
                "evaluation": "pending",
                "visualization": "pending",
                "report": "pending",
            },
            "logs": [
                "Job created"
            ]
        }

        self.jobs[job_id] = job_data
        return job_data

    def get_job(self, job_id: str) -> Dict[str, Any] | None:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> Dict[str, Any] | None:
        job = self.jobs.get(job_id)
        if not job:
            return None

        for key, value in updates.items():
            if key == "progress" and isinstance(value, dict):
                job.setdefault("progress", {})
                job["progress"].update(value)
            elif key == "logs" and isinstance(value, list):
                job.setdefault("logs", [])
                job["logs"].extend(value)
            else:
                job[key] = value

        return job