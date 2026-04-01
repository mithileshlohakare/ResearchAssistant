from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    job_id: str
    user_goal: str
    problem_type: str
    plan: Dict[str, Any]
    dataset_info: Dict[str, Any]
    model_info: Dict[str, Any]
    training_info: Dict[str, Any]
    evaluation_info: Dict[str, Any]
    visualization_info: Dict[str, Any]
    report_info: Dict[str, Any]
    artifacts: List[Dict[str, str]]
    logs: List[str]
    status: str
    error: Optional[str]