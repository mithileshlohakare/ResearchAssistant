from typing import Optional, Literal
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    user_goal: str = Field(..., min_length=5, max_length=500)
    preferred_dataset: Optional[str] = None
    preferred_model: Optional[str] = None
    problem_type: Optional[
        Literal["classification", "regression", "detection", "time_series", "auto"]
    ] = "auto"


class ResearchResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ArtifactResponse(BaseModel):
    artifact_type: str
    filename: str
    endpoint: str