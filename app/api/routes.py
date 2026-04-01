import os
import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.job_service import JobService
from app.agents.planner_agent import PlannerAgent
from app.agents.web_research_agent import WebResearchAgent
from app.agents.dataset_agent import DatasetAgent
from app.agents.model_selection_agent import ModelSelectionAgent
from app.agents.training_agent import TrainingAgent
from app.agents.evaluation_agent import EvaluationAgent
from app.agents.visualization_agent import VisualizationAgent
from app.agents.report_generator_agent import ReportGeneratorAgent

router = APIRouter()
job_service = JobService()
planner_agent = PlannerAgent()
web_research_agent = WebResearchAgent()
dataset_agent = DatasetAgent()
model_selection_agent = ModelSelectionAgent()
training_agent = TrainingAgent()
evaluation_agent = EvaluationAgent()
visualization_agent = VisualizationAgent()
report_generator_agent = ReportGeneratorAgent()


async def run_research_pipeline(job_id: str, payload: ResearchRequest):
    job_service.update_job(
        job_id,
        {
            "status": "running",
            "current_step": "planner_started",
            "progress": {"planner": "running"},
            "logs": ["Planner agent started"]
        }
    )

    await asyncio.sleep(0)

    try:
        plan = planner_agent.run(
            user_goal=payload.user_goal,
            problem_type=payload.problem_type or "auto"
        )

        job_service.update_job(
            job_id,
            {
                "plan": plan,
                "current_step": "planner_completed",
                "progress": {"planner": "completed"},
                "logs": ["Planner agent completed"]
            }
        )

        job_service.update_job(
            job_id,
            {
                "current_step": "web_research_started",
                "progress": {"web_research": "running"},
                "logs": ["Web research agent started"]
            }
        )

        await asyncio.sleep(0)

        web_research_info = web_research_agent.run(
            user_goal=payload.user_goal,
            problem_type=plan["problem_type"]
        )

        if web_research_info.get("status") != "completed":
            job_service.update_job(
                job_id,
                {
                    "status": "web_research_failed",
                    "current_step": "web_research_failed",
                    "progress": {"web_research": "failed"},
                    "error": web_research_info.get("message", "Web research failed"),
                    "web_research_info": web_research_info,
                    "logs": ["Web research agent failed"]
                }
            )
            return

        job_service.update_job(
            job_id,
            {
                "web_research_info": web_research_info,
                "current_step": "web_research_completed",
                "progress": {"web_research": "completed"},
                "logs": ["Web research agent completed"]
            }
        )

        job_service.update_job(
            job_id,
            {
                "current_step": "dataset_started",
                "progress": {"dataset": "running"},
                "logs": ["Dataset agent started"]
            }
        )

        await asyncio.sleep(0)

        dataset_info = dataset_agent.run(
            user_goal=payload.user_goal,
            base_dataset_dir=settings.DATASET_DIR
        )

        if dataset_info.get("status") != "completed":
            job_service.update_job(
                job_id,
                {
                    "status": "dataset_failed",
                    "current_step": "dataset_failed",
                    "dataset_info": dataset_info,
                    "progress": {"dataset": "failed"},
                    "error": dataset_info.get("message", "Dataset stage failed"),
                    "logs": ["Dataset agent failed"]
                }
            )
            return

        job_service.update_job(
            job_id,
            {
                "dataset_info": dataset_info,
                "current_step": "dataset_completed",
                "progress": {"dataset": "completed"},
                "logs": ["Dataset agent completed"]
            }
        )

        job_service.update_job(
            job_id,
            {
                "current_step": "model_selection_started",
                "progress": {"model_selection": "running"},
                "logs": ["Model selection agent started"]
            }
        )

        await asyncio.sleep(0)

        model_info = model_selection_agent.run(
            user_goal=payload.user_goal,
            plan=plan,
            dataset_info=dataset_info
        )

        if model_info.get("status") != "completed":
            job_service.update_job(
                job_id,
                {
                    "status": "model_selection_failed",
                    "current_step": "model_selection_failed",
                    "model_info": model_info,
                    "progress": {"model_selection": "failed"},
                    "error": model_info.get("message", "Model selection failed"),
                    "logs": ["Model selection agent failed"]
                }
            )
            return

        job_service.update_job(
            job_id,
            {
                "model_info": model_info,
                "current_step": "model_selection_completed",
                "progress": {"model_selection": "completed"},
                "logs": ["Model selection agent completed"]
            }
        )

        job_service.update_job(
            job_id,
            {
                "current_step": "training_started",
                "progress": {"training": "running"},
                "logs": ["Training agent started"]
            }
        )

        await asyncio.sleep(0)

        training_info = training_agent.run(
            dataset_info=dataset_info,
            model_info=model_info,
            model_dir=settings.MODEL_DIR
        )

        if training_info.get("status") != "completed":
            job_service.update_job(
                job_id,
                {
                    "status": "training_failed",
                    "current_step": "training_failed",
                    "training_info": training_info,
                    "progress": {"training": "failed"},
                    "error": training_info.get("message", "Training failed"),
                    "logs": ["Training agent failed"]
                }
            )
            return

        job_service.update_job(
            job_id,
            {
                "training_info": training_info,
                "current_step": "training_completed",
                "progress": {"training": "completed"},
                "logs": ["Training agent completed"]
            }
        )

        job_service.update_job(
            job_id,
            {
                "current_step": "evaluation_started",
                "progress": {"evaluation": "running"},
                "logs": ["Evaluation agent started"]
            }
        )

        await asyncio.sleep(0)

        evaluation_info = evaluation_agent.run(
            dataset_info=dataset_info,
            model_info=model_info,
            training_info=training_info
        )

        if evaluation_info.get("status") != "completed":
            job_service.update_job(
                job_id,
                {
                    "status": "evaluation_failed",
                    "current_step": "evaluation_failed",
                    "evaluation_info": evaluation_info,
                    "progress": {"evaluation": "failed"},
                    "error": evaluation_info.get("message", "Evaluation failed"),
                    "logs": ["Evaluation agent failed"]
                }
            )
            return

        job_service.update_job(
            job_id,
            {
                "evaluation_info": evaluation_info,
                "current_step": "evaluation_completed",
                "progress": {"evaluation": "completed"},
                "logs": ["Evaluation agent completed"]
            }
        )

        job_service.update_job(
            job_id,
            {
                "current_step": "visualization_started",
                "progress": {"visualization": "running"},
                "logs": ["Visualization agent started"]
            }
        )

        await asyncio.sleep(0)

        visualization_info = visualization_agent.run(
            dataset_info=dataset_info,
            model_info=model_info,
            training_info=training_info,
            evaluation_info=evaluation_info,
            report_dir=settings.REPORT_DIR
        )

        if visualization_info.get("status") != "completed":
            job_service.update_job(
                job_id,
                {
                    "status": "visualization_failed",
                    "current_step": "visualization_failed",
                    "visualization_info": visualization_info,
                    "progress": {"visualization": "failed"},
                    "error": visualization_info.get("message", "Visualization failed"),
                    "logs": ["Visualization agent failed"]
                }
            )
            return

        job_service.update_job(
            job_id,
            {
                "visualization_info": visualization_info,
                "current_step": "visualization_completed",
                "progress": {"visualization": "completed"},
                "logs": ["Visualization agent completed"]
            }
        )

        job_service.update_job(
            job_id,
            {
                "current_step": "report_started",
                "progress": {"report": "running"},
                "logs": ["Report generator agent started"]
            }
        )

        await asyncio.sleep(0)

        report_info = report_generator_agent.run(
            user_goal=payload.user_goal,
            dataset_info=dataset_info,
            model_info=model_info,
            training_info=training_info,
            evaluation_info=evaluation_info,
            visualization_info=visualization_info,
            report_dir=settings.REPORT_DIR
        )

        if report_info.get("status") != "completed":
            job_service.update_job(
                job_id,
                {
                    "status": "report_failed",
                    "current_step": "report_failed",
                    "report_info": report_info,
                    "progress": {"report": "failed"},
                    "error": report_info.get("message", "Report generation failed"),
                    "logs": ["Report generator agent failed"]
                }
            )
            return

        artifacts = []

        if training_info.get("model_path"):
            artifacts.append({
                "type": "model",
                "filename": training_info.get("saved_model_filename"),
                "endpoint": f"{settings.API_V1_PREFIX}/research/{job_id}/download/model"
            })

        if report_info.get("markdown_report", {}).get("filename"):
            artifacts.append({
                "type": "markdown_report",
                "filename": report_info["markdown_report"]["filename"],
                "endpoint": f"{settings.API_V1_PREFIX}/research/{job_id}/download/report/md"
            })

        if report_info.get("pdf_report", {}).get("filename"):
            artifacts.append({
                "type": "pdf_report",
                "filename": report_info["pdf_report"]["filename"],
                "endpoint": f"{settings.API_V1_PREFIX}/research/{job_id}/download/report/pdf"
            })

        for index, plot in enumerate(visualization_info.get("files", [])):
            artifacts.append({
                "type": plot.get("type", f"plot_{index}"),
                "filename": plot.get("filename"),
                "endpoint": f"{settings.API_V1_PREFIX}/research/{job_id}/download/plot/{index}"
            })

        job_service.update_job(
            job_id,
            {
                "status": "completed",
                "current_step": "report_completed",
                "report_info": report_info,
                "artifacts": artifacts,
                "progress": {"report": "completed"},
                "logs": ["Report generator agent completed", "Research pipeline completed successfully"]
            }
        )

    except Exception as e:
        job_service.update_job(
            job_id,
            {
                "status": "failed",
                "current_step": "pipeline_failed",
                "error": str(e),
                "logs": [f"Pipeline crashed: {str(e)}"]
            }
        )


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV
    }


@router.post("/research", response_model=ResearchResponse)
async def create_research_job(payload: ResearchRequest):
    job = job_service.create_job(user_goal=payload.user_goal)
    asyncio.create_task(run_research_pipeline(job["job_id"], payload))

    return ResearchResponse(
        job_id=job["job_id"],
        status="queued",
        message="Research job accepted and started in background"
    )


@router.get("/research/{job_id}")
async def get_research_job(job_id: str):
    job = job_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@router.get("/research/{job_id}/download/model")
async def download_model(job_id: str):
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    training_info = job.get("training_info", {})
    model_path = training_info.get("model_path")

    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        path=model_path,
        filename=os.path.basename(model_path),
        media_type="application/octet-stream"
    )


@router.get("/research/{job_id}/download/report/pdf")
async def download_pdf_report(job_id: str):
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    report_info = job.get("report_info", {})
    pdf_path = report_info.get("pdf_report", {}).get("path")

    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF report not found")

    return FileResponse(
        path=pdf_path,
        filename=os.path.basename(pdf_path),
        media_type="application/pdf"
    )


@router.get("/research/{job_id}/download/report/md")
async def download_markdown_report(job_id: str):
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    report_info = job.get("report_info", {})
    md_path = report_info.get("markdown_report", {}).get("path")

    if not md_path or not os.path.exists(md_path):
        raise HTTPException(status_code=404, detail="Markdown report not found")

    return FileResponse(
        path=md_path,
        filename=os.path.basename(md_path),
        media_type="text/markdown"
    )


@router.get("/research/{job_id}/download/plot/{plot_index}")
async def download_plot(job_id: str, plot_index: int):
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    visualization_info = job.get("visualization_info", {})
    plots = visualization_info.get("files", [])

    if plot_index < 0 or plot_index >= len(plots):
        raise HTTPException(status_code=404, detail="Plot not found")

    plot_path = plots[plot_index].get("path")

    if not plot_path or not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Plot file not found")

    return FileResponse(
        path=plot_path,
        filename=os.path.basename(plot_path),
        media_type="image/png"
    )