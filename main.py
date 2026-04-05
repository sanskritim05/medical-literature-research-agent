import io
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from agent import run_research, simplify_text


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
SESSION_MEMORY: dict[str, list[dict[str, str]]] = {}

app = FastAPI(title="Medical Literature Research Agent")
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Clinical question in natural language.")
    max_results: int = Field(default=5, ge=1, le=10)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    study_type: str = Field(default="")
    include_trials: bool = Field(default=True)
    session_id: str | None = Field(default=None)
    mode: str = Field(default="standard")
    comparison_question: str | None = Field(default=None)


class SimplifyRequest(BaseModel):
    question: str = Field(..., min_length=5)
    answer: str = Field(..., min_length=10)


class PDFRequest(BaseModel):
    question: str
    answer: str
    confidence: dict[str, Any] = Field(default_factory=dict)
    references: list[dict[str, Any]] = Field(default_factory=list)
    trials: list[dict[str, Any]] = Field(default_factory=list)


def _get_session_history(session_id: str) -> list[dict[str, str]]:
    return SESSION_MEMORY.get(session_id, [])


def _append_session_turns(session_id: str, question: str, answer: str) -> None:
    turns = SESSION_MEMORY.setdefault(session_id, [])
    turns.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    SESSION_MEMORY[session_id] = turns[-12:]


def _run_research_with_activity(request: ResearchRequest) -> tuple[dict[str, Any], list[dict[str, str]]]:
    activity: list[dict[str, str]] = []
    session_id = request.session_id or str(uuid.uuid4())
    history = _get_session_history(session_id)

    activity.append({"step": "context", "detail": "Loaded conversation history and prepared the research plan."})
    activity.append({"step": "pubmed", "detail": "Searching PubMed with the selected filters and checking the local Chroma cache."})
    if request.include_trials:
        activity.append({"step": "trials", "detail": "Pulling relevant ongoing studies from ClinicalTrials.gov."})
    if request.mode == "compare" and request.comparison_question:
        activity.append({"step": "compare", "detail": "Running parallel searches for both treatment paths."})
    activity.append({"step": "synthesis", "detail": "Summarizing abstracts, scoring evidence confidence, and drafting the final answer."})

    result = run_research(
        request.question,
        max_results=request.max_results,
        year_from=request.year_from,
        year_to=request.year_to,
        study_type=request.study_type,
        include_trials=request.include_trials,
        history=history,
        comparison_question=request.comparison_question,
        mode=request.mode,
    )
    _append_session_turns(session_id, request.question, result.get("answer", ""))
    result["session_id"] = session_id
    result["activity"] = activity + [{"step": "done", "detail": "Research completed and results are ready."}]
    return result, result["activity"]


def _build_pdf(request: PDFRequest) -> bytes:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            textColor=HexColor("#0d3b66"),
            fontSize=18,
            leading=24,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            textColor=HexColor("#1d70b8"),
            spaceAfter=8,
        )
    )

    story: list[Any] = [
        Paragraph("Medical Literature Research Report", styles["ReportTitle"]),
        Spacer(1, 0.15 * inch),
        Paragraph(f"<b>Question:</b> {request.question}", styles["BodyText"]),
        Spacer(1, 0.15 * inch),
        Paragraph("<b>Synthesized Answer</b>", styles["SectionHeading"]),
        Paragraph(request.answer.replace("\n", "<br/>"), styles["BodyText"]),
        Spacer(1, 0.15 * inch),
    ]

    if request.confidence:
        confidence_line = f"{request.confidence.get('label', 'Unknown')} ({request.confidence.get('score', 'NA')})"
        story.extend(
            [
                Paragraph("<b>Confidence</b>", styles["SectionHeading"]),
                Paragraph(f"{confidence_line}<br/>{request.confidence.get('rationale', '')}", styles["BodyText"]),
                Spacer(1, 0.15 * inch),
            ]
        )

    if request.references:
        story.append(Paragraph("<b>References</b>", styles["SectionHeading"]))
        for index, reference in enumerate(request.references, start=1):
            authors = ", ".join(reference.get("authors", [])[:8]) or "Authors unavailable"
            text = (
                f"{index}. <b>{reference.get('title', 'Untitled')}</b><br/>"
                f"{authors}<br/>"
                f"PMID: {reference.get('pmid', 'Unavailable')}<br/>"
                f"{reference.get('link', '')}"
            )
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 0.12 * inch))

    if request.trials:
        story.append(Paragraph("<b>Ongoing Trials</b>", styles["SectionHeading"]))
        for trial in request.trials:
            text = (
                f"<b>{trial.get('title', 'Untitled trial')}</b><br/>"
                f"NCT ID: {trial.get('nct_id', 'Unavailable')}<br/>"
                f"Status: {trial.get('status', 'Unknown')}<br/>"
                f"{trial.get('link', '')}"
            )
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 0.12 * inch))

    document.build(story)
    return buffer.getvalue()


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/research")
async def research(request: ResearchRequest) -> dict[str, Any]:
    try:
        result, _ = _run_research_with_activity(request)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/research/stream")
async def research_stream(request: ResearchRequest) -> StreamingResponse:
    def event_stream():
        try:
            session_id = request.session_id or str(uuid.uuid4())
            intro_events = [
                {"type": "activity", "data": {"step": "context", "detail": "Loaded conversation memory and validated the request."}},
                {"type": "activity", "data": {"step": "pubmed", "detail": "Searching PubMed and consulting the Chroma cache."}},
            ]
            if request.include_trials:
                intro_events.append({"type": "activity", "data": {"step": "trials", "detail": "Retrieving ClinicalTrials.gov studies."}})
            if request.mode == "compare" and request.comparison_question:
                intro_events.append({"type": "activity", "data": {"step": "compare", "detail": "Running parallel evidence searches for both treatments."}})
            intro_events.append({"type": "activity", "data": {"step": "synthesis", "detail": "Summarizing evidence and drafting the final answer."}})
            for event in intro_events:
                yield f"data: {json.dumps(event)}\n\n"

            result, _ = _run_research_with_activity(request.model_copy(update={"session_id": session_id}))
            yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(exc)}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/simplify")
async def simplify_answer(request: SimplifyRequest) -> dict[str, str]:
    try:
        return {"simplified_answer": simplify_text(request.question, request.answer)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/export/pdf")
async def export_pdf(request: PDFRequest) -> StreamingResponse:
    try:
        pdf_bytes = _build_pdf(request)
        headers = {"Content-Disposition": 'attachment; filename="medical-literature-report.pdf"'}
        return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers=headers)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
