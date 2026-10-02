"""Run the research pipeline on a fixed set of clinical questions and write eval/results.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import _invoke_llm, _safe_json_loads, run_research  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402


QUESTIONS_PATH = Path(__file__).with_name("questions.json")
RESULTS_PATH = Path(__file__).with_name("results.json")
SUBSET_RESULTS_PATH = Path(__file__).with_name("results_subset.json")
SUBSET_IDS = (
    "statin-secondary-prevention",
    "ace-hfref",
    "smoking-cessation-lung-cancer",
    "metformin-t2d",
    "warfarin-af-stroke",
)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 3)


def grade_answer(question: str, reference_conclusion: str, answer: str) -> dict:
    """Ask the model whether the answer reaches the same conclusion as the reference."""
    if not answer.strip() or not reference_conclusion.strip():
        return {"answer_correct": None, "answer_grade_reason": "Missing answer or reference conclusion."}
    system_prompt = """
You grade a medical research answer against a reference conclusion.
Judge only the bottom-line direction (benefit, harm, or no meaningful benefit).
Ignore wording, citations, and extra caveats when the direction still matches.
correct is false when the answer is empty, answers a different question, or reaches the opposite conclusion.
Return valid JSON only: {"correct": true, "reason": "one sentence"}
""".strip()
    user_prompt = (
        f"Question: {question}\n\n"
        f"Reference conclusion: {reference_conclusion}\n\n"
        f"Answer:\n{answer}"
    )
    try:
        response = _invoke_llm([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        parsed = _safe_json_loads(getattr(response, "content", "")) or {}
        correct = parsed.get("correct")
        if not isinstance(correct, bool):
            return {"answer_correct": None, "answer_grade_reason": "Grader did not return a boolean."}
        return {
            "answer_correct": correct,
            "answer_grade_reason": str(parsed.get("reason") or "").strip(),
        }
    except Exception as exc:
        return {"answer_correct": None, "answer_grade_reason": f"{type(exc).__name__}: {exc}"}


def evaluate_question(item: dict, *, max_results: int) -> dict:
    started = time.perf_counter()
    try:
        result = run_research(
            item["question"],
            max_results=max_results,
            include_trials=False,
            mode="standard",
        )
        latency = round(time.perf_counter() - started, 2)
        audit = result.get("citation_audit") or {}
        label = str((result.get("confidence") or {}).get("label") or "")
        answer = str(result.get("answer") or "").strip()
        grade = grade_answer(item["question"], item.get("reference_conclusion") or "", answer)
        return {
            "id": item["id"],
            "question": item["question"],
            "expected_strength": item["expected_strength"],
            "reference_conclusion": item.get("reference_conclusion") or "",
            "answer": answer,
            "confidence_label": label,
            "confidence_matches_expected": label == item["expected_strength"],
            "latency_seconds": latency,
            "citation_count": int(audit.get("citation_count") or 0),
            "valid_citation_count": int(audit.get("valid_citation_count") or 0),
            "claim_count": int(audit.get("claim_count") or 0),
            "supported_claim_count": int(audit.get("supported_claim_count") or 0),
            "validity_rate": audit.get("validity_rate"),
            "support_rate": audit.get("support_rate"),
            "unsupported_claims": audit.get("unsupported_claims") or [],
            "answer_correct": grade["answer_correct"],
            "answer_grade_reason": grade["answer_grade_reason"],
            "error": None,
        }
    except Exception as exc:
        return {
            "id": item["id"],
            "question": item["question"],
            "expected_strength": item["expected_strength"],
            "reference_conclusion": item.get("reference_conclusion") or "",
            "answer": "",
            "confidence_label": None,
            "confidence_matches_expected": False,
            "latency_seconds": round(time.perf_counter() - started, 2),
            "citation_count": 0,
            "valid_citation_count": 0,
            "claim_count": 0,
            "supported_claim_count": 0,
            "validity_rate": None,
            "support_rate": None,
            "unsupported_claims": [],
            "answer_correct": None,
            "answer_grade_reason": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_results(rows: list[dict], questions: list[dict], max_results: int, *, results_path: Path) -> dict:
    completed = [row for row in rows if not row["error"]]
    citation_total = sum(row["citation_count"] for row in completed)
    valid_total = sum(row["valid_citation_count"] for row in completed)
    uncited_answers = sum(1 for row in completed if row["citation_count"] == 0)
    claim_total = sum(row["claim_count"] for row in completed)
    supported_total = sum(row["supported_claim_count"] for row in completed)
    latencies = [row["latency_seconds"] for row in completed]
    matches = sum(1 for row in completed if row["confidence_matches_expected"])
    graded = [row for row in completed if isinstance(row.get("answer_correct"), bool)]
    correct = sum(1 for row in graded if row["answer_correct"])

    summary = {
        "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        "max_article_summaries": int(os.getenv("MAX_ARTICLE_SUMMARIES", "4")),
        "question_count": len(questions),
        "completed_count": len(completed),
        "error_count": len(rows) - len(completed),
        "max_results": max_results,
        "include_trials": False,
        "citation_validity_rate": _rate(valid_total, citation_total + uncited_answers),
        "uncited_answers": uncited_answers,
        "claim_support_rate": _rate(supported_total, claim_total),
        "average_latency_seconds": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "confidence_label_match_rate": _rate(matches, len(completed)),
        "answer_correctness_rate": _rate(correct, len(graded)),
        "graded_count": len(graded),
        "correct_count": correct,
        "citation_count": citation_total,
        "valid_citation_count": valid_total,
        "claim_count": claim_total,
        "supported_claim_count": supported_total,
    }
    payload = {"summary": summary, "questions": rows}
    results_path.write_text(json.dumps(payload, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clinical-question eval.")
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Run 5 fixed questions (statins, ACE inhibitors, smoking cessation, metformin, warfarin).",
    )
    args = parser.parse_args()
    questions = json.loads(QUESTIONS_PATH.read_text())
    results_path = RESULTS_PATH
    if args.subset:
        by_id = {item["id"]: item for item in questions}
        missing = [item_id for item_id in SUBSET_IDS if item_id not in by_id]
        if missing:
            raise SystemExit(f"Subset questions missing from questions.json: {', '.join(missing)}")
        questions = [by_id[item_id] for item_id in SUBSET_IDS]
        results_path = SUBSET_RESULTS_PATH
    max_results = int(os.getenv("EVAL_MAX_RESULTS", "4"))
    rows = []
    for index, item in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {item['id']}", flush=True)
        row = evaluate_question(item, max_results=max_results)
        rows.append(row)
        summary = _write_results(rows, questions, max_results, results_path=results_path)
        print(
            f"  label={row['confidence_label']} expected={row['expected_strength']} "
            f"correct={row['answer_correct']} latency={row['latency_seconds']}s error={row['error']}",
            flush=True,
        )
    print(json.dumps(summary, indent=2), flush=True)
    print("\nSample graded answers", flush=True)
    sample = rows[:4]
    seen = {row["id"] for row in sample}
    for row in rows:
        if row["id"] in seen:
            continue
        if row.get("answer_correct") is False:
            sample.append(row)
        if len(sample) >= 8:
            break
    for row in sample:
        print(f"\n[{row['id']}] correct={row.get('answer_correct')}", flush=True)
        print(f"Reference: {row.get('reference_conclusion')}", flush=True)
        print(f"Reason: {row.get('answer_grade_reason')}", flush=True)
        print(f"Answer: {(row.get('answer') or '')[:700]}", flush=True)


if __name__ == "__main__":
    main()
