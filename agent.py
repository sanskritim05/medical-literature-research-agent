import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from pubmed_tool import assess_confidence, extract_highlight_sentences, search_cached_literature, search_clinical_trials, search_pubmed


load_dotenv()


class ResearchState(TypedDict, total=False):
    question: str
    comparison_question: str | None
    mode: str
    max_results: int
    filters: dict[str, Any]
    history: list[dict[str, str]]
    include_trials: bool
    articles: list[dict[str, Any]]
    comparison_articles: list[dict[str, Any]]
    cached_matches: list[dict[str, Any]]
    trials: list[dict[str, Any]]
    comparison_trials: list[dict[str, Any]]
    synthesis: dict[str, Any]


def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        return ChatOllama(model=model, temperature=temperature)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file or switch LLM_PROVIDER=ollama.")

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(groq_api_key=api_key, model_name=model, temperature=temperature)


def _safe_json_loads(content: str) -> dict[str, Any] | None:
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _stringify_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "No previous conversation."
    lines: list[str] = []
    for turn in history[-6:]:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No previous conversation."


def _format_articles(articles: list[dict[str, Any]]) -> str:
    if not articles:
        return "No articles retrieved."
    lines: list[str] = []
    for index, article in enumerate(articles, start=1):
        lines.append(
            "\n".join(
                [
                    f"[{index}] PMID {article.get('pmid', 'NA')}",
                    f"Title: {article.get('title', 'Untitled')}",
                    f"Authors: {', '.join(article.get('authors', [])[:6]) or 'Unavailable'}",
                    f"Journal/Year: {article.get('journal', 'Unknown')} ({article.get('year', 'Unknown')})",
                    f"Publication types: {', '.join(article.get('publication_types', [])) or article.get('study_type', 'Unknown')}",
                    f"Abstract: {article.get('abstract', '')}",
                ]
            )
        )
    return "\n\n".join(lines)


def _format_trials(trials: list[dict[str, Any]]) -> str:
    if not trials:
        return "No relevant ClinicalTrials.gov records retrieved."
    lines: list[str] = []
    for index, trial in enumerate(trials, start=1):
        lines.append(
            "\n".join(
                [
                    f"[T{index}] {trial.get('title', 'Untitled trial')}",
                    f"NCT ID: {trial.get('nct_id', 'Unavailable')}",
                    f"Status: {trial.get('status', 'Unknown')}",
                    f"Phase: {trial.get('phase', 'Unspecified')}",
                    f"Interventions: {trial.get('interventions', 'Unknown')}",
                    f"Condition: {trial.get('condition', 'Unknown')}",
                ]
            )
        )
    return "\n\n".join(lines)


def _prepare_reference_payload(articles: list[dict[str, Any]], summaries: dict[str, str], question: str, answer: str) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for index, article in enumerate(articles, start=1):
        highlights = extract_highlight_sentences(article.get("abstract", ""), question, answer)
        references.append(
            {
                "index": index,
                "title": article.get("title", ""),
                "authors": article.get("authors", []),
                "link": article.get("link", ""),
                "pmid": article.get("pmid", ""),
                "journal": article.get("journal", ""),
                "year": article.get("year", ""),
                "study_type": article.get("study_type", ""),
                "summary": summaries.get(article.get("pmid", ""), ""),
                "highlights": highlights,
                "abstract": article.get("abstract", ""),
            }
        )
    return references


def _retrieve_literature(state: ResearchState) -> ResearchState:
    filters = state.get("filters", {})
    question = state["question"]
    comparison_question = state.get("comparison_question")
    include_trials = bool(state.get("include_trials", True))
    max_results = state.get("max_results", 5)

    with ThreadPoolExecutor(max_workers=4) as executor:
        articles_future = executor.submit(
            search_pubmed,
            question,
            max_results,
            filters.get("year_from"),
            filters.get("year_to"),
            filters.get("study_type"),
        )
        cached_future = executor.submit(search_cached_literature, question, min(max_results, 3))
        trials_future = executor.submit(search_clinical_trials, question, 3) if include_trials else None

        comparison_articles_future = None
        comparison_trials_future = None
        if state.get("mode") == "compare" and comparison_question:
            comparison_articles_future = executor.submit(
                search_pubmed,
                comparison_question,
                max_results,
                filters.get("year_from"),
                filters.get("year_to"),
                filters.get("study_type"),
            )
            if include_trials:
                comparison_trials_future = executor.submit(search_clinical_trials, comparison_question, 3)

        result: ResearchState = {
            "articles": articles_future.result(),
            "cached_matches": cached_future.result(),
            "trials": trials_future.result() if trials_future else [],
        }
        if comparison_articles_future:
            result["comparison_articles"] = comparison_articles_future.result()
        if comparison_trials_future:
            result["comparison_trials"] = comparison_trials_future.result()
        return result


def _synthesize(state: ResearchState) -> ResearchState:
    llm = _build_llm()
    question = state["question"]
    comparison_question = state.get("comparison_question")
    mode = state.get("mode", "standard")
    history = _stringify_history(state.get("history", []))
    articles = state.get("articles", [])
    comparison_articles = state.get("comparison_articles", [])
    trials = state.get("trials", [])
    comparison_trials = state.get("comparison_trials", [])

    system_prompt = """
You are a careful medical literature research assistant.
Rely only on the provided evidence.
Do not invent results, effect sizes, or recommendations.

Return valid JSON with this shape:
{
  "answer": "Main answer with inline numeric citations like [1] and [2].",
  "plain_language_summary": "Short simpler explanation for a non-specialist reader.",
  "article_summaries": [
    {"pmid": "12345678", "summary": "1-2 sentence summary of the abstract."}
  ],
  "confidence_explanation": "Why the evidence appears strong, moderate, or weak."
}
""".strip()

    if mode == "compare" and comparison_question:
        user_prompt = f"""
Conversation history:
{history}

Primary clinical question:
{question}

Comparison clinical question:
{comparison_question}

Primary published literature:
{_format_articles(articles)}

Primary ongoing trials:
{_format_trials(trials)}

Comparison published literature:
{_format_articles(comparison_articles)}

Comparison ongoing trials:
{_format_trials(comparison_trials)}

Write a head-to-head comparison that:
- summarizes the evidence for each side
- states where one treatment appears stronger, weaker, or equivalent
- mentions if evidence is weak, indirect, or conflicting
- uses inline citations only for PubMed papers, numbered in the same order they were provided in each evidence block

Return JSON only.
""".strip()
    else:
        user_prompt = f"""
Conversation history:
{history}

Clinical question:
{question}

Published literature:
{_format_articles(articles)}

ClinicalTrials.gov records:
{_format_trials(trials)}

Write a concise evidence synthesis that:
- summarizes each abstract in 1-2 sentences
- synthesizes a practical answer in plain language
- mentions any uncertainty, conflict, or evidence limitations
- notes relevant ongoing trials if present
- uses inline citations that match the PubMed order above

Return JSON only.
""".strip()

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    parsed = _safe_json_loads(getattr(response, "content", ""))
    if not parsed:
        parsed = {
            "answer": getattr(response, "content", "").strip() or "No answer generated.",
            "plain_language_summary": "",
            "article_summaries": [],
            "confidence_explanation": "",
        }
    return {"synthesis": parsed}


def _build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("retrieve_literature", _retrieve_literature)
    graph.add_node("synthesize", _synthesize)
    graph.add_edge(START, "retrieve_literature")
    graph.add_edge("retrieve_literature", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


GRAPH = _build_graph()


def run_research(
    question: str,
    *,
    max_results: int = 5,
    year_from: int | None = None,
    year_to: int | None = None,
    study_type: str | None = None,
    include_trials: bool = True,
    history: list[dict[str, str]] | None = None,
    comparison_question: str | None = None,
    mode: str = "standard",
) -> dict[str, Any]:
    filters = {
        "year_from": year_from,
        "year_to": year_to,
        "study_type": study_type or "",
    }
    state: ResearchState = {
        "question": question,
        "comparison_question": comparison_question,
        "mode": mode,
        "max_results": max_results,
        "filters": filters,
        "history": history or [],
        "include_trials": include_trials,
    }
    result = GRAPH.invoke(state)
    synthesis = result.get("synthesis", {})
    articles = result.get("articles", [])
    reference_summaries = {
        str(item.get("pmid", "")).strip(): str(item.get("summary", "")).strip()
        for item in synthesis.get("article_summaries", [])
        if isinstance(item, dict)
    }
    references = _prepare_reference_payload(
        articles=articles,
        summaries=reference_summaries,
        question=question,
        answer=str(synthesis.get("answer", "")),
    )
    confidence = assess_confidence(articles, question)
    if synthesis.get("confidence_explanation"):
        confidence["model_explanation"] = str(synthesis.get("confidence_explanation", "")).strip()

    comparison_references: list[dict[str, Any]] = []
    if mode == "compare":
        comparison_references = _prepare_reference_payload(
            articles=result.get("comparison_articles", []),
            summaries={},
            question=comparison_question or "",
            answer=str(synthesis.get("answer", "")),
        )

    return {
        "answer": str(synthesis.get("answer", "")).strip(),
        "plain_language_summary": str(synthesis.get("plain_language_summary", "")).strip(),
        "confidence": confidence,
        "references": references,
        "comparison_references": comparison_references,
        "trials": result.get("trials", []),
        "comparison_trials": result.get("comparison_trials", []),
        "cached_matches": result.get("cached_matches", []),
        "mode": mode,
        "comparison_question": comparison_question,
        "filters": filters,
    }


def simplify_text(question: str, answer: str) -> str:
    llm = _build_llm()
    system_prompt = "Rewrite medical evidence summaries at an accessible reading level without changing the meaning."
    user_prompt = f"Question: {question}\n\nOriginal answer:\n{answer}\n\nRewrite this in clear plain language at roughly an 8th-grade reading level."
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return getattr(response, "content", "").strip()
