import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from pubmed_tool import (
    _normalize_pubmed_query,
    extract_highlight_sentences,
    search_cached_literature,
    search_clinical_trials,
    resolve_mesh_heading,
    search_pubmed,
)


load_dotenv()

logger = logging.getLogger(__name__)


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
    query_plan: dict[str, Any]
    comparison_query_plan: dict[str, Any]
    article_summaries: list[dict[str, str]]
    synthesis: dict[str, Any]
    needs_expert_review: bool
    relevance_scored: int
    relevance_total: int


MAX_ARTICLE_SUMMARIES = int(os.getenv("MAX_ARTICLE_SUMMARIES", "4"))
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "15"))
_SUMMARY_CACHE_LOCK = threading.Lock()
_SUMMARY_CACHE_PATH = (
    Path("/tmp/literature-cache/summary_cache.json")
    if os.getenv("VERCEL")
    else Path(__file__).resolve().parent / ".cache" / "literature" / "summary_cache.json"
)


def _model_supports_low_reasoning() -> bool:
    return "gpt-oss" in os.getenv("GROQ_MODEL", "")


def _build_llm(max_tokens: int | None = None, reasoning_effort: str | None = None):
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "LLM_PROVIDER=ollama requires langchain-ollama. Install it locally or switch to groq for Vercel."
            ) from exc
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        return ChatOllama(model=model, temperature=temperature)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file or Vercel environment variables.")

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    kwargs: dict[str, Any] = {
        "groq_api_key": api_key,
        "model_name": model,
        "temperature": temperature,
        "timeout": 120,
        "max_retries": 0,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort:
        kwargs["model_kwargs"] = {"reasoning_effort": reasoning_effort}
    return ChatGroq(**kwargs)


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", str(exc), flags=re.IGNORECASE)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2) or 0)
        return min(minutes * 60 + seconds + 3, 20 * 60)
    return float(2 ** attempt)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def _response_metadata(response: Any) -> dict[str, Any]:
    meta = getattr(response, "response_metadata", None) or {}
    return meta if isinstance(meta, dict) else {}


def _reply_was_cut_off(response: Any) -> bool:
    if _response_text(response).strip():
        return False
    meta = _response_metadata(response)
    usage = meta.get("token_usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    details = usage.get("completion_tokens_details") or {}
    reasoning = int(details.get("reasoning_tokens") or 0) if isinstance(details, dict) else 0
    completion = int(usage.get("completion_tokens") or 0)
    finish = str(meta.get("finish_reason") or "")
    return finish == "length" or (completion > 0 and reasoning >= completion)


def _invoke_llm(messages: list[Any], *, reasoning_effort: str | None = None) -> Any:
    llm = _build_llm(reasoning_effort=reasoning_effort)
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = llm.invoke(messages)
            if not _response_text(response).strip():
                meta = _response_metadata(response)
                logger.warning(
                    "Empty model reply finish_reason=%s token_usage=%s",
                    meta.get("finish_reason"),
                    meta.get("token_usage"),
                )
                if _reply_was_cut_off(response):
                    higher = 4096
                    logger.warning("Empty reply was cut off; retrying with max_tokens=%s", higher)
                    return _build_llm(max_tokens=higher, reasoning_effort=reasoning_effort).invoke(messages)
            return response
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = any(
                token in message
                for token in (
                    "429",
                    "rate limit",
                    "rate_limit",
                    "503",
                    "over capacity",
                    "tokens per minute",
                    "timeout",
                    "timed out",
                    "connection",
                    "ssl",
                )
            )
            if not retryable or attempt == 4:
                raise
            if "tokens per minute" in message or "413" in message:
                time.sleep(min(65, 25 + 15 * attempt))
            elif "timeout" in message or "timed out" in message or "connection" in message or "ssl" in message:
                time.sleep(5 * (attempt + 1))
            else:
                time.sleep(_retry_delay_seconds(exc, attempt))
    raise last_error or RuntimeError("LLM call failed.")


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _safe_json_loads(content: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        parsed = _json_object(fenced.group(1).strip())
        if parsed is not None:
            return parsed
    return _json_object(text)


_DESIGN_TIERS: list[tuple[float, str, tuple[str, ...]]] = [
    (4.0, "meta-analysis or systematic review", ("meta-analysis", "systematic review")),
    (3.0, "randomized controlled trial", ("randomized controlled trial", "randomised controlled trial", "randomized trial")),
    (2.2, "clinical trial", ("clinical trial",)),
    (2.0, "observational study", ("observational", "cohort", "case-control", "case control", "comparative study")),
    (1.0, "case report", ("case report", "case reports")),
]

_RELEVANCE_STOPWORDS = {
    "about",
    "after",
    "among",
    "and",
    "are",
    "compared",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "patients",
    "than",
    "that",
    "the",
    "this",
    "what",
    "when",
    "with",
    "would",
}


def _valid_pubmed_query(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if len(text) < 3 or len(text) > 500:
        return False
    return bool(re.search(r"[A-Za-z]", text))


def _clean_query_term(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace('"', "")).strip()


def _mesh_candidates(parsed: dict[str, Any], field: str, phrase: str) -> list[str]:
    mesh = parsed.get("mesh")
    values: list[Any] = []
    if isinstance(mesh, dict):
        raw = mesh.get(field) or []
        values = raw if isinstance(raw, list) else [raw]
    candidates = [_clean_query_term(item) for item in values]
    if phrase:
        candidates.append(phrase)
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate.lower() not in {item.lower() for item in unique}:
            unique.append(candidate)
    return unique


_GENERIC_MESH_HEADINGS = {
    "adolescent",
    "adult",
    "aged",
    "aged, 80 and over",
    "child",
    "female",
    "humans",
    "infant",
    "male",
    "middle aged",
    "young adult",
}


def _field_group(phrase: str, candidates: list[str], *, use_mesh: bool) -> str:
    parts: list[str] = []
    if use_mesh:
        for candidate in candidates:
            heading = resolve_mesh_heading(candidate)
            if not heading or heading.lower() in _GENERIC_MESH_HEADINGS:
                continue
            for tag in ("Mesh", "Pharmacological Action"):
                part = f'"{heading}"[{tag}]'
                if part not in parts:
                    parts.append(part)
    if phrase:
        parts.append(f'"{phrase}"[tiab]')
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", phrase.lower())
            if len(token) > 3 and token not in _RELEVANCE_STOPWORDS
        ]
        if len(tokens) >= 2:
            parts.append("(" + " AND ".join(f"{token}[tiab]" for token in tokens) + ")")
        if len(tokens) >= 3:
            parts.append(f"({tokens[0]}[tiab] AND {tokens[-1]}[tiab])")
        for size in (2, 3):
            if len(tokens) < size:
                continue
            for start in range(len(tokens) - size + 1):
                window = " ".join(tokens[start : start + size])
                part = f'"{window}"[tiab]'
                if part not in parts and part != f'"{phrase}"[tiab]':
                    parts.append(part)
    if not parts:
        return ""
    return "(" + " OR ".join(parts) + ")"


def build_pico_boolean_query(
    population: str,
    intervention: str,
    comparison: str,
    outcome: str,
    mesh: dict[str, list[str]],
    *,
    use_mesh: bool,
) -> str:
    """Build a parenthesized PubMed query from PICO fields. The model does not write the Boolean syntax."""
    groups: list[str] = []
    for field, phrase in (
        ("population", population),
        ("intervention", intervention),
        ("comparison", comparison),
        ("outcome", outcome),
    ):
        group = _field_group(phrase, mesh.get(field) or [], use_mesh=use_mesh)
        if group:
            groups.append(group)
    if not groups:
        return ""
    return "(" + " AND ".join(groups) + ")"


def _coerce_query_plan(question: str, parsed: dict[str, Any] | None) -> dict[str, Any]:
    """Build the PubMed query from PICO fields. A missing plan falls back to keywords."""
    keyword_query = _normalize_pubmed_query(question) or question.strip()
    rule_based = {
        "population": "",
        "intervention": "",
        "comparison": "",
        "outcome": "",
        "pubmed_query": keyword_query,
        "fallback_query": question.strip(),
        "source": "rule_based",
    }
    if not parsed:
        return rule_based
    population = _clean_query_term(parsed.get("population"))
    intervention = _clean_query_term(parsed.get("intervention"))
    comparison = _clean_query_term(parsed.get("comparison"))
    outcome = _clean_query_term(parsed.get("outcome"))
    if not any((population, intervention, comparison, outcome)):
        return rule_based
    mesh = {
        field: _mesh_candidates(parsed, field, phrase)
        for field, phrase in (
            ("population", population),
            ("intervention", intervention),
            ("comparison", comparison),
            ("outcome", outcome),
        )
    }
    pubmed_query = build_pico_boolean_query(
        population, intervention, comparison, outcome, mesh, use_mesh=True
    )
    fallback_query = build_pico_boolean_query(
        population, intervention, comparison, outcome, mesh, use_mesh=False
    )
    return {
        "population": population,
        "intervention": intervention,
        "comparison": comparison,
        "outcome": outcome,
        "pubmed_query": pubmed_query or keyword_query,
        "fallback_query": fallback_query or keyword_query,
        "source": "pico",
    }


def study_design_score(article: dict[str, Any]) -> tuple[float, str]:
    """Score study design from PubMed publication-type metadata, not the abstract text."""
    publication_types = list(article.get("publication_types") or [])
    if not publication_types and article.get("study_type"):
        publication_types = [str(article["study_type"])]
    typed = " ".join(publication_types).lower()
    for score, label, keys in _DESIGN_TIERS:
        if any(key in typed for key in keys):
            return score, label
    return 1.5, "unspecified design"


def relevance_score(question: str, article: dict[str, Any]) -> float:
    terms = {
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if len(token) > 3 and token not in _RELEVANCE_STOPWORDS
    }
    if not terms:
        return 0.0
    document = set(re.findall(r"[a-z0-9]+", f"{article.get('title') or ''} {article.get('abstract') or ''}".lower()))
    return len(terms & document) / len(terms)


def recency_score(article: dict[str, Any], now_year: int | None = None) -> float:
    year_match = re.search(r"(19|20)\d{2}", str(article.get("year") or ""))
    if not year_match:
        return 0.4
    age = max(0, (now_year or date.today().year) - int(year_match.group(0)))
    return max(0.0, round(2.0 - age * 0.08, 3))


def _term_in_document(term: str, document: set[str]) -> bool:
    if term in document:
        return True
    for token in document:
        shorter, longer = (term, token) if len(term) <= len(token) else (token, term)
        if len(shorter) >= 4 and longer.startswith(shorter):
            return True
    return False


def pico_alignment(article: dict[str, Any], plan: dict[str, Any] | None) -> tuple[bool | None, str]:
    """Score whether population, intervention, and comparator match the planned PICO.

    None means the plan did not name those elements, so ranking stays design-based.
    """
    if not isinstance(plan, dict):
        return None, ""
    fields = (
        ("population", plan.get("population")),
        ("intervention", plan.get("intervention")),
        ("comparator", plan.get("comparison")),
    )
    document = set(
        re.findall(
            r"[a-z0-9]+",
            f"{article.get('title') or ''} {article.get('abstract') or ''}".lower(),
        )
    )
    considered: list[str] = []
    matched: list[str] = []
    for name, value in fields:
        terms = {
            token
            for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
            if len(token) > 3 and token not in _RELEVANCE_STOPWORDS
        }
        if not terms:
            continue
        considered.append(name)
        hits = sum(1 for term in terms if _term_in_document(term, document))
        if hits / len(terms) >= 0.5:
            matched.append(name)
    if not considered:
        return None, ""
    required = min(2, len(considered))
    on_topic = len(matched) >= required and ("intervention" not in considered or "intervention" in matched)
    detail = f"pico matched {', '.join(matched) or 'none'} of {', '.join(considered)}"
    return on_topic, detail


def rank_articles(
    articles: list[dict[str, Any]],
    question: str,
    now_year: int | None = None,
    plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for article in articles:
        design, design_label = study_design_score(article)
        relevance = relevance_score(question, article)
        recency = recency_score(article, now_year=now_year)
        on_topic, pico_detail = pico_alignment(article, plan)
        # An off-topic review keeps no design bonus, so an on-topic trial ranks above it.
        design_weight = 0.0 if on_topic is False else design
        score = round(design_weight * 2 + relevance * 2 + recency, 3)
        if on_topic is True:
            topic_label = "on-topic"
        elif on_topic is False:
            topic_label = "off-topic"
        else:
            topic_label = "no PICO plan"
        reason = f"design={design_label} ({design}), relevance={relevance:.2f}, recency={recency:.2f}, {topic_label}"
        if pico_detail:
            reason = f"{reason}; {pico_detail}"
        updated = dict(article)
        updated["rank_score"] = score
        updated["rank_reason"] = reason
        if isinstance(article.get("relevance_llm_score"), int):
            updated["relevance_llm_score"] = int(article["relevance_llm_score"])
            updated["rank_reason"] = f"llm_relevance={updated['relevance_llm_score']}, {reason}"
        ranked.append(updated)
    if any(isinstance(item.get("relevance_llm_score"), int) for item in ranked):
        ranked.sort(
            key=lambda item: (
                int(item["relevance_llm_score"]) if isinstance(item.get("relevance_llm_score"), int) else -1,
                study_design_score(item)[0],
                item["rank_score"],
            ),
            reverse=True,
        )
    else:
        ranked.sort(key=lambda item: item["rank_score"], reverse=True)
    return ranked


def assign_citation_indexes(
    primary: list[dict[str, Any]],
    comparison: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Number primary articles from [1], then continue that sequence for comparison articles."""
    primary_out: list[dict[str, Any]] = []
    for offset, article in enumerate(primary, start=1):
        updated = dict(article)
        updated["citation_index"] = offset
        primary_out.append(updated)
    comparison_out: list[dict[str, Any]] = []
    start = len(primary_out) + 1
    for offset, article in enumerate(comparison or []):
        updated = dict(article)
        updated["citation_index"] = start + offset
        comparison_out.append(updated)
    return primary_out, comparison_out


def citation_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text or ""):
        for part in group.split(","):
            numbers.append(int(part.strip()))
    return numbers


def _strip_invalid_citations(answer: str, valid_indexes: set[int]) -> tuple[str, list[int]]:
    invalid: list[int] = []
    sentences = re.split(r"(?<=[.!?])\s+", answer or "")
    kept_sentences: list[str] = []

    def replace(match: re.Match[str]) -> str:
        kept: list[str] = []
        for part in match.group(1).split(","):
            number = int(part.strip())
            if number in valid_indexes:
                kept.append(str(number))
            else:
                invalid.append(number)
        return f"[{', '.join(kept)}]" if kept else ""

    for sentence in sentences:
        numbers = citation_numbers(sentence)
        if numbers and not any(number in valid_indexes for number in numbers):
            invalid.extend(number for number in numbers if number not in valid_indexes)
            continue
        cleaned = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", replace, sentence).strip()
        if cleaned:
            kept_sentences.append(cleaned)
    return re.sub(r"\s{2,}", " ", " ".join(kept_sentences)).strip(), invalid


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


def _article_index(article: dict[str, Any], offset: int, start_index: int) -> int:
    stored = article.get("citation_index")
    if isinstance(stored, int) and stored > 0:
        return stored
    return start_index + offset


def _format_articles(articles: list[dict[str, Any]], start_index: int = 1) -> str:
    if not articles:
        return "No articles retrieved."
    lines: list[str] = []
    for offset, article in enumerate(articles):
        index = _article_index(article, offset, start_index)
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


def _prepare_reference_payload(
    articles: list[dict[str, Any]],
    summaries: dict[str, str],
    question: str,
    answer: str,
    start_index: int = 1,
    evidence: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    evidence = evidence or {}
    references: list[dict[str, Any]] = []
    for offset, article in enumerate(articles):
        highlights = extract_highlight_sentences(article.get("abstract", ""), question, answer)
        references.append(
            {
                "index": _article_index(article, offset, start_index),
                "title": article.get("title", ""),
                "authors": article.get("authors", []),
                "link": article.get("link", ""),
                "pmid": article.get("pmid", ""),
                "journal": article.get("journal", ""),
                "year": article.get("year", ""),
                "study_type": article.get("study_type", ""),
                "summary": summaries.get(article.get("pmid", ""), ""),
                "population": (evidence.get(article.get("pmid", "")) or {}).get("population", ""),
                "intervention": (evidence.get(article.get("pmid", "")) or {}).get("intervention", ""),
                "comparator": (evidence.get(article.get("pmid", "")) or {}).get("comparator", ""),
                "outcome": (evidence.get(article.get("pmid", "")) or {}).get("outcome", ""),
                "effect_direction": (evidence.get(article.get("pmid", "")) or {}).get("effect_direction", ""),
                "significance": (evidence.get(article.get("pmid", "")) or {}).get("significance", ""),
                "highlights": highlights,
                "abstract": article.get("abstract", ""),
                "rank_score": article.get("rank_score"),
                "rank_reason": article.get("rank_reason", ""),
            }
        )
    return references


def _query_candidates(plan: dict[str, Any] | None) -> list[str]:
    if not plan:
        return []
    return [str(plan.get("pubmed_query") or ""), str(plan.get("fallback_query") or "")]


def _plan_one(question: str) -> dict[str, Any]:
    prompt_question = question.strip()
    if not prompt_question:
        return _coerce_query_plan(question, None)
    try:
        system_prompt = """
You turn a clinical question into PICO concepts for a PubMed search.
Return valid JSON only. Do not write Boolean operators, parentheses, or field tags.
{
  "population": "who the question is about",
  "intervention": "the treatment, exposure, or test",
  "comparison": "the comparator, or empty if none",
  "outcome": "the outcome of interest",
  "mesh": {
    "population": ["possible official MeSH heading"],
    "intervention": ["possible official MeSH heading"],
    "comparison": ["possible official MeSH heading"],
    "outcome": ["possible official MeSH heading"]
  }
}
Use only concepts present in the question. Do not add unrelated drugs or diseases.
""".strip()
        response = _invoke_llm([SystemMessage(content=system_prompt), HumanMessage(content=prompt_question)])
        return _coerce_query_plan(prompt_question, _safe_json_loads(getattr(response, "content", "")))
    except Exception:
        return _coerce_query_plan(prompt_question, None)


def _plan_query(state: ResearchState) -> ResearchState:
    question = state["question"]
    comparison_question = state.get("comparison_question")
    if state.get("mode") == "compare" and comparison_question:
        with ThreadPoolExecutor(max_workers=2) as executor:
            primary_future = executor.submit(_plan_one, question)
            comparison_future = executor.submit(_plan_one, comparison_question)
            return {
                "query_plan": primary_future.result(),
                "comparison_query_plan": comparison_future.result(),
            }
    return {"query_plan": _plan_one(question)}


def _format_summaries(articles: list[dict[str, Any]], summaries_by_pmid: dict[str, dict[str, Any]]) -> str:
    if not articles:
        return "No article summaries."
    lines: list[str] = []
    for offset, article in enumerate(articles):
        index = _article_index(article, offset, 1)
        pmid = str(article.get("pmid") or "")
        record = summaries_by_pmid.get(pmid) or {}
        lines.append(
            "\n".join(
                [
                    f"[{index}] PMID {pmid or 'NA'}",
                    f"Study type: {article.get('study_type') or 'Unknown'}",
                    f"Population: {record.get('population') or 'not reported'}",
                    f"Intervention: {record.get('intervention') or 'not reported'}",
                    f"Comparator: {record.get('comparator') or 'not reported'}",
                    f"Outcome: {record.get('outcome') or 'not reported'}",
                    f"Effect direction: {record.get('effect_direction') or 'unclear'}",
                    f"Significance: {record.get('significance') or 'not reported'}",
                ]
            )
        )
    return "\n\n".join(lines)


def _articles_in_citation_order(state: ResearchState) -> list[dict[str, Any]]:
    combined = list(state.get("articles") or []) + list(state.get("comparison_articles") or [])
    return sorted(combined, key=lambda article: int(article.get("citation_index") or 0))


def _select_for_summary(primary: list[dict[str, Any]], comparison: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    cap = max(1, cap)
    if not comparison:
        return primary[:cap]
    primary_slots = min(len(primary), max(1, cap // 2))
    comparison_slots = min(len(comparison), max(0, cap - primary_slots))
    leftover = cap - primary_slots - comparison_slots
    if leftover and len(primary) > primary_slots:
        primary_slots = min(len(primary), primary_slots + leftover)
        leftover = cap - primary_slots - comparison_slots
    if leftover and len(comparison) > comparison_slots:
        comparison_slots = min(len(comparison), comparison_slots + leftover)
    return primary[:primary_slots] + comparison[:comparison_slots]


def _blank_summary(pmid: str) -> dict[str, str]:
    return {
        "pmid": pmid,
        "population": "",
        "intervention": "",
        "comparator": "",
        "outcome": "",
        "effect_direction": "unclear",
        "significance": "not reported",
        "summary": "",
    }


def _summary_sentence(record: dict[str, Any]) -> str:
    return (
        f"Population: {record.get('population') or 'not reported'}. "
        f"Intervention: {record.get('intervention') or 'not reported'}. "
        f"Comparator: {record.get('comparator') or 'not reported'}. "
        f"Outcome: {record.get('outcome') or 'not reported'}. "
        f"Effect direction: {record.get('effect_direction') or 'unclear'}. "
        f"Significance: {record.get('significance') or 'not reported'}."
    )


def _normalize_effect(value: Any) -> str:
    text = str(value or "").strip().lower()
    allowed = {"benefit", "harm", "no difference", "mixed", "unclear"}
    return text if text in allowed else "unclear"


def _normalize_significance(value: Any) -> str:
    text = str(value or "").strip().lower()
    allowed = {"significant", "not significant", "not reported"}
    return text if text in allowed else "not reported"


def _normalize_quote_text(text: str) -> str:
    """Fold case, whitespace, dashes, punctuation, and a few spelling variants."""
    folded = (text or "").translate(
        str.maketrans(
            {
                "\u00ad": "",
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
            }
        )
    )
    folded = folded.lower()
    for british, american in (
        ("haemorrhage", "hemorrhage"),
        ("anaemia", "anemia"),
        ("oedema", "edema"),
    ):
        folded = folded.replace(british, american)
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def quote_in_abstract(quote: str, abstract: str) -> bool:
    """A supporting quote must be a verbatim stretch of the cited abstract.

    Case, whitespace, punctuation, and British/American spellings such as
    haemorrhage/hemorrhage are ignored so a copied sentence still matches.
    A quote joined with "..." is supported only when every segment appears
    in that same abstract.
    """
    normalized_abstract = _normalize_quote_text(abstract)
    if not normalized_abstract:
        return False
    segments = [
        _normalize_quote_text(part)
        for part in re.split(r"\.{3,}|…", quote or "")
    ]
    segments = [segment for segment in segments if segment]
    if not segments or len(" ".join(segments)) < 20:
        return False
    return all(segment in normalized_abstract for segment in segments)


def _summary_cache_key(question: str, pmid: str) -> str:
    raw = f"{question.strip().lower()}::{pmid}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_summary_cache() -> dict[str, Any]:
    if not _SUMMARY_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(_SUMMARY_CACHE_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_summary_cache(cache: dict[str, Any]) -> None:
    try:
        _SUMMARY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SUMMARY_CACHE_PATH.write_text(json.dumps(cache))
    except OSError:
        return


def _summarize_one(question: str, article: dict[str, Any]) -> dict[str, str]:
    pmid = str(article.get("pmid") or "")
    abstract = str(article.get("abstract") or "").strip()
    cache_key = _summary_cache_key(question, pmid)
    with _SUMMARY_CACHE_LOCK:
        cached = _read_summary_cache().get(cache_key)
    if isinstance(cached, dict) and cached.get("pmid"):
        return cached
    record = _blank_summary(pmid)
    if not abstract:
        record["summary"] = _summary_sentence(record)
        return record
    parsed: dict[str, Any] = {}
    try:
        system_prompt = """
Extract structured facts from one PubMed abstract.
Use only facts stated in the abstract. If a field is absent, use an empty string.
effect_direction must be one of: benefit, harm, no difference, mixed, unclear.
significance must be one of: significant, not significant, not reported.
Judge effect_direction for the question's main outcome, not a surrogate the question did not ask about.
Return valid JSON:
{
  "pmid": "the given pmid",
  "population": "",
  "intervention": "",
  "comparator": "",
  "outcome": "",
  "effect_direction": "unclear",
  "significance": "not reported"
}
""".strip()
        user_prompt = (
            f"Question: {question}\n"
            f"PMID: {pmid}\n"
            f"Title: {article.get('title', '')}\n"
            f"Abstract: {abstract}"
        )
        response = _invoke_llm([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        parsed = _safe_json_loads(getattr(response, "content", "")) or {}
        record["population"] = str(parsed.get("population") or "").strip()
        record["intervention"] = str(parsed.get("intervention") or "").strip()
        record["comparator"] = str(parsed.get("comparator") or "").strip()
        record["outcome"] = str(parsed.get("outcome") or "").strip()
        record["effect_direction"] = _normalize_effect(parsed.get("effect_direction"))
        record["significance"] = _normalize_significance(parsed.get("significance"))
    except Exception:
        parsed = {}
    record["summary"] = _summary_sentence(record)
    if parsed:
        with _SUMMARY_CACHE_LOCK:
            cache = _read_summary_cache()
            cache[cache_key] = record
            _write_summary_cache(cache)
    return record


def _label_for_confidence_score(score: int) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Moderate"
    return "Low"


def answered_confidence(articles: list[dict[str, Any]], support_rate: float | None) -> dict[str, Any]:
    """One score from kept-article relevance and claim support. The label is that score."""
    relevance_values = [
        min(2, max(0, int(article["relevance_llm_score"])))
        for article in articles
        if isinstance(article.get("relevance_llm_score"), int)
    ]
    relevance = sum(relevance_values) / (2 * len(relevance_values)) if relevance_values else 0.0
    try:
        support = float(support_rate) if support_rate is not None else 0.0
    except (TypeError, ValueError):
        support = 0.0
    support = min(1.0, max(0.0, support))
    score = int(round(relevance * support * 100))
    return {
        "score": score,
        "label": _label_for_confidence_score(score),
        "rationale": f"Kept-article relevance and claim support give a confidence of {score}/100.",
    }


def abstain_confidence(retrieved_count: int) -> dict[str, Any]:
    count = max(0, int(retrieved_count))
    return {
        "score": None,
        "label": None,
        "review_status": "needs expert review",
        "rationale": (
            f"{count} abstracts were retrieved, but fewer than two were directly on topic, "
            "so this question needs expert review."
        ),
    }


def _apply_citation_confidence(confidence: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    failures = len(audit.get("invalid_citations") or []) + len(audit.get("unsupported_claims") or [])
    zero_support = audit.get("support_rate") == 0
    if not failures and not zero_support:
        return confidence
    updated = dict(confidence)
    if failures:
        score = max(0.05, round(float(updated.get("score") or 0) - 0.1 * failures, 2))
        if score >= 0.75:
            label = "High"
        elif score >= 0.5:
            label = "Moderate"
        else:
            label = "Low"
        updated["score"] = score
        updated["label"] = label
        updated["citation_penalty"] = failures
        rationale = str(updated.get("rationale") or "").strip()
        penalty_note = f" Citation check lowered confidence after {failures} unsupported or invalid citation(s)."
        updated["rationale"] = f"{rationale}{penalty_note}".strip()
    if zero_support:
        updated["score"] = min(float(updated.get("score") or 0), 0.49)
        updated["label"] = "Low"
        rationale = str(updated.get("rationale") or "").strip()
        cap_note = " Claim support was zero, so confidence is capped at Low."
        if cap_note.strip() not in rationale:
            updated["rationale"] = f"{rationale}{cap_note}".strip()
    return updated


def _retrieve_literature(state: ResearchState) -> ResearchState:
    filters = state.get("filters", {})
    question = state["question"]
    comparison_question = state.get("comparison_question")
    include_trials = bool(state.get("include_trials", True))
    pool = RETRIEVAL_CANDIDATES
    primary_candidates = _query_candidates(state.get("query_plan"))
    comparison_candidates = _query_candidates(state.get("comparison_query_plan"))

    with ThreadPoolExecutor(max_workers=4) as executor:
        articles_future = executor.submit(
            search_pubmed,
            question,
            pool,
            filters.get("year_from"),
            filters.get("year_to"),
            filters.get("study_type"),
            primary_candidates,
        )
        cached_future = executor.submit(search_cached_literature, question, min(int(state.get("max_results") or 3), 3))
        trials_future = executor.submit(search_clinical_trials, question, 3) if include_trials else None

        comparison_articles_future = None
        comparison_trials_future = None
        if state.get("mode") == "compare" and comparison_question:
            comparison_articles_future = executor.submit(
                search_pubmed,
                comparison_question,
                pool,
                filters.get("year_from"),
                filters.get("year_to"),
                filters.get("study_type"),
                comparison_candidates,
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


def _parse_relevance_payload(parsed: dict[str, Any] | None, known: set[str]) -> dict[str, int]:
    """Read a PMID-keyed score object. A list of {pmid, score} objects is accepted too."""
    if not isinstance(parsed, dict):
        return {}
    raw = parsed.get("scores", parsed)
    pairs: list[tuple[Any, Any]] = []
    if isinstance(raw, dict):
        pairs = [(key, value) for key, value in raw.items() if key != "scores"]
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                pairs.append((item.get("pmid"), item.get("score")))
    scores: dict[str, int] = {}
    for pmid, score in pairs:
        key = str(pmid or "").strip()
        if key not in known:
            continue
        try:
            value = int(score)
        except (TypeError, ValueError):
            continue
        scores[key] = min(2, max(0, value))
    return scores


def _score_relevance_batch(question: str, articles: list[dict[str, Any]]) -> dict[str, int]:
    """Ask for one JSON object keyed by PMID and keep only scores for this batch."""
    if not articles:
        return {}
    known = {str(article.get("pmid") or "").strip() for article in articles if str(article.get("pmid") or "").strip()}
    blocks: list[str] = []
    for article in articles:
        abstract = re.sub(r"\s+", " ", str(article.get("abstract") or ""))
        if len(abstract) > 700:
            abstract = f"{abstract[:400]} ... {abstract[-250:]}"
        blocks.append(f"PMID {article.get('pmid')}\nTitle: {article.get('title', '')}\nAbstract: {abstract}")
    system_prompt = """
Score how well each article matches the clinical question.
Use the question's population, intervention, comparator, and outcome.
2 = the article studies that population, intervention, comparator, and outcome.
1 = it matches only some of those elements.
0 = it answers a different question.
Return one JSON object and nothing else. Keys are the PMIDs. Values are 0, 1, or 2.
Include every PMID from this batch.
Example: {"7968073": 2, "12114036": 0}
""".strip()
    user_prompt = f"Question: {question}\n\nArticles:\n\n" + "\n\n".join(blocks)
    response = _invoke_llm(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        reasoning_effort="low" if _model_supports_low_reasoning() else None,
    )
    text = _response_text(response)
    parsed = _safe_json_loads(text)
    if not isinstance(parsed, dict):
        parsed = _safe_json_loads(text.rstrip() + "}")
    return _parse_relevance_payload(parsed if isinstance(parsed, dict) else None, known)


def _batch_relevance_scores(question: str, articles: list[dict[str, Any]]) -> dict[str, int]:
    """Score candidates in parallel batches of 6, then retry any PMID that came back without a score."""
    pending = [article for article in articles if str(article.get("pmid") or "").strip()]
    if not pending:
        return {}
    batch_size = 6
    batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]

    def _run(batch: list[dict[str, Any]]) -> dict[str, int]:
        try:
            return _score_relevance_batch(question, batch)
        except Exception as exc:
            logger.warning("Relevance batch failed: %s", exc)
            return {}

    scores: dict[str, int] = {}
    workers = min(4, len(batches))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch_scores in executor.map(_run, batches):
            scores.update(batch_scores)

    missing = [article for article in pending if str(article.get("pmid")) not in scores]
    if missing:
        retry_batches = [missing[start : start + batch_size] for start in range(0, len(missing), batch_size)]
        with ThreadPoolExecutor(max_workers=min(4, len(retry_batches))) as executor:
            for batch_scores in executor.map(_run, retry_batches):
                scores.update(batch_scores)
        still_missing = [str(article.get("pmid")) for article in missing if str(article.get("pmid")) not in scores]
        if still_missing:
            logger.warning(
                "Relevance score missing after retry for %s of %s PMIDs: %s",
                len(still_missing),
                len(pending),
                ", ".join(still_missing),
            )
    return scores


def _attach_relevance_scores(question: str, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not articles or all(isinstance(article.get("relevance_llm_score"), int) for article in articles):
        return articles
    try:
        scores = _batch_relevance_scores(question, articles)
    except Exception as exc:
        logger.warning("Relevance scoring failed: %s", exc)
        scores = {}
    scored: list[dict[str, Any]] = []
    for article in articles:
        updated = dict(article)
        pmid = str(updated.get("pmid") or "").strip()
        if pmid in scores:
            updated["relevance_llm_score"] = scores[pmid]
        scored.append(updated)
    covered = sum(1 for article in scored if isinstance(article.get("relevance_llm_score"), int))
    print(f"relevance scored {covered}/{len(articles)}", flush=True)
    return scored


def _top_articles_match_pico(articles: list[dict[str, Any]], plan: dict[str, Any] | None = None) -> bool:
    """Require two of the top four articles to score 2. Otherwise the pipeline abstains."""
    del plan
    top = list(articles or [])[:4]
    strong = sum(1 for article in top if article.get("relevance_llm_score") == 2)
    return strong >= 2


def _abstain_synthesis() -> dict[str, Any]:
    answer = "The retrieved evidence does not address the question. This answer needs expert review."
    return {
        "answer": answer,
        "plain_language_summary": answer,
        "needs_expert_review": True,
        "citation_audit": {
            "validity_rate": None,
            "citation_count": 0,
            "valid_citation_count": 0,
            "invalid_citations": [],
            "support_rate": None,
            "claim_count": 0,
            "supported_claim_count": 0,
            "unsupported_claims": [],
            "removed_or_changed": [],
        },
    }


def _rank_evidence(state: ResearchState) -> ResearchState:
    keep = min(int(state.get("max_results") or MAX_ARTICLE_SUMMARIES), MAX_ARTICLE_SUMMARIES)
    keep = max(1, keep)
    primary_scored = _attach_relevance_scores(state["question"], state.get("articles") or [])
    comparison_question = state.get("comparison_question") or state["question"]
    comparison_scored = _attach_relevance_scores(comparison_question, state.get("comparison_articles") or [])
    primary = rank_articles(
        primary_scored,
        state["question"],
        plan=state.get("query_plan"),
    )[:keep]
    comparison = rank_articles(
        comparison_scored,
        comparison_question,
        plan=state.get("comparison_query_plan") or state.get("query_plan"),
    )
    if state.get("mode") == "compare":
        comparison = comparison[:keep]
    else:
        comparison = []
    primary, comparison = assign_citation_indexes(primary, comparison)
    relevance_scored = sum(1 for article in primary_scored if isinstance(article.get("relevance_llm_score"), int))
    relevance_total = len(primary_scored)
    if _top_articles_match_pico(primary, state.get("query_plan")):
        return {
            "articles": primary,
            "comparison_articles": comparison,
            "needs_expert_review": False,
            "relevance_scored": relevance_scored,
            "relevance_total": relevance_total,
        }
    return {
        "articles": primary,
        "comparison_articles": comparison,
        "needs_expert_review": True,
        "relevance_scored": relevance_scored,
        "relevance_total": relevance_total,
        "synthesis": _abstain_synthesis(),
    }


def _route_after_rank(state: ResearchState) -> str:
    if state.get("needs_expert_review"):
        return "abstain"
    return "summarize"


def _summarize_articles(state: ResearchState) -> ResearchState:
    question = state["question"]
    comparison_question = state.get("comparison_question") or question
    primary = state.get("articles") or []
    comparison = state.get("comparison_articles") or []
    selected = _select_for_summary(primary, comparison, MAX_ARTICLE_SUMMARIES)
    if not selected:
        return {"article_summaries": []}

    comparison_ids = {id(article) for article in comparison}

    def summarize(article: dict[str, Any]) -> dict[str, Any]:
        prompt_question = comparison_question if id(article) in comparison_ids else question
        return _summarize_one(prompt_question, article)

    workers = min(4, len(selected))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        summaries = list(executor.map(summarize, selected))
    return {"article_summaries": summaries}


def _synthesize(state: ResearchState) -> ResearchState:
    question = state["question"]
    comparison_question = state.get("comparison_question")
    mode = state.get("mode", "standard")
    history = _stringify_history(state.get("history", []))
    articles = state.get("articles", [])
    comparison_articles = state.get("comparison_articles", [])
    trials = state.get("trials", [])
    comparison_trials = state.get("comparison_trials", [])
    summaries_by_pmid = {
        str(item.get("pmid") or "").strip(): item
        for item in state.get("article_summaries") or []
        if isinstance(item, dict) and str(item.get("pmid") or "").strip()
    }

    system_prompt = """
You are a careful medical literature research assistant.
Rely only on the provided structured study summaries. Do not use outside knowledge.
Do not invent results, effect sizes, or recommendations.
Cite a summary only with its printed number, such as [1].
Every sentence that states a result must include a citation.
State the conclusion supported by most of the higher-quality evidence.
Weigh study design and study size: a systematic review or large randomized trial outweighs a smaller or weaker study.
After that conclusion, note conflicting results as a caveat.
Do not conclude that the evidence conflicts unless the strongest studies disagree with each other.
Count how many summarized studies support the conclusion, how many oppose it, and how many are unclear.

Return valid JSON with this shape:
{
  "answer": "Main answer with inline numeric citations like [1] and [2]. Include the study counts.",
  "plain_language_summary": "Short simpler explanation for a non-specialist reader.",
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

Primary article summaries:
{_format_summaries(articles, summaries_by_pmid)}

Primary ongoing trials:
{_format_trials(trials)}

Comparison article summaries:
{_format_summaries(comparison_articles, summaries_by_pmid)}

Comparison ongoing trials:
{_format_trials(comparison_trials)}

Write a head-to-head comparison that:
- uses only the structured summaries above
- states how many studies on each side support benefit, harm, no difference, or an unclear effect
- calls out conflicting effect directions when they are present
- cites PubMed summaries with one continuous numbering sequence: primary starts at [1], and comparison numbers continue after the last primary number instead of restarting at [1]

Return JSON only.
""".strip()
    else:
        user_prompt = f"""
Conversation history:
{history}

Clinical question:
{question}

Article summaries:
{_format_summaries(articles, summaries_by_pmid)}

ClinicalTrials.gov records:
{_format_trials(trials)}

Write a concise evidence synthesis that:
- uses only the structured summaries above
- answers the question in plain language
- says how many studies support the conclusion and how many do not
- calls out conflicting evidence when effect directions disagree
- notes relevant ongoing trials if present
- uses inline citations that match the summary numbers above

Return JSON only.
""".strip()

    response = _invoke_llm([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    parsed = _safe_json_loads(getattr(response, "content", ""))
    if not parsed:
        parsed = {
            "answer": getattr(response, "content", "").strip() or "No answer generated.",
            "plain_language_summary": "",
            "confidence_explanation": "",
        }
    parsed["article_summaries"] = list(summaries_by_pmid.values())
    return {"synthesis": parsed}


def _invoke_verifier(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Ask the fact-checker for JSON, then retry once if the reply has no claim list."""

    def _read(prompt: str) -> dict[str, Any] | None:
        response = _invoke_llm(
            [SystemMessage(content=prompt), HumanMessage(content=user_prompt)],
            reasoning_effort="low" if _model_supports_low_reasoning() else None,
        )
        parsed = _safe_json_loads(getattr(response, "content", ""))
        if isinstance(parsed, dict) and isinstance(parsed.get("claims"), list):
            return parsed
        return None

    parsed = _read(system_prompt)
    if parsed is not None:
        return parsed
    strict_prompt = (
        system_prompt
        + "\n\nYour previous reply was not a JSON object with a claims array. "
        + "Reply with one JSON object only. Do not use markdown code fences. "
        + "The object must include a claims array."
    )
    return _read(strict_prompt) or {}


def _verify_citations(state: ResearchState) -> ResearchState:
    """Check that each [n] citation exists and that the cited abstract supports the sentence.

    Invalid citation numbers are removed even if the model call fails. Support judgments
    fail open: a model or JSON failure keeps the answer after the index check.
    """
    synthesis = dict(state.get("synthesis") or {})
    answer = str(synthesis.get("answer", "")).strip()
    articles = _articles_in_citation_order(state)
    if not answer:
        return {}

    valid_indexes = {
        int(article["citation_index"])
        for article in articles
        if isinstance(article.get("citation_index"), int)
    }
    if not valid_indexes:
        valid_indexes = set(range(1, len(articles) + 1))

    cited = citation_numbers(answer)
    cited_set = set(cited)
    index_invalid = [number for number in cited if number not in valid_indexes]
    index_valid = [number for number in cited if number in valid_indexes]
    validity_rate = 0.0 if not cited else round(len(index_valid) / len(cited), 3)

    cleaned_answer, stripped = _strip_invalid_citations(answer, valid_indexes)
    invalid_citations = list(dict.fromkeys(index_invalid + stripped))
    audit: dict[str, Any] = {
        "validity_rate": validity_rate,
        "citation_count": len(cited),
        "valid_citation_count": len(index_valid),
        "invalid_citations": invalid_citations,
        "support_rate": None,
        "claim_count": 0,
        "supported_claim_count": 0,
        "unsupported_claims": [],
        "removed_or_changed": [],
    }

    if not articles:
        if not cited:
            audit["support_rate"] = 0.0
            audit["claim_count"] = 1
            audit["supported_claim_count"] = 0
            audit["unsupported_claims"] = [
                {
                    "citation": None,
                    "sentence": answer[:300],
                    "supported": False,
                    "quote": "",
                    "reason": "The answer has no [n] citations.",
                }
            ]
        synthesis["answer"] = cleaned_answer or answer
        synthesis["citation_audit"] = audit
        synthesis["verifier_removed_or_changed"] = []
        return {"synthesis": synthesis}

    try:
        system_prompt = """
You are a strict fact-checker reviewing a medical evidence synthesis before it is shown to a user.
You will be given a generated answer with inline citations like [1], [2], and the source
abstracts those citation numbers refer to. Numbers are continuous across primary and comparison articles.

For every sentence that contains a citation:
- supported is true only when that sentence is explicitly supported by the cited abstract.
- quote must be an exact sentence or clause copied from that abstract. Do not paraphrase the quote.
- If you cannot copy a supporting sentence, supported is false and quote is empty.
- If the sentence adds a recommendation, clinical-utility judgment, or conclusion beyond the abstract, supported is false.
- If it calls a study "the only available study" or otherwise describes the evidence base in a way the abstract does not, supported is false.
- If it misstates or overstates the abstract, supported is false.

Then rewrite the answer so unsupported sentences are removed. Do not add new facts or citations.

Return valid JSON:
{
  "verified_answer": "Corrected answer with the same citation style.",
  "claims": [
    {"citation": 1, "sentence": "the cited sentence", "supported": true, "quote": "exact sentence copied from the abstract", "reason": "short reason"}
  ],
  "removed_or_changed": ["One short entry per removed or changed sentence. Empty if none."]
}
""".strip()
        user_prompt = f"""
Generated answer to check:
{cleaned_answer or answer}

Source abstracts. Use the printed citation numbers, which do not restart for comparison articles:
{_format_articles(articles)}

Return JSON only.
""".strip()
        parsed = _invoke_verifier(system_prompt, user_prompt)
        verified = str(parsed.get("verified_answer") or "").strip()
        claims = [item for item in parsed.get("claims") or [] if isinstance(item, dict)]
        abstracts_by_index = {
            int(article["citation_index"]): str(article.get("abstract") or "")
            for article in articles
            if isinstance(article.get("citation_index"), int)
        }
        judged: list[dict[str, Any]] = []
        for item in claims:
            citation = item.get("citation")
            try:
                citation_number = int(citation)
            except (TypeError, ValueError):
                citation_number = None
            quote = str(item.get("quote") or "").strip()
            supported = item.get("supported") is True
            reason = str(item.get("reason") or "").strip()
            abstract = abstracts_by_index.get(citation_number or -1, "")
            if citation_number is None or citation_number not in cited_set:
                supported = False
                reason = "Claim is not tied to a [n] citation in the answer."
            elif supported and not quote_in_abstract(quote, abstract):
                supported = False
                reason = "The claim did not include an exact supporting sentence from the cited abstract."
            judged.append(
                {
                    "citation": citation,
                    "sentence": str(item.get("sentence") or "").strip(),
                    "supported": supported,
                    "quote": quote,
                    "reason": reason,
                }
            )
        unsupported = [item for item in judged if not item["supported"]]
        if judged:
            supported_count = sum(1 for item in judged if item["supported"])
            audit["claim_count"] = len(judged)
            audit["supported_claim_count"] = supported_count
            audit["support_rate"] = round(supported_count / len(judged), 3)
        else:
            audit["claim_count"] = len(cited) if cited else 1
            audit["supported_claim_count"] = 0
            audit["support_rate"] = 0.0
            unsupported = [
                {
                    "citation": None,
                    "sentence": "",
                    "supported": False,
                    "quote": "",
                    "reason": "Verifier returned no claim list.",
                }
            ]
        audit["unsupported_claims"] = unsupported
        audit["removed_or_changed"] = parsed.get("removed_or_changed") or []
        if verified:
            verified, extra_invalid = _strip_invalid_citations(verified, valid_indexes)
            audit["invalid_citations"] = list(dict.fromkeys(invalid_citations + extra_invalid))
            for claim in unsupported:
                sentence = claim["sentence"]
                if sentence and sentence in verified:
                    verified = verified.replace(sentence, "").strip()
            synthesis["answer"] = re.sub(r"\s{2,}", " ", verified).strip() or cleaned_answer
        else:
            synthesis["answer"] = cleaned_answer or answer
    except Exception:
        synthesis["answer"] = cleaned_answer or answer
        if audit["support_rate"] is None:
            audit["claim_count"] = len(cited) if cited else 1
            audit["supported_claim_count"] = 0
            audit["support_rate"] = 0.0
            audit["unsupported_claims"] = [
                {
                    "citation": None,
                    "sentence": "",
                    "supported": False,
                    "quote": "",
                    "reason": "Verifier returned no claim list.",
                }
            ]

    if not cited:
        audit["validity_rate"] = 0.0
        audit["support_rate"] = 0.0
        audit["supported_claim_count"] = 0
        if not audit["claim_count"]:
            audit["claim_count"] = 1
        if not audit["unsupported_claims"]:
            audit["unsupported_claims"] = [
                {
                    "citation": None,
                    "sentence": answer[:300],
                    "supported": False,
                    "quote": "",
                    "reason": "The answer has no [n] citations.",
                }
            ]

    synthesis["citation_audit"] = audit
    synthesis["verifier_removed_or_changed"] = audit["removed_or_changed"]
    return {"synthesis": synthesis}


def _with_node_timing(name: str, fn):
    def wrapped(state: ResearchState) -> ResearchState:
        started = time.perf_counter()
        result = fn(state)
        elapsed = round(time.perf_counter() - started, 2)
        print(f"node {name} {elapsed}s", flush=True)
        timings = dict(state.get("node_timings") or {})
        timings[name] = elapsed
        if isinstance(result, dict):
            result = dict(result)
            result["node_timings"] = timings
        return result

    return wrapped


def _build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("plan_query", _with_node_timing("plan_query", _plan_query))
    graph.add_node("retrieve_literature", _with_node_timing("retrieve_literature", _retrieve_literature))
    graph.add_node("rank_evidence", _with_node_timing("rank_evidence", _rank_evidence))
    graph.add_node("summarize_articles", _with_node_timing("summarize_articles", _summarize_articles))
    graph.add_node("synthesize", _with_node_timing("synthesize", _synthesize))
    graph.add_node("verify_citations", _with_node_timing("verify_citations", _verify_citations))
    graph.add_edge(START, "plan_query")
    graph.add_edge("plan_query", "retrieve_literature")
    graph.add_edge("retrieve_literature", "rank_evidence")
    graph.add_conditional_edges(
        "rank_evidence",
        _route_after_rank,
        {"summarize": "summarize_articles", "abstain": END},
    )
    graph.add_edge("summarize_articles", "synthesize")
    graph.add_edge("synthesize", "verify_citations")
    graph.add_edge("verify_citations", END)
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
    evidence_by_pmid = {
        str(item.get("pmid", "")).strip(): item
        for item in synthesis.get("article_summaries", [])
        if isinstance(item, dict) and str(item.get("pmid", "")).strip()
    }
    reference_summaries = {
        pmid: str(item.get("summary", "")).strip() for pmid, item in evidence_by_pmid.items()
    }
    references = _prepare_reference_payload(
        articles=articles,
        summaries=reference_summaries,
        question=question,
        answer=str(synthesis.get("answer", "")),
        evidence=evidence_by_pmid,
    )
    comparison_articles = result.get("comparison_articles", [])
    citation_audit = synthesis.get("citation_audit") or {
        "validity_rate": None,
        "invalid_citations": [],
        "support_rate": None,
        "unsupported_claims": [],
        "removed_or_changed": synthesis.get("verifier_removed_or_changed") or [],
    }
    if result.get("needs_expert_review"):
        confidence = abstain_confidence(int(result.get("relevance_total") or 0))
    else:
        confidence = answered_confidence(articles, citation_audit.get("support_rate"))
        if synthesis.get("confidence_explanation"):
            confidence["model_explanation"] = str(synthesis.get("confidence_explanation", "")).strip()

    comparison_references: list[dict[str, Any]] = []
    if mode == "compare":
        comparison_references = _prepare_reference_payload(
            articles=comparison_articles,
            summaries=reference_summaries,
            question=comparison_question or "",
            answer=str(synthesis.get("answer", "")),
            start_index=len(articles) + 1,
            evidence=evidence_by_pmid,
        )

    def ranking_rows(items: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
        return [
            {
                "pmid": item.get("pmid", ""),
                "index": item.get("citation_index"),
                "rank_score": item.get("rank_score"),
                "rank_reason": item.get("rank_reason", ""),
                "side": side,
            }
            for item in items
        ]

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
        "query_plan": result.get("query_plan") or {},
        "comparison_query_plan": result.get("comparison_query_plan") or {},
        "needs_expert_review": bool(result.get("needs_expert_review")),
        "relevance_scored": int(result.get("relevance_scored") or 0),
        "relevance_total": int(result.get("relevance_total") or 0),
        "rankings": ranking_rows(articles, "primary") + ranking_rows(comparison_articles, "comparison"),
        "citation_audit": citation_audit,
    }


def simplify_text(question: str, answer: str) -> str:
    system_prompt = "Rewrite medical evidence summaries at an accessible reading level without changing the meaning."
    user_prompt = f"Question: {question}\n\nOriginal answer:\n{answer}\n\nRewrite this in clear plain language at roughly an 8th-grade reading level."
    response = _invoke_llm([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return getattr(response, "content", "").strip()
