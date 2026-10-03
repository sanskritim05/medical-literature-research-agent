#!/usr/bin/env python3
"""Live citation-faithfulness evaluation against PubMedQA pilot PMIDs.

Uses the real live Groq generator/verifier from agent.py (requested:
llama-3.1-8b-instant; falls back to the closest available Groq chat model when
that ID is retired), and an independent larger Groq judge so the judge is not
the same model that produced the answers.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agent import (
    ResearchState,
    _format_articles,
    _safe_json_loads,
    _synthesize,
    _verify_citations,
)
from pubmed_tool import assess_confidence


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "pubmedqa" / "data" / "ori_pqal.json"
OUT_PATH = ROOT / "eval_results_live.json"

PILOT_PMIDS = [
    "10135926",
    "10456814",
    "14551704",
    "15208005",
    "15369037",
    "18783922",
    "19854401",
    "20537205",
    "20605051",
    "20608141",
    "21394762",
    "22428608",
    "22564465",
    "24153338",
    "24160268",
    "25480629",
    "27448572",
    "8738894",
    "8847047",
    "9278754",
]

GENERATOR_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _ensure_dataset() -> dict[str, Any]:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
        print(f"Downloading {url} ...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        DATA_PATH.write_bytes(response.content)
    with DATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _contexts_to_abstract(item: dict[str, Any]) -> str:
    contexts = item.get("CONTEXTS") or []
    labels = item.get("LABELS") or []
    parts: list[str] = []
    for index, section in enumerate(contexts):
        label = labels[index].strip() if index < len(labels) and labels[index] else ""
        text = str(section).strip()
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts).strip()


def _article_from_pubmedqa(pmid: str, item: dict[str, Any], publication_types: list[str]) -> dict[str, Any]:
    """Format a PubMedQA item as this pipeline's PubMed article dict."""
    year = str(item.get("YEAR") or "")
    study_type = publication_types[0] if publication_types else ""
    return {
        "source": "pubmedqa",
        "pmid": pmid,
        "title": f"PubMedQA abstract {pmid}",
        "journal": "PubMedQA",
        "year": year,
        "authors": [],
        "abstract": _contexts_to_abstract(item),
        "publication_types": publication_types,
        "study_type": study_type,
        "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def _fetch_publication_types(pmids: list[str]) -> dict[str, list[str]]:
    """Best-effort live NCBI E-utilities lookup for publication types."""
    result: dict[str, list[str]] = {pmid: [] for pmid in pmids}
    try:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "tool": "medical-literature-research-agent-eval",
            "email": os.getenv("NCBI_EMAIL", "eval@localhost"),
        }
        api_key = os.getenv("NCBI_API_KEY", "").strip()
        if api_key:
            params["api_key"] = api_key
        response = requests.get(EFETCH_URL, params=params, timeout=45)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        for article in root.findall(".//PubmedArticle"):
            pmid = (article.findtext(".//PMID") or "").strip()
            if not pmid:
                continue
            types = []
            for item in article.findall(".//PublicationTypeList/PublicationType"):
                value = "".join(item.itertext()).strip()
                if value:
                    types.append(value)
            result[pmid] = types
        return result
    except Exception as exc:
        print(f"WARNING: could not fetch PubMed publication types via E-utilities ({exc}).")
        print("Using publication_types=[] for all items (same as the pilot fallback).")
        return {pmid: [] for pmid in pmids}


def _build_judge_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for the independent judge.")
    return ChatGroq(groq_api_key=api_key, model_name=JUDGE_MODEL, temperature=0)


def _patch_llm_retries() -> None:
    """Retry Groq 429s for both generator and judge calls."""
    original_invoke = ChatGroq.invoke

    def invoke_with_retries(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        delay = 4.0
        last_exc: Exception | None = None
        for attempt in range(1, 9):
            try:
                return original_invoke(self, *args, **kwargs)
            except Exception as exc:
                last_exc = exc
                text = str(exc).lower()
                if "rate_limit" in text or "429" in text or "tokens per minute" in text:
                    wait = delay
                    match = re.search(r"try again in ([0-9.]+)s", text)
                    if match:
                        wait = max(delay, float(match.group(1)) + 0.75)
                    model_name = getattr(self, "model_name", "groq")
                    print(f"         rate-limited ({model_name}); sleeping {wait:.1f}s (attempt {attempt}/8)")
                    time.sleep(wait)
                    delay = min(delay * 1.7, 60.0)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    ChatGroq.invoke = invoke_with_retries  # type: ignore[method-assign]


def _judge_faithfulness(answer: str, abstract: str, judge_llm: ChatGroq) -> dict[str, Any]:
    system_prompt = """
You are an independent medical citation-faithfulness judge.
Judge the answer ONLY against the literal cited abstract text provided.
Do NOT use outside knowledge. Do NOT use any gold long answer.

Classify the answer as exactly one of:
- "fully supported": every claim is stated or directly entailed by the literal cited abstract text
- "partially supported": the factual claims are accurate, but at least one clause adds a
  recommendation, clinical-utility judgment, or generalization not stated in the source
- "unsupported": a claim is fabricated or contradicts the source

Also identify the answer's overall stance as exactly one of: "yes", "no", "maybe".

Return valid JSON only:
{
  "judgment": "fully supported" | "partially supported" | "unsupported",
  "stance": "yes" | "no" | "maybe",
  "rationale": "1-2 sentences explaining the judgment",
  "failure_mode": "none" | "unauthorized_recommendation_or_utility_judgment" | "fabrication" | "contradiction" | "omission" | "other"
}
""".strip()
    user_prompt = f"""
Cited abstract (literal source text):
{abstract}

Answer to judge:
{answer}

Return JSON only.
""".strip()
    response = judge_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    parsed = _safe_json_loads(getattr(response, "content", "")) or {}
    judgment = str(parsed.get("judgment", "")).strip().lower()
    if judgment not in {"fully supported", "partially supported", "unsupported"}:
        judgment = "unsupported"
    stance = str(parsed.get("stance", "")).strip().lower()
    if stance not in {"yes", "no", "maybe"}:
        stance = "maybe"
    failure_mode = str(parsed.get("failure_mode", "other")).strip().lower() or "other"
    return {
        "judgment": judgment,
        "stance": stance,
        "rationale": str(parsed.get("rationale", "")).strip(),
        "failure_mode": failure_mode,
    }


def _run_pipeline_on_article(question: str, article: dict[str, Any]) -> tuple[str, str, list[Any]]:
    """Skip retrieve; inject one article; run synthesize -> verify_citations."""
    state: ResearchState = {
        "question": question,
        "comparison_question": None,
        "mode": "standard",
        "max_results": 1,
        "filters": {"year_from": None, "year_to": None, "study_type": ""},
        "history": [],
        "include_trials": False,
        "articles": [article],
        "comparison_articles": [],
        "cached_matches": [],
        "trials": [],
        "comparison_trials": [],
    }
    synth_update = _synthesize(state)
    state = {**state, **synth_update}
    baseline = str(state.get("synthesis", {}).get("answer", "")).strip()
    verify_update = _verify_citations(state)
    state = {**state, **verify_update}
    verified = str(state.get("synthesis", {}).get("answer", "")).strip() or baseline
    removed = state.get("synthesis", {}).get("verifier_removed_or_changed", [])
    if not isinstance(removed, list):
        removed = []
    return baseline, verified, removed


def _normalize_decision(value: str) -> str:
    value = (value or "").strip().lower()
    if value in {"yes", "no", "maybe"}:
        return value
    return value


def main() -> int:
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("LLM_PROVIDER", "groq")
    os.environ.setdefault("GROQ_MODEL", GENERATOR_MODEL)
    _patch_llm_retries()

    dataset = _ensure_dataset()
    missing = [pmid for pmid in PILOT_PMIDS if pmid not in dataset]
    if missing:
        raise SystemExit(f"Missing PMIDs in ori_pqal.json: {missing}")

    print(f"Generator model: {GENERATOR_MODEL} (live Groq)")
    print(f"Judge model:     {JUDGE_MODEL} (independent, live Groq)")
    print(f"Items:           {len(PILOT_PMIDS)}")
    print()

    pub_types = _fetch_publication_types(PILOT_PMIDS)
    used_live_pubtypes = any(bool(pub_types[p]) for p in PILOT_PMIDS)
    if used_live_pubtypes:
        print("Publication types: fetched via live NCBI E-utilities for assess_confidence().")
    else:
        print("Publication types: empty lists for all items (pilot-compatible fallback).")
    print()

    # Smoke-check article formatting matches pipeline helper.
    sample_article = _article_from_pubmedqa(PILOT_PMIDS[0], dataset[PILOT_PMIDS[0]], pub_types[PILOT_PMIDS[0]])
    formatted = _format_articles([sample_article])
    assert f"PMID {PILOT_PMIDS[0]}" in formatted
    print("Article formatting: reusing agent._format_articles (verified).\n")

    judge_llm = _build_judge_llm()
    results: list[dict[str, Any]] = []
    done_pmids: set[str] = set()
    if OUT_PATH.exists():
        try:
            prior = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            for row in prior.get("results", []):
                if row.get("pmid") in PILOT_PMIDS and row.get("baseline_answer"):
                    results.append(row)
                    done_pmids.add(row["pmid"])
            if done_pmids:
                print(f"Resuming with {len(done_pmids)} completed items from {OUT_PATH.name}.\n")
        except Exception:
            results = []
            done_pmids = set()

    for index, pmid in enumerate(PILOT_PMIDS, start=1):
        if pmid in done_pmids:
            print(f"[{index:02d}/{len(PILOT_PMIDS)}] PMID {pmid} ... skipped (cached)")
            continue
        item = dataset[pmid]
        question = str(item["QUESTION"]).strip()
        gold = _normalize_decision(str(item.get("final_decision", "")))
        article = _article_from_pubmedqa(pmid, item, pub_types.get(pmid, []))
        abstract = article["abstract"]

        print(f"[{index:02d}/{len(PILOT_PMIDS)}] PMID {pmid} ...", flush=True)
        baseline, verified, removed = _run_pipeline_on_article(question, article)
        confidence = assess_confidence([article], question)

        time.sleep(1.5)
        baseline_judge = _judge_faithfulness(baseline, abstract, judge_llm)
        time.sleep(1.5)
        verified_judge = _judge_faithfulness(verified, abstract, judge_llm)

        row = {
            "pmid": pmid,
            "question": question,
            "gold_decision": gold,
            "baseline_answer": baseline,
            "baseline_judgment": baseline_judge["judgment"],
            "baseline_stance": baseline_judge["stance"],
            "baseline_judge_rationale": baseline_judge["rationale"],
            "baseline_failure_mode": baseline_judge["failure_mode"],
            "verified_answer": verified,
            "verified_judgment": verified_judge["judgment"],
            "verified_stance": verified_judge["stance"],
            "verified_judge_rationale": verified_judge["rationale"],
            "verified_failure_mode": verified_judge["failure_mode"],
            "verifier_removed_or_changed": removed,
            "confidence": {
                "score": confidence.get("score"),
                "label": confidence.get("label"),
                "rationale": confidence.get("rationale"),
            },
            "decision_match_baseline": baseline_judge["stance"] == gold,
            "decision_match_verified": verified_judge["stance"] == gold,
        }
        results.append(row)
        # Keep results ordered by PILOT_PMIDS.
        by_pmid = {r["pmid"]: r for r in results}
        results = [by_pmid[p] for p in PILOT_PMIDS if p in by_pmid]
        _write_partial(results, used_live_pubtypes)
        print(
            f"         baseline={baseline_judge['judgment']}; "
            f"verified={verified_judge['judgment']}; "
            f"conf={confidence.get('label')}",
            flush=True,
        )
        time.sleep(2.0)

    payload = _build_payload(results, used_live_pubtypes)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_summary(results)
    return 0


def _build_payload(results: list[dict[str, Any]], used_live_pubtypes: bool) -> dict[str, Any]:
    return {
        "meta": {
            "requested_generator_model": "llama-3.1-8b-instant",
            "generator_model": GENERATOR_MODEL,
            "judge_model": JUDGE_MODEL,
            "judge_independent": JUDGE_MODEL != GENERATOR_MODEL,
            "model_note": (
                "Groq no longer exposes llama-3.1-8b-instant or llama-3.3-70b-versatile "
                "on this account; used the closest live Groq chat models with an "
                "independent larger judge."
            ),
            "publication_types_source": "ncbi_eutils_live" if used_live_pubtypes else "empty_list_pilot_fallback",
            "n_items": len(results),
            "pilot_reference": {
                "baseline_fully_supported": "17/20 (85%)",
                "verified_fully_supported": "20/20 (100%)",
                "all_low_confidence": True,
                "decision_agreement": "20/20",
                "notes": "Claude-substituted generator, self-judged",
            },
        },
        "results": results,
    }


def _write_partial(results: list[dict[str, Any]], used_live_pubtypes: bool) -> None:
    OUT_PATH.write_text(json.dumps(_build_payload(results, used_live_pubtypes), indent=2), encoding="utf-8")


def _print_summary(results: list[dict[str, Any]]) -> None:
    def count_fully(key: str) -> int:
        return sum(1 for row in results if row[key] == "fully supported")

    baseline_full = count_fully("baseline_judgment")
    verified_full = count_fully("verified_judgment")
    low_conf = sum(1 for row in results if str(row["confidence"].get("label", "")).lower() == "low")
    decision_agree = sum(1 for row in results if row["decision_match_verified"])
    n = len(results) or 1

    print("\n" + "=" * 72)
    print("SUMMARY vs pilot")
    print("=" * 72)
    print("Pilot (Claude-substituted, self-judged):")
    print("  - 17/20 (85%) baseline fully supported")
    print("  - 20/20 (100%) verified fully supported")
    print("  - 20/20 labeled Low confidence")
    print("  - 20/20 decision-level agreement with gold")
    print()
    print(
        "This run (live Groq "
        f"{GENERATOR_MODEL}, independent judge {JUDGE_MODEL}):"
    )
    print(f"  - {baseline_full}/{len(results)} ({baseline_full/n:.0%}) baseline fully supported")
    print(f"  - {verified_full}/{len(results)} ({verified_full/n:.0%}) verified fully supported")
    print(f"  - {low_conf}/{len(results)} labeled Low confidence")
    print(f"  - {decision_agree}/{len(results)} decision-level agreement with gold (verified stance)")
    print()
    print(f"Wrote {OUT_PATH}")

    print("\nFailure-mode notes (vs pilot's unauthorized recommendation/utility judgment):")
    flagged = False
    for row in results:
        for stage in ("baseline", "verified"):
            judgment = row[f"{stage}_judgment"]
            mode = row[f"{stage}_failure_mode"]
            if judgment != "fully supported" and mode not in {
                "none",
                "unauthorized_recommendation_or_utility_judgment",
            }:
                flagged = True
                print(
                    f"  - PMID {row['pmid']} {stage}: judgment={judgment}, "
                    f"failure_mode={mode}"
                )
    if not flagged:
        print("  - No fabrication/contradiction/omission flagged beyond pilot-style failures.")


if __name__ == "__main__":
    sys.exit(main())
