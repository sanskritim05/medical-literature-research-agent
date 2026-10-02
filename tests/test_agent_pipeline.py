import json

from agent import (
    _apply_citation_confidence,
    _coerce_query_plan,
    _prepare_reference_payload,
    _safe_json_loads,
    _strip_invalid_citations,
    _verify_citations,
    assign_citation_indexes,
    quote_in_abstract,
    rank_articles,
)
from pubmed_tool import exclude_retracted


def test_rank_articles_prefers_stronger_study_design():
    question = "Does metformin lower blood glucose in type 2 diabetes?"
    articles = [
        {
            "pmid": "1",
            "title": "Case report of metformin",
            "abstract": "Metformin was given to one patient with type 2 diabetes and blood glucose changed.",
            "year": "2024",
            "publication_types": ["Case Reports"],
            "study_type": "Case report",
        },
        {
            "pmid": "2",
            "title": "Metformin for type 2 diabetes",
            "abstract": "This systematic review found metformin lowers blood glucose in type 2 diabetes.",
            "year": "2018",
            "publication_types": ["Meta-Analysis", "Systematic Review"],
            "study_type": "Systematic review",
        },
        {
            "pmid": "3",
            "title": "Randomized trial of metformin",
            "abstract": "A randomized controlled trial of metformin lowered blood glucose in type 2 diabetes.",
            "year": "2021",
            "publication_types": ["Randomized Controlled Trial"],
            "study_type": "Randomized controlled trial",
        },
    ]

    ranked = rank_articles(articles, question, now_year=2026)

    assert [item["pmid"] for item in ranked] == ["2", "3", "1"]
    assert ranked[0]["rank_score"] > ranked[1]["rank_score"] > ranked[2]["rank_score"]
    assert "systematic review" in ranked[0]["rank_reason"]
    assert "case report" in ranked[2]["rank_reason"]


def test_study_design_ignores_title_when_publication_type_is_weaker():
    question = "Does metformin lower blood glucose in type 2 diabetes?"
    articles = [
        {
            "pmid": "1",
            "title": "Meta-analysis of metformin",
            "abstract": "This systematic review and meta-analysis found metformin lowers blood glucose in type 2 diabetes.",
            "year": "2024",
            "publication_types": ["Journal Article"],
            "study_type": "Journal Article",
        },
        {
            "pmid": "2",
            "title": "Metformin trial",
            "abstract": "A trial of metformin lowered blood glucose in type 2 diabetes.",
            "year": "2020",
            "publication_types": ["Randomized Controlled Trial"],
            "study_type": "Randomized controlled trial",
        },
    ]

    ranked = rank_articles(articles, question, now_year=2026)

    assert ranked[0]["pmid"] == "2"
    assert "unspecified design" in ranked[1]["rank_reason"]


def test_exclude_retracted_articles():
    kept = exclude_retracted(
        [
            {"pmid": "1", "publication_types": ["Retracted Publication", "Journal Article"]},
            {"pmid": "2", "publication_types": ["Retraction of Publication"]},
            {"pmid": "3", "publication_types": ["Meta-Analysis", "Systematic Review"]},
        ]
    )

    assert [item["pmid"] for item in kept] == ["3"]


def test_compare_citation_numbers_are_continuous():
    primary = [{"pmid": "a", "title": "Primary A"}, {"pmid": "b", "title": "Primary B"}]
    comparison = [{"pmid": "c", "title": "Comparison C"}]

    numbered_primary, numbered_comparison = assign_citation_indexes(primary, comparison)
    primary_refs = _prepare_reference_payload(numbered_primary, {}, "question", "answer")
    comparison_refs = _prepare_reference_payload(
        numbered_comparison,
        {},
        "question",
        "answer",
        start_index=len(numbered_primary) + 1,
    )

    assert [item["citation_index"] for item in numbered_primary] == [1, 2]
    assert numbered_comparison[0]["citation_index"] == 3
    assert [item["index"] for item in primary_refs] == [1, 2]
    assert [item["index"] for item in comparison_refs] == [3]


def test_safe_json_loads_extracts_object_from_prose():
    parsed = _safe_json_loads(
        'Here is the plan:\n{"pubmed_query": "metformin AND diabetes", "fallback_query": "metformin"}'
    )

    assert parsed is not None
    assert parsed["pubmed_query"] == "metformin AND diabetes"


def test_safe_json_loads_returns_none_for_invalid_text():
    assert _safe_json_loads("no json here") is None
    assert _safe_json_loads('["not", "an", "object"]') is None


def test_query_plan_falls_back_when_llm_json_fails_validation():
    plan = _coerce_query_plan(
        "Does metformin lower blood glucose in type 2 diabetes?",
        {"pubmed_query": "", "population": "adults"},
    )

    assert plan["source"] == "rule_based"
    assert "metformin" in plan["pubmed_query"].lower()
    assert plan["population"] == ""


def test_strip_invalid_citations_drops_sentences_with_only_bad_indexes():
    cleaned, invalid = _strip_invalid_citations(
        "Metformin lowered glucose [1]. It cures cancer [9].",
        {1},
    )

    assert cleaned == "Metformin lowered glucose [1]."
    assert invalid == [9]


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, _messages):
        return _FakeResponse(self.content)


def test_verifier_removes_unsupported_claims_and_invalid_citations(monkeypatch):
    payload = {
        "verified_answer": "Metformin lowered glucose [1]. It cures cancer [9].",
        "claims": [
            {
                "citation": 1,
                "sentence": "Metformin lowered glucose [1].",
                "supported": True,
                "quote": "Metformin lowered glucose in adults with type 2 diabetes.",
                "reason": "The abstract states glucose fell.",
            },
            {
                "citation": 9,
                "sentence": "It cures cancer [9].",
                "supported": False,
                "reason": "No abstract supports a cancer cure.",
            },
        ],
        "removed_or_changed": ["Removed the cancer cure claim because citation 9 is not a retrieved article."],
    }
    monkeypatch.setattr("agent._build_llm", lambda: _FakeLLM(json.dumps(payload)))

    state = {
        "synthesis": {"answer": "Metformin lowered glucose [1]. It cures cancer [9]."},
        "articles": [
            {
                "pmid": "111",
                "citation_index": 1,
                "title": "Metformin trial",
                "abstract": "Metformin lowered glucose in adults with type 2 diabetes.",
                "authors": ["Ada Lovelace"],
                "journal": "Lancet",
                "year": "2020",
                "publication_types": ["Randomized Controlled Trial"],
            }
        ],
        "comparison_articles": [],
    }

    result = _verify_citations(state)
    synthesis = result["synthesis"]
    audit = synthesis["citation_audit"]

    assert "cures cancer" not in synthesis["answer"].lower()
    assert "[1]" in synthesis["answer"]
    assert "[9]" not in synthesis["answer"]
    assert audit["invalid_citations"] == [9]
    assert audit["validity_rate"] == 0.5
    assert audit["support_rate"] == 0.5
    assert audit["unsupported_claims"][0]["citation"] == 9

    confidence = _apply_citation_confidence(
        {"score": 0.8, "label": "High", "rationale": "Several trials informed this answer."},
        audit,
    )
    assert confidence["label"] == "Moderate"
    assert confidence["score"] == 0.6
    assert confidence["citation_penalty"] == 2


def test_verifier_rejects_support_without_exact_quote(monkeypatch):
    payload = {
        "verified_answer": "Metformin cures diabetes [1].",
        "claims": [
            {
                "citation": 1,
                "sentence": "Metformin cures diabetes [1].",
                "supported": True,
                "quote": "This sentence was paraphrased and is not in the abstract.",
                "reason": "The model thought it was close enough.",
            }
        ],
        "removed_or_changed": [],
    }
    monkeypatch.setattr("agent._build_llm", lambda: _FakeLLM(json.dumps(payload)))
    state = {
        "synthesis": {"answer": "Metformin cures diabetes [1]."},
        "articles": [
            {
                "pmid": "111",
                "citation_index": 1,
                "title": "Metformin trial",
                "abstract": "Metformin lowered glucose in adults with type 2 diabetes.",
                "authors": [],
                "journal": "Lancet",
                "year": "2020",
                "publication_types": ["Randomized Controlled Trial"],
            }
        ],
    }

    audit = _verify_citations(state)["synthesis"]["citation_audit"]

    assert audit["support_rate"] == 0.0
    assert audit["unsupported_claims"][0]["supported"] is False
    assert quote_in_abstract(payload["claims"][0]["quote"], state["articles"][0]["abstract"]) is False
    assert quote_in_abstract(
        "Metformin lowered glucose in adults with type 2 diabetes.",
        state["articles"][0]["abstract"],
    )
