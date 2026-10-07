import json

from agent import (
    _apply_citation_confidence,
    _batch_relevance_scores,
    _coerce_query_plan,
    _invoke_llm,
    _prepare_reference_payload,
    _rank_evidence,
    _safe_json_loads,
    _strip_invalid_citations,
    _verify_citations,
    assign_citation_indexes,
    quote_in_abstract,
    rank_articles,
)
from pubmed_tool import exclude_retracted, search_pubmed


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


def test_safe_json_loads_strips_code_fence():
    parsed = _safe_json_loads(
        '```json\n{"verified_answer": "ok", "claims": [], "removed_or_changed": []}\n```'
    )

    assert parsed is not None
    assert parsed["verified_answer"] == "ok"
    assert parsed["claims"] == []


def test_query_plan_falls_back_when_pico_fields_are_missing():
    plan = _coerce_query_plan(
        "Does metformin lower blood glucose in type 2 diabetes?",
        {"pubmed_query": ""},
    )

    assert plan["source"] == "rule_based"
    assert "metformin" in plan["pubmed_query"].lower()
    assert plan["population"] == ""


def test_pico_query_is_parenthesized_and_drops_invalid_mesh(monkeypatch):
    def resolve(term: str):
        if term.lower() == "coronary disease":
            return "Coronary Disease"
        return None

    monkeypatch.setattr("agent.resolve_mesh_heading", resolve)
    plan = _coerce_query_plan(
        "Do statins reduce events?",
        {
            "population": "coronary heart disease",
            "intervention": "statins",
            "comparison": "placebo",
            "outcome": "major cardiovascular events",
            "mesh": {"population": ["Coronary Disease"], "intervention": ["Statins"]},
        },
    )

    query = plan["pubmed_query"]
    assert plan["source"] == "pico"
    assert query.startswith("(") and query.endswith(")")
    assert '"Coronary Disease"[Mesh]' in query
    assert "[Mesh]" in query and '"statins"[tiab]' in query
    assert '"Statins"[Mesh]' not in query
    assert " AND " in query
    assert '"major cardiovascular events"[tiab]' in query
    assert "[Mesh]" not in plan["fallback_query"]


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
    monkeypatch.setattr("agent._build_llm", lambda *args, **kwargs: _FakeLLM(json.dumps(payload)))

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
    monkeypatch.setattr("agent._build_llm", lambda *args, **kwargs: _FakeLLM(json.dumps(payload)))
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


def _metformin_article() -> dict:
    return {
        "pmid": "111",
        "citation_index": 1,
        "title": "Metformin trial",
        "abstract": "Metformin lowered glucose in adults with type 2 diabetes.",
        "authors": [],
        "journal": "Lancet",
        "year": "2020",
        "publication_types": ["Randomized Controlled Trial"],
    }


def test_zero_citations_fail_validity_and_support(monkeypatch):
    payload = {
        "verified_answer": "Statins reduce cardiovascular events.",
        "claims": [
            {
                "citation": 1,
                "sentence": "Statins reduce cardiovascular events.",
                "supported": True,
                "quote": "Metformin lowered glucose in adults with type 2 diabetes.",
                "reason": "The model treated an uncited answer as supported.",
            }
        ],
        "removed_or_changed": [],
    }
    monkeypatch.setattr("agent._build_llm", lambda *args, **kwargs: _FakeLLM(json.dumps(payload)))
    state = {
        "synthesis": {"answer": "Statins reduce cardiovascular events."},
        "articles": [_metformin_article()],
    }

    audit = _verify_citations(state)["synthesis"]["citation_audit"]

    assert audit["validity_rate"] == 0.0
    assert audit["support_rate"] == 0.0
    assert audit["citation_count"] == 0
    assert audit["supported_claim_count"] == 0


def test_missing_claim_list_fails_support(monkeypatch):
    payload = {
        "verified_answer": "Metformin lowered glucose [1].",
        "removed_or_changed": [],
    }
    monkeypatch.setattr("agent._build_llm", lambda *args, **kwargs: _FakeLLM(json.dumps(payload)))
    state = {
        "synthesis": {"answer": "Metformin lowered glucose [1]."},
        "articles": [_metformin_article()],
    }

    audit = _verify_citations(state)["synthesis"]["citation_audit"]

    assert audit["validity_rate"] == 1.0
    assert audit["support_rate"] == 0.0
    assert audit["claim_count"] == 1
    assert audit["supported_claim_count"] == 0
    assert audit["unsupported_claims"][0]["reason"] == "Verifier returned no claim list."


def test_claims_not_tied_to_citation_markers_are_unsupported(monkeypatch):
    payload = {
        "verified_answer": "Metformin lowered glucose [1].",
        "claims": [
            {
                "citation": None,
                "sentence": "Metformin lowered glucose.",
                "supported": True,
                "quote": "Metformin lowered glucose in adults with type 2 diabetes.",
                "reason": "Looks supported.",
            }
        ],
        "removed_or_changed": [],
    }
    monkeypatch.setattr("agent._build_llm", lambda *args, **kwargs: _FakeLLM(json.dumps(payload)))
    state = {
        "synthesis": {"answer": "Metformin lowered glucose [1]."},
        "articles": [_metformin_article()],
    }

    audit = _verify_citations(state)["synthesis"]["citation_audit"]

    assert audit["validity_rate"] == 1.0
    assert audit["support_rate"] == 0.0
    assert audit["unsupported_claims"][0]["reason"] == "Claim is not tied to a [n] citation in the answer."


def test_verifier_retries_once_when_the_first_reply_is_not_json(monkeypatch):
    good = {
        "verified_answer": "Metformin lowered glucose [1].",
        "claims": [
            {
                "citation": 1,
                "sentence": "Metformin lowered glucose [1].",
                "supported": True,
                "quote": "Metformin lowered glucose in adults with type 2 diabetes.",
                "reason": "The abstract states this.",
            }
        ],
        "removed_or_changed": [],
    }
    replies = iter(["Here is my review, not JSON.", "```json\n" + json.dumps(good) + "\n```"])
    prompts: list[str] = []

    def fake_invoke(messages, **kwargs):
        prompts.append(messages[0].content)
        return _FakeResponse(next(replies))

    monkeypatch.setattr("agent._invoke_llm", fake_invoke)
    state = {
        "synthesis": {"answer": "Metformin lowered glucose [1]."},
        "articles": [_metformin_article()],
    }

    audit = _verify_citations(state)["synthesis"]["citation_audit"]

    assert len(prompts) == 2
    assert "one JSON object only" in prompts[1]
    assert audit["support_rate"] == 1.0
    assert audit["supported_claim_count"] == 1


def test_quote_match_ignores_hyphen_case_and_trailing_punctuation():
    abstract = (
        "Low-intensity anticoagulation with warfarin prevented cerebral infarction "
        "in patients with nonrheumatic atrial fibrillation without producing an excess risk of major hemorrhage. "
        "Giving anticoagulants to older people with concomitant atrial fibrillation and chronic kidney disease "
        "was associated with an increased rate of ischaemic stroke and haemorrhage but a paradoxical lowered rate of all cause mortality."
    )
    hyphen_quote = (
        "Low\u2011intensity anticoagulation with warfarin prevented cerebral infarction "
        "in patients with nonrheumatic atrial fibrillation without producing an excess risk of major hemorrhage."
    )
    truncated_quote = (
        "Giving anticoagulants to older people with concomitant atrial fibrillation and chronic kidney disease "
        "was associated with an increased rate of ischaemic stroke and haemorrhage."
    )

    assert quote_in_abstract(hyphen_quote, abstract) is True
    assert quote_in_abstract(truncated_quote.upper(), abstract) is True
    assert quote_in_abstract("This sentence was paraphrased and is not in the abstract.", abstract) is False


def test_off_topic_review_ranks_below_on_topic_trial():
    question = "In adults with coronary heart disease, do statins reduce events compared with placebo?"
    plan = {
        "population": "adults with coronary heart disease",
        "intervention": "statins",
        "comparison": "placebo",
    }
    review = {
        "pmid": "1",
        "title": "Niacin for primary prevention of cardiovascular events",
        "abstract": "This systematic review of niacin found no clear benefit.",
        "year": "2024",
        "publication_types": ["Meta-Analysis"],
    }
    trial = {
        "pmid": "2",
        "title": "Statin therapy after coronary disease",
        "abstract": "Adults with coronary heart disease were assigned to a statin or placebo.",
        "year": "2010",
        "publication_types": ["Randomized Controlled Trial"],
    }

    without_plan = rank_articles([review, trial], question, now_year=2024)
    with_plan = rank_articles([review, trial], question, now_year=2024, plan=plan)

    assert without_plan[0]["pmid"] == "1"
    assert with_plan[0]["pmid"] == "2"
    assert "off-topic" in with_plan[1]["rank_reason"]
    assert "on-topic" in with_plan[0]["rank_reason"]


def test_confidence_capped_at_low_when_claim_support_is_zero():
    confidence = _apply_citation_confidence(
        {"score": 0.9, "label": "High", "rationale": "A review was retrieved."},
        {
            "invalid_citations": [],
            "unsupported_claims": [{"citation": 1, "supported": False}],
            "support_rate": 0.0,
        },
    )

    assert confidence["label"] == "Low"
    assert confidence["score"] <= 0.49
    assert "capped at Low" in confidence["rationale"]


def test_quote_match_folds_british_and_american_spelling():
    abstract = (
        "Warfarin reduced major hemorrhage without causing anemia or oedema "
        "in adults with atrial fibrillation."
    )
    quote = (
        "Warfarin reduced major haemorrhage without causing anaemia or edema "
        "in adults with atrial fibrillation."
    )

    assert quote_in_abstract(quote, abstract) is True


def test_ellipsis_quote_is_supported_when_every_segment_is_in_the_abstract():
    abstract = (
        "There were highly significant reductions of about one-quarter in the first event rate "
        "for non-fatal myocardial infarction or coronary death. "
        "For the first occurrence of any of these major vascular events, there was a definite 24% reduction in the event rate."
    )
    stitched = (
        "There were highly significant reductions of about one-quarter in the first event rate "
        "for non-fatal myocardial infarction or coronary death ... "
        "For the first occurrence of any of these major vascular events, there was a definite 24% reduction in the event rate."
    )
    invented = (
        "There were highly significant reductions of about one-quarter in the first event rate "
        "for non-fatal myocardial infarction or coronary death ... "
        "Statins eliminated every cardiovascular event in the placebo arm."
    )

    assert quote_in_abstract(stitched, abstract) is True
    assert quote_in_abstract(invented, abstract) is False


def test_off_topic_articles_skip_synthesis(monkeypatch):
    monkeypatch.setattr("agent._batch_relevance_scores", lambda question, articles: {})
    plan = {
        "population": "adults with coronary heart disease",
        "intervention": "statins",
        "comparison": "placebo",
        "pubmed_query": "statins coronary heart disease",
    }
    state = {
        "question": "Do statins reduce events compared with placebo?",
        "mode": "standard",
        "max_results": 4,
        "query_plan": plan,
        "articles": [
            {
                "pmid": "1",
                "title": "Niacin for primary prevention of cardiovascular events",
                "abstract": "This systematic review of niacin found no clear benefit.",
                "year": "2020",
                "publication_types": ["Meta-Analysis"],
                "relevance_llm_score": 0,
            }
        ],
        "comparison_articles": [],
    }

    result = _rank_evidence(state)

    assert result["needs_expert_review"] is True
    assert "needs expert review" in result["synthesis"]["answer"]


def test_one_strong_article_still_abstains():
    state = {
        "question": "Do statins reduce events compared with placebo?",
        "mode": "standard",
        "max_results": 4,
        "query_plan": {"population": "coronary heart disease", "intervention": "statins", "comparison": "placebo"},
        "articles": [
            {
                "pmid": "2",
                "title": "Simvastatin in coronary heart disease",
                "abstract": "Adults with coronary heart disease were assigned to a statin or placebo.",
                "year": "1994",
                "publication_types": ["Randomized Controlled Trial"],
                "relevance_llm_score": 2,
            },
            {
                "pmid": "3",
                "title": "Omega-3 fatty acids",
                "abstract": "Fish oil for primary prevention.",
                "year": "2018",
                "publication_types": ["Meta-Analysis"],
                "relevance_llm_score": 0,
            },
        ],
        "comparison_articles": [],
    }

    result = _rank_evidence(state)

    assert result["needs_expert_review"] is True


def test_on_topic_articles_continue_to_synthesis():
    article = {
        "title": "Simvastatin in coronary heart disease",
        "abstract": "Adults with coronary heart disease were assigned to a statin or placebo.",
        "year": "1994",
        "publication_types": ["Randomized Controlled Trial"],
        "relevance_llm_score": 2,
    }
    state = {
        "question": "Do statins reduce events compared with placebo?",
        "mode": "standard",
        "max_results": 4,
        "query_plan": {"population": "coronary heart disease", "intervention": "statins", "comparison": "placebo"},
        "articles": [{**article, "pmid": "2"}, {**article, "pmid": "4"}],
        "comparison_articles": [],
    }

    result = _rank_evidence(state)

    assert result["needs_expert_review"] is False
    assert "synthesis" not in result


def test_relevance_score_ranks_ahead_of_study_design():
    question = "Do statins reduce events in coronary heart disease compared with placebo?"
    review = {
        "pmid": "1",
        "title": "Niacin review",
        "abstract": "Niacin for primary prevention.",
        "year": "2024",
        "publication_types": ["Meta-Analysis"],
        "relevance_llm_score": 0,
    }
    trial = {
        "pmid": "7968073",
        "title": "Scandinavian Simvastatin Survival Study",
        "abstract": "Simvastatin versus placebo in coronary heart disease.",
        "year": "1994",
        "publication_types": ["Randomized Controlled Trial"],
        "relevance_llm_score": 2,
    }
    on_topic_review = {**review, "pmid": "9", "relevance_llm_score": 2}

    ranked = rank_articles([review, trial], question, now_year=2024)
    both_strong = rank_articles([on_topic_review, trial], question, now_year=2024)

    assert ranked[0]["pmid"] == "7968073"
    assert both_strong[0]["pmid"] == "9"


def test_rct_ids_are_kept_when_reviews_fill_the_pool(monkeypatch):
    def fake_esearch(term, retmax):
        if "Randomized Controlled Trial" in term:
            return ["9001"]
        if "Systematic Review" in term or "Meta-Analysis" in term:
            return [str(1000 + index) for index in range(retmax)]
        return []

    def lookup(pmids):
        return {
            pmid: {
                "pmid": pmid,
                "title": "Trial" if pmid == "9001" else "Review",
                "abstract": "An abstract long enough to keep.",
                "publication_types": ["Randomized Controlled Trial"] if pmid == "9001" else ["Meta-Analysis"],
            }
            for pmid in pmids
        }

    monkeypatch.setattr("pubmed_tool._esearch_pmids", fake_esearch)
    monkeypatch.setattr("pubmed_tool._lookup_cached_pubmed_articles", lookup)
    monkeypatch.setattr("pubmed_tool._read_retrieval_cache", lambda: {})
    monkeypatch.setattr("pubmed_tool._write_retrieval_cache", lambda _cache: None)

    articles = search_pubmed("metformin placebo diabetes", max_results=5)

    assert "9001" in [article["pmid"] for article in articles]


class _SequencedLLM:
    def __init__(self, responses, max_tokens=None):
        self.responses = responses
        self.max_tokens = max_tokens

    def invoke(self, _messages):
        return self.responses.pop(0)


def test_empty_cut_off_reply_retries_with_higher_max_tokens(monkeypatch):
    built: list[int | None] = []

    class _EmptyReply:
        content = ""
        response_metadata = {
            "finish_reason": "length",
            "token_usage": {
                "completion_tokens": 128,
                "completion_tokens_details": {"reasoning_tokens": 128},
            },
        }

    class _FullReply:
        content = '{"claims": []}'
        response_metadata = {"finish_reason": "stop", "token_usage": {"completion_tokens": 20}}

    def build(max_tokens=None, reasoning_effort=None):
        built.append(max_tokens)
        if max_tokens is None:
            return _SequencedLLM([_EmptyReply()])
        return _SequencedLLM([_FullReply()], max_tokens=max_tokens)

    monkeypatch.setattr("agent._build_llm", build)

    response = _invoke_llm([])

    assert built == [None, 4096]
    assert response.content == '{"claims": []}'


def test_narrow_query_merges_the_fallback(monkeypatch):
    def fake_esearch(term, retmax):
        if "narrow-query" in term:
            return ["1"] if "Randomized Controlled Trial" in term else ["2"]
        if "Randomized Controlled Trial" in term:
            return ["3"]
        return ["4", "5", "6"]

    def lookup(pmids):
        return {
            pmid: {
                "pmid": pmid,
                "title": "Study",
                "abstract": "An abstract long enough to keep.",
                "publication_types": ["Journal Article"],
            }
            for pmid in pmids
        }

    monkeypatch.setattr("pubmed_tool._esearch_pmids", fake_esearch)
    monkeypatch.setattr("pubmed_tool._lookup_cached_pubmed_articles", lookup)
    monkeypatch.setattr("pubmed_tool._read_retrieval_cache", lambda: {})
    monkeypatch.setattr("pubmed_tool._write_retrieval_cache", lambda _cache: None)

    articles = search_pubmed(
        "unused question",
        max_results=15,
        query_candidates=["narrow-query", "broad-query"],
    )
    pmids = [article["pmid"] for article in articles]

    assert "2" in pmids
    assert "4" in pmids


def test_relevance_batches_score_every_pmid_and_retry_missing(monkeypatch):
    articles = [{"pmid": str(index), "title": f"Study {index}", "abstract": "An abstract about the question."} for index in range(1, 13)]
    calls: list[list[str]] = []
    dropped = {"done": False}

    def fake_invoke(messages, **kwargs):
        text = messages[-1].content
        pmids = []
        for line in text.splitlines():
            if line.startswith("PMID "):
                pmids.append(line.replace("PMID ", "").strip())
        calls.append(pmids)
        payload = {pmid: 2 for pmid in pmids}
        if "12" in pmids and not dropped["done"]:
            dropped["done"] = True
            payload.pop("12")
        return _FakeResponse(json.dumps(payload))

    monkeypatch.setattr("agent._invoke_llm", fake_invoke)

    scores = _batch_relevance_scores("Do statins help?", articles)

    assert set(scores) == {str(index) for index in range(1, 13)}
    assert scores["12"] == 2
    assert any(call == ["12"] or "12" in call and len(call) < 6 for call in calls[1:])
