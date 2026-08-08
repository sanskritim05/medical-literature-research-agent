import hashlib
import html
import json
import math
import os
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CLINICAL_TRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"
REQUEST_TIMEOUT = 20
BASE_DIR = Path(__file__).resolve().parent
# Vercel only allows writes under /tmp; local keeps a project .cache folder.
CACHE_DIR = Path("/tmp/literature-cache") if os.getenv("VERCEL") else (BASE_DIR / ".cache" / "literature")
CACHE_FILE = CACHE_DIR / "literature_cache.json"
MAX_CACHE_ITEMS = 250
_CACHE_LOCK = threading.Lock()

STUDY_TYPE_FILTERS = {
    "clinical-trial": '"Clinical Trial"[Publication Type]',
    "meta-analysis": '"Meta-Analysis"[Publication Type]',
    "systematic-review": '"Systematic Review"[Publication Type]',
    "review": '"Review"[Publication Type]',
    "randomized-controlled-trial": '"Randomized Controlled Trial"[Publication Type]',
}

PUBMED_QUERY_STOPWORDS = {
    "a",
    "adults",
    "an",
    "and",
    "are",
    "compared",
    "comparedwith",
    "do",
    "does",
    "evidence",
    "for",
    "in",
    "inform",
    "is",
    "of",
    "on",
    "or",
    "the",
    "there",
    "to",
    "use",
    "what",
    "with",
}

TRIAL_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "compared",
    "comparedwith",
    "care",
    "do",
    "does",
    "evidence",
    "for",
    "in",
    "is",
    "major",
    "of",
    "on",
    "or",
    "standard",
    "the",
    "to",
    "what",
    "with",
}


def _embed_text(text: str, dims: int = 96) -> list[float]:
    vector = [0.0] * dims
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        slot = int(digest[:8], 16) % dims
        sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
        weight = 1.0 + (len(token) / 10.0)
        vector[slot] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _ncbi_params(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = dict(extra or {})
    api_key = os.getenv("NCBI_API_KEY", "").strip()
    email = os.getenv("NCBI_EMAIL", "").strip()
    tool = os.getenv("NCBI_TOOL", "medical-literature-research-agent").strip()
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    if tool:
        params["tool"] = tool
    return params


def _load_cache() -> dict[str, Any]:
    try:
        if not CACHE_FILE.exists():
            return {"items": {}}
        with CACHE_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), dict):
            return {"items": {}}
        return payload
    except Exception:
        return {"items": {}}


def _save_cache(payload: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    items = payload.get("items", {})
    if len(items) > MAX_CACHE_ITEMS:
        # Drop oldest entries when the ephemeral cache grows too large.
        ordered = sorted(items.items(), key=lambda pair: pair[1].get("updated_at", 0))
        items = dict(ordered[-MAX_CACHE_ITEMS:])
        payload = {"items": items}
    tmp_path = CACHE_FILE.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    tmp_path.replace(CACHE_FILE)


def _clean_text(value: str) -> str:
    text = html.unescape(value or "")
    return re.sub(r"\s+", " ", text).strip()


def _extract_abstract(article: ET.Element) -> str:
    parts: list[str] = []
    for abstract_text in article.findall(".//Abstract/AbstractText"):
        label = abstract_text.attrib.get("Label", "").strip()
        section = "".join(abstract_text.itertext()).strip()
        if not section:
            continue
        parts.append(f"{label}: {section}" if label else section)
    return _clean_text(" ".join(parts))


def _extract_authors(article: ET.Element) -> list[str]:
    authors: list[str] = []
    for author in article.findall(".//AuthorList/Author"):
        last_name = author.findtext("LastName", default="").strip()
        initials = author.findtext("Initials", default="").strip()
        collective_name = author.findtext("CollectiveName", default="").strip()
        if collective_name:
            authors.append(collective_name)
            continue
        name = " ".join(part for part in [last_name, initials] if part)
        if name:
            authors.append(name)
    return authors


def _extract_publication_types(article: ET.Element) -> list[str]:
    types: list[str] = []
    for item in article.findall(".//PublicationTypeList/PublicationType"):
        value = _clean_text("".join(item.itertext()))
        if value:
            types.append(value)
    return types


def _extract_year(article: ET.Element) -> str:
    candidates = [
        article.findtext(".//PubDate/Year", default=""),
        article.findtext(".//ArticleDate/Year", default=""),
        article.findtext(".//PubMedPubDate[@PubStatus='pubmed']/Year", default=""),
    ]
    for candidate in candidates:
        candidate = _clean_text(candidate)
        if candidate:
            return candidate
    medline_date = _clean_text(article.findtext(".//PubDate/MedlineDate", default=""))
    match = re.search(r"(19|20)\d{2}", medline_date)
    return match.group(0) if match else ""


def _extract_article_data(article: ET.Element) -> dict[str, Any]:
    pmid = _clean_text(article.findtext(".//PMID", default=""))
    title_element = article.find(".//ArticleTitle")
    title = _clean_text("".join(title_element.itertext()) if title_element is not None else "")
    journal = _clean_text(article.findtext(".//Journal/Title", default=""))
    pub_year = _extract_year(article)
    abstract = _extract_abstract(article)
    authors = _extract_authors(article)
    publication_types = _extract_publication_types(article)
    return {
        "source": "pubmed",
        "pmid": pmid,
        "title": title,
        "journal": journal,
        "year": pub_year,
        "authors": authors,
        "abstract": abstract,
        "publication_types": publication_types,
        "study_type": publication_types[0] if publication_types else "",
        "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
    }


def _build_pubmed_term(query: str, year_from: int | None = None, year_to: int | None = None, study_type: str | None = None) -> str:
    clauses = [f"({query})"]
    if year_from or year_to:
        start = year_from or 1900
        end = year_to or 3000
        clauses.append(f'("{start}"[Date - Publication] : "{end}"[Date - Publication])')
    if study_type and study_type in STUDY_TYPE_FILTERS:
        clauses.append(f"({STUDY_TYPE_FILTERS[study_type]})")
    return " AND ".join(clauses)


def _normalize_pubmed_query(query: str) -> str:
    cleaned = re.sub(r"[?.,:;()/]+", " ", query.lower())
    cleaned = cleaned.replace("glp-1", "glp1")
    tokens = re.findall(r"[a-z0-9+-]+", cleaned)
    filtered = [token for token in tokens if token not in PUBMED_QUERY_STOPWORDS and len(token) > 2]

    normalized_text = " ".join(filtered)
    phrase_replacements = {
        "atrial fibrillation": '"atrial fibrillation"',
        "stroke prevention": '"stroke prevention"',
        "acute low back pain": '"acute low back pain"',
        "sudden sensorineural hearing loss": '"sudden sensorineural hearing loss"',
        "type 2 diabetes": '"type 2 diabetes"',
        "glp1 receptor agonists": '"GLP-1 receptor agonists"',
        "glp1 receptor agonist": '"GLP-1 receptor agonist"',
        "sglt2 inhibitors": '"SGLT2 inhibitors"',
        "oral steroids": '"oral steroids"',
        "oral corticosteroids": '"oral corticosteroids"',
        "intratympanic steroids": '"intratympanic steroids"',
        "apixaban": "apixaban",
        "rivaroxaban": "rivaroxaban",
        "semaglutide": "semaglutide",
        "tirzepatide": "tirzepatide",
    }

    phrases: list[str] = []
    for source, replacement in phrase_replacements.items():
        if source in normalized_text:
            phrases.append(replacement)
            normalized_text = normalized_text.replace(source, " ")

    remaining_tokens = [token for token in normalized_text.split() if token not in {"adults", "children"}]
    compact = phrases + remaining_tokens[:6]
    compact = [token for token in compact if token]
    return " ".join(compact).strip() or query.strip()


def _upsert_articles_in_cache(articles: list[dict[str, Any]]) -> None:
    if not articles:
        return

    try:
        import time

        with _CACHE_LOCK:
            payload = _load_cache()
            items = payload.setdefault("items", {})
            now = time.time()
            for article in articles:
                pmid = str(article.get("pmid", "")).strip()
                abstract = article.get("abstract", "")
                if not pmid or not abstract:
                    continue
                items[f"pubmed:{pmid}"] = {
                    "source": "pubmed",
                    "pmid": pmid,
                    "title": article.get("title", ""),
                    "journal": article.get("journal", ""),
                    "year": article.get("year", ""),
                    "link": article.get("link", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                    "study_type": article.get("study_type", ""),
                    "abstract": abstract,
                    "embedding": _embed_text(f"{article.get('title', '')} {abstract}"),
                    "updated_at": now,
                }
            _save_cache(payload)
    except Exception:
        # Cache is best-effort on serverless; never fail the research request.
        return


def _lookup_cached_pubmed_articles(pmids: list[str]) -> dict[str, dict[str, Any]]:
    if not pmids:
        return {}
    try:
        with _CACHE_LOCK:
            items = _load_cache().get("items", {})
        results: dict[str, dict[str, Any]] = {}
        for pmid in pmids:
            item = items.get(f"pubmed:{pmid}")
            if not item:
                continue
            results[pmid] = {
                "source": "pubmed",
                "pmid": pmid,
                "title": item.get("title", ""),
                "journal": item.get("journal", ""),
                "year": item.get("year", ""),
                "authors": [],
                "abstract": item.get("abstract", ""),
                "publication_types": [item.get("study_type", "")] if item.get("study_type") else [],
                "study_type": item.get("study_type", ""),
                "link": item.get("link", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                "cached": True,
            }
        return results
    except Exception:
        return {}


def search_cached_literature(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    try:
        with _CACHE_LOCK:
            items = list(_load_cache().get("items", {}).values())
        if not items or not query.strip():
            return []

        query_embedding = _embed_text(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                embedding = _embed_text(f"{item.get('title', '')} {item.get('abstract', '')}")
            similarity = _cosine_similarity(query_embedding, embedding)
            if similarity <= 0.05:
                continue
            scored.append((similarity, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: list[dict[str, Any]] = []
        for similarity, item in scored[:max_results]:
            results.append(
                {
                    "source": item.get("source", "pubmed"),
                    "pmid": item.get("pmid", ""),
                    "title": item.get("title", ""),
                    "journal": item.get("journal", ""),
                    "year": item.get("year", ""),
                    "study_type": item.get("study_type", ""),
                    "link": item.get("link", ""),
                    "abstract": item.get("abstract", ""),
                    "similarity": round(float(similarity), 3),
                }
            )
        return results
    except Exception:
        return []


def _normalize_trial_query(query: str) -> str:
    cleaned = re.sub(r"[?.,:;()/]+", " ", query.lower())
    cleaned = cleaned.replace("glp-1", "glp1")
    tokens = re.findall(r"[a-z0-9+-]+", cleaned)
    filtered = [token for token in tokens if token not in TRIAL_QUERY_STOPWORDS and len(token) > 2]

    phrase_replacements = {
        "type 2 diabetes": "type 2 diabetes",
        "type 1 diabetes": "type 1 diabetes",
        "acute low back pain": "acute low back pain",
        "sudden sensorineural hearing loss": "sudden sensorineural hearing loss",
        "glp1 receptor agonists": "glp-1 receptor agonist",
        "glp1 receptor agonist": "glp-1 receptor agonist",
        "cardiovascular events": "cardiovascular",
    }

    normalized_text = " ".join(filtered)
    phrases: list[str] = []
    for source, replacement in phrase_replacements.items():
        if source in normalized_text:
            phrases.append(replacement)
            normalized_text = normalized_text.replace(source, " ")

    remaining_tokens = [token for token in normalized_text.split() if token not in {"adults", "children"}]
    compact = phrases + remaining_tokens[:6]
    compact = [token for token in compact if token]
    return " ".join(compact).strip() or query.strip()


def search_pubmed(
    query: str,
    max_results: int = 5,
    year_from: int | None = None,
    year_to: int | None = None,
    study_type: str | None = None,
) -> list[dict[str, Any]]:
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    candidate_queries: list[str] = []
    normalized_query = _normalize_pubmed_query(query)
    for candidate in [normalized_query, query.strip()]:
        if candidate and candidate not in candidate_queries:
            candidate_queries.append(candidate)

    pmids: list[str] = []
    for candidate in candidate_queries:
        esearch_params = _ncbi_params(
            {
                "db": "pubmed",
                "term": _build_pubmed_term(candidate, year_from=year_from, year_to=year_to, study_type=study_type),
                "retmode": "xml",
                "retmax": max_results,
                "sort": "relevance",
            }
        )
        esearch_response = requests.get(ESEARCH_URL, params=esearch_params, timeout=REQUEST_TIMEOUT)
        esearch_response.raise_for_status()

        root = ET.fromstring(esearch_response.text)
        pmids = [elem.text.strip() for elem in root.findall(".//IdList/Id") if elem.text]
        if pmids:
            break

    if not pmids:
        return []

    cached_articles = _lookup_cached_pubmed_articles(pmids)
    missing_pmids = [pmid for pmid in pmids if pmid not in cached_articles]
    fetched_articles: list[dict[str, Any]] = []

    if missing_pmids:
        efetch_params = _ncbi_params(
            {
                "db": "pubmed",
                "id": ",".join(missing_pmids),
                "retmode": "xml",
                "rettype": "abstract",
            }
        )
        efetch_response = requests.get(EFETCH_URL, params=efetch_params, timeout=REQUEST_TIMEOUT)
        efetch_response.raise_for_status()

        fetch_root = ET.fromstring(efetch_response.text)
        for article in fetch_root.findall(".//PubmedArticle"):
            article_data = _extract_article_data(article)
            if article_data["title"] and article_data["abstract"]:
                fetched_articles.append(article_data)
        _upsert_articles_in_cache(fetched_articles)

    combined = {article["pmid"]: article for article in fetched_articles}
    combined.update(cached_articles)

    ordered_articles: list[dict[str, Any]] = []
    for pmid in pmids:
        article = combined.get(pmid)
        if article and article.get("abstract"):
            ordered_articles.append(article)
    return ordered_articles[:max_results]


def search_clinical_trials(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    if not query or not query.strip():
        return []

    candidate_queries = []
    normalized_query = _normalize_trial_query(query)
    for candidate in [normalized_query, query.strip()]:
        if candidate and candidate not in candidate_queries:
            candidate_queries.append(candidate)

    payload = None
    for candidate in candidate_queries:
        params = {
            "query.term": candidate,
            "pageSize": max_results,
            "format": "json",
        }
        response = requests.get(CLINICAL_TRIALS_URL, params=params, timeout=REQUEST_TIMEOUT)
        if response.ok:
            payload = response.json()
            break
        if response.status_code != 400:
            response.raise_for_status()

    if payload is None:
        return []

    studies = payload.get("studies", [])
    results: list[dict[str, Any]] = []
    for study in studies:
        protocol = study.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        arms_module = protocol.get("armsInterventionsModule", {})
        design_module = protocol.get("designModule", {})
        contacts_module = protocol.get("contactsLocationsModule", {})

        interventions = []
        for item in arms_module.get("interventions", []) or []:
            name = item.get("name")
            if name:
                interventions.append(name)

        nct_id = identification.get("nctId", "")
        phases = design_module.get("phases", []) or []
        condition_list = conditions_module.get("conditions", []) or []
        overall_status = status_module.get("overallStatus", "Unknown")
        enrollment_info = design_module.get("enrollmentInfo", {}) or {}
        sponsor_name = identification.get("organization", {}).get("fullName", "")
        locations = contacts_module.get("locations", []) or []
        location_count = len(locations)
        results.append(
            {
                "source": "clinicaltrials",
                "nct_id": nct_id,
                "title": identification.get("briefTitle", "Untitled trial"),
                "condition": ", ".join(condition_list),
                "interventions": ", ".join(interventions),
                "status": overall_status,
                "phase": ", ".join(phases) if phases else "Unspecified",
                "study_type": design_module.get("studyType", "Unspecified"),
                "enrollment": str(enrollment_info.get("count", "Unknown")),
                "sponsor": sponsor_name,
                "location_count": location_count,
                "link": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "https://clinicaltrials.gov/",
            }
        )
    return results


def extract_highlight_sentences(abstract: str, query: str, synthesis_text: str, max_sentences: int = 2) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", abstract or "") if sentence.strip()]
    if not sentences:
        return []

    query_terms = set(re.findall(r"[a-z0-9]+", f"{query} {synthesis_text}".lower()))
    scored: list[tuple[int, str]] = []
    for sentence in sentences:
        sentence_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        score = len(query_terms & sentence_terms)
        if re.search(r"\b(reduced|improved|significant|associated|compared|no difference|superior|inferior)\b", sentence.lower()):
            score += 2
        scored.append((score, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    for _, sentence in scored:
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) == max_sentences:
            break
    return selected


def assess_confidence(articles: list[dict[str, Any]], question: str) -> dict[str, Any]:
    if not articles:
        return {
            "score": 0.18,
            "label": "Low",
            "rationale": "No relevant abstracts were retrieved, so the answer is based on very limited evidence.",
        }

    score = 0.35
    publication_types = " ".join(" ".join(article.get("publication_types", [])) for article in articles).lower()
    combined_text = " ".join(article.get("abstract", "") for article in articles).lower()
    article_count = len(articles)

    score += min(article_count, 5) * 0.08
    if "meta-analysis" in publication_types or "systematic review" in publication_types:
        score += 0.18
    if "randomized controlled trial" in publication_types or "clinical trial" in publication_types:
        score += 0.14
    if any(re.search(r"\b(no difference|conflicting|mixed results|inconsistent)\b", article.get("abstract", "").lower()) for article in articles):
        score -= 0.18
    if article_count < 3:
        score -= 0.1
    if re.search(r"\b(case report|pilot)\b", publication_types):
        score -= 0.08

    score = max(0.05, min(score, 0.98))
    if score >= 0.75:
        label = "High"
    elif score >= 0.5:
        label = "Moderate"
    else:
        label = "Low"

    rationale_parts = [f"{article_count} PubMed abstract{'s' if article_count != 1 else ''} informed this answer"]
    if "meta-analysis" in publication_types or "systematic review" in publication_types:
        rationale_parts.append("the evidence includes higher-level review data")
    if "randomized controlled trial" in publication_types or "clinical trial" in publication_types:
        rationale_parts.append("the set includes interventional studies")
    if re.search(r"\b(no difference|conflicting|mixed results|inconsistent)\b", combined_text):
        rationale_parts.append("some abstracts describe mixed or conflicting findings")
    rationale = ". ".join(part.capitalize() for part in rationale_parts) + "."
    return {"score": round(score, 2), "label": label, "rationale": rationale}
