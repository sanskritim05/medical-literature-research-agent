import type { Filters, Mode, ResearchResult, StudyType } from "./research-types";
import { withoutEmDashes } from "./utils";

const STUDY_TYPE_TO_API: Record<StudyType, string> = {
  any: "",
  "clinical trial": "clinical-trial",
  "randomized controlled trial": "randomized-controlled-trial",
  "meta-analysis": "meta-analysis",
  "systematic review": "systematic-review",
  review: "review",
};

export type ActivityStep = {
  step: string;
  detail: string;
};

type ApiReference = {
  index?: number;
  title?: string;
  authors?: string[];
  link?: string;
  pmid?: string;
  journal?: string;
  year?: string;
  study_type?: string;
  summary?: string;
  highlights?: string[];
  abstract?: string;
};

type ApiTrial = {
  nct_id?: string;
  title?: string;
  status?: string;
  phase?: string;
  interventions?: string;
  condition?: string;
  enrollment?: string;
  location_count?: number;
  link?: string;
};

type ApiCached = {
  pmid?: string;
  title?: string;
  similarity?: number;
  link?: string;
  abstract?: string;
};

type ApiConfidence = {
  label?: string;
  score?: number;
  rationale?: string;
  model_explanation?: string;
};

type ApiResult = {
  answer?: string;
  plain_language_summary?: string;
  confidence?: ApiConfidence;
  references?: ApiReference[];
  comparison_references?: ApiReference[];
  trials?: ApiTrial[];
  comparison_trials?: ApiTrial[];
  cached_matches?: ApiCached[];
  session_id?: string;
  mode?: string;
  comparison_question?: string | null;
  filters?: Record<string, unknown>;
  activity?: ActivityStep[];
};

function mapConfidence(confidence: ApiConfidence | undefined): ResearchResult["synthesis"]["confidence"] {
  const label = confidence?.label;
  const level: ResearchResult["synthesis"]["confidence"]["level"] =
    label === "High" || label === "Moderate" || label === "Low" ? label : "Moderate";
  const raw = Number(confidence?.score ?? 0.5);
  const score = raw <= 1 ? Math.round(raw * 100) : Math.round(raw);
  const rationale =
    [confidence?.rationale, confidence?.model_explanation].filter(Boolean).join(" ") ||
    "Based on the retrieved abstracts.";
  return { level, score, rationale: withoutEmDashes(rationale) };
}

function mapReference(ref: ApiReference, fallbackIndex: number) {
  const authors = Array.isArray(ref.authors) ? ref.authors.join(", ") : "Authors unavailable";
  return {
    index: ref.index ?? fallbackIndex,
    pmid: ref.pmid ?? "",
    title: withoutEmDashes(ref.title ?? "Untitled"),
    authors: withoutEmDashes(authors),
    journal: withoutEmDashes(ref.journal ?? "Unknown"),
    year: ref.year ?? "",
    studyType: ref.study_type || "Unknown",
    abstract: withoutEmDashes(ref.abstract ?? ""),
    summary: withoutEmDashes(ref.summary || "No summary available."),
    highlights: Array.isArray(ref.highlights) ? ref.highlights.map(withoutEmDashes) : [],
    url: ref.link || (ref.pmid ? `https://pubmed.ncbi.nlm.nih.gov/${ref.pmid}/` : "#"),
  };
}

function mapTrial(trial: ApiTrial) {
  const enrollmentRaw = trial.enrollment;
  const enrollment =
    enrollmentRaw && enrollmentRaw !== "Unknown" && !Number.isNaN(Number(enrollmentRaw))
      ? Number(enrollmentRaw)
      : null;
  const interventions = (trial.interventions || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const conditions = (trial.condition || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const locationCount = trial.location_count;
  return {
    nctId: trial.nct_id ?? "",
    title: withoutEmDashes(trial.title ?? "Untitled trial"),
    status: trial.status ?? "Unknown",
    phase: trial.phase ?? "Unspecified",
    interventions,
    conditions,
    enrollment,
    locations:
      typeof locationCount === "number" && locationCount > 0
        ? `${locationCount} location${locationCount === 1 ? "" : "s"}`
        : "",
    url: trial.link || (trial.nct_id ? `https://clinicaltrials.gov/study/${trial.nct_id}` : "#"),
  };
}

function mapCached(item: ApiCached, question: string) {
  return {
    pmid: item.pmid ?? "",
    title: withoutEmDashes(item.title ?? "Untitled"),
    question,
    similarity: Number(item.similarity ?? 0),
    url: item.link || (item.pmid ? `https://pubmed.ncbi.nlm.nih.gov/${item.pmid}/` : "#"),
  };
}

export function mapApiResultToUi(
  payload: ApiResult,
  input: {
    question: string;
    compareQuestion?: string;
    mode: Mode;
    filters: Filters;
    sessionId: string;
  },
): ResearchResult {
  const references = (payload.references ?? []).map((ref, i) => mapReference(ref, i + 1));
  const comparisonReferences = (payload.comparison_references ?? []).map((ref, i) =>
    mapReference(ref, i + 1),
  );
  const trials = (payload.trials ?? []).map(mapTrial);
  const cached = (payload.cached_matches ?? []).map((item) => mapCached(item, input.question));
  const confidence = mapConfidence(payload.confidence);
  const result: ResearchResult = {
    sessionId: payload.session_id || input.sessionId,
    question: input.question,
    mode: input.mode,
    filters: input.filters,
    synthesis: {
      answer: withoutEmDashes(payload.answer?.trim() || "No answer generated."),
      plain_language_summary: withoutEmDashes(payload.plain_language_summary?.trim() || ""),
      confidence,
      key_takeaways: [],
    },
    references,
    comparisonReferences,
    trials,
    cached,
    createdAt: Date.now(),
  };
  if (input.compareQuestion) {
    result.compareQuestion = input.compareQuestion;
  }
  return result;
}

export async function runResearchStream(options: {
  question: string;
  compareQuestion?: string;
  mode: Mode;
  filters: Filters;
  sessionId: string;
  onActivity: (step: ActivityStep) => void;
}): Promise<ResearchResult> {
  const { question, compareQuestion, mode, filters, sessionId, onActivity } = options;
  const body: Record<string, unknown> = {
    question,
    max_results: filters.maxPapers,
    study_type: STUDY_TYPE_TO_API[filters.studyType],
    include_trials: filters.includeTrials,
    session_id: sessionId,
    mode,
  };
  if (filters.yearFrom) body["year_from"] = Number(filters.yearFrom);
  if (filters.yearTo) body["year_to"] = Number(filters.yearTo);
  if (mode === "compare" && compareQuestion) body["comparison_question"] = compareQuestion;

  const response = await fetch("/api/research/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) {
    const detail = await response.text();
    throw new Error(detail || `Research request failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let mapped: ResearchResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      const line = chunk
        .split("\n")
        .map((part) => part.trim())
        .find((part) => part.startsWith("data:"));
      if (!line) continue;
      const event = JSON.parse(line.slice(5).trim()) as {
        type: string;
        data: ApiResult & ActivityStep & { message?: string };
      };
      if (event.type === "activity") {
        onActivity({ step: event.data.step, detail: event.data.detail });
      } else if (event.type === "result") {
        mapped = mapApiResultToUi(event.data, {
          question,
          mode,
          filters,
          sessionId,
          ...(compareQuestion ? { compareQuestion } : {}),
        });
      } else if (event.type === "error") {
        throw new Error(event.data.message || "Research failed.");
      }
    }
  }

  if (!mapped) {
    throw new Error("Research completed without a result payload.");
  }
  return mapped;
}

export async function simplifyAnswer(question: string, answer: string): Promise<string> {
  const response = await fetch("/api/simplify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, answer }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Simplify request failed (${response.status})`);
  }
  const payload = (await response.json()) as { simplified_answer?: string };
  return withoutEmDashes(payload.simplified_answer?.trim() || "");
}
