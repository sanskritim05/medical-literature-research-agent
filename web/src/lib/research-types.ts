export type StudyType =
  | "any"
  | "clinical trial"
  | "randomized controlled trial"
  | "meta-analysis"
  | "systematic review"
  | "review";

export type Mode = "standard" | "compare";

export interface Filters {
  studyType: StudyType;
  yearFrom: string;
  yearTo: string;
  maxPapers: number;
  includeTrials: boolean;
}

export interface Reference {
  index: number;
  pmid: string;
  title: string;
  authors: string;
  journal: string;
  year: string;
  studyType: string;
  abstract: string;
  summary: string;
  highlights: string[];
  url: string;
}

export interface Trial {
  nctId: string;
  title: string;
  status: string;
  phase: string;
  interventions: string[];
  conditions: string[];
  enrollment: number | null;
  locations: string;
  url: string;
}

export interface CachedAbstract {
  pmid: string;
  title: string;
  question: string;
  similarity: number;
  url: string;
}

export interface Confidence {
  level: "Low" | "Moderate" | "High";
  score: number;
  rationale: string;
}

export interface Synthesis {
  answer: string;
  plain_language_summary: string;
  confidence: Confidence;
  key_takeaways: string[];
}

export interface ResearchResult {
  sessionId: string;
  question: string;
  compareQuestion?: string | undefined;
  mode: Mode;
  filters: Filters;
  synthesis: Synthesis;
  references: Reference[];
  comparisonReferences: Reference[];
  trials: Trial[];
  cached: CachedAbstract[];
  createdAt: number;
  simplified?: string | undefined;
}

export const DEFAULT_FILTERS: Filters = {
  studyType: "any",
  yearFrom: "",
  yearTo: "",
  maxPapers: 5,
  includeTrials: true,
};
