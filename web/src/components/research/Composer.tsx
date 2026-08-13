import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Select } from "@/components/ui/select";
import { Loader2, Microscope, Sparkles } from "lucide-react";
import type { Filters, Mode, StudyType } from "@/lib/research-types";

const EXAMPLES = [
  "In adults with acute low back pain, are NSAIDs more effective than acetaminophen for pain relief at 1 week?",
  "Does early mobilisation after hip fracture surgery reduce 30-day mortality in patients over 75?",
  "In type 2 diabetes with CKD, do SGLT2 inhibitors slow eGFR decline compared with GLP-1 agonists?",
];

const STUDY_TYPES: { value: StudyType; label: string }[] = [
  { value: "any", label: "Any study type" },
  { value: "clinical trial", label: "Clinical trial" },
  { value: "randomized controlled trial", label: "Randomized controlled trial" },
  { value: "meta-analysis", label: "Meta-analysis" },
  { value: "systematic review", label: "Systematic review" },
  { value: "review", label: "Review" },
];

interface Props {
  question: string;
  setQuestion: (v: string) => void;
  compareQuestion: string;
  setCompareQuestion: (v: string) => void;
  mode: Mode;
  setMode: (m: Mode) => void;
  filters: Filters;
  setFilters: (f: Filters) => void;
  busy: boolean;
  onRun: () => void;
}

export function Composer({
  question,
  setQuestion,
  compareQuestion,
  setCompareQuestion,
  mode,
  setMode,
  filters,
  setFilters,
  busy,
  onRun,
}: Props) {
  return (
    <section className="panel composer">
      <div className="composer-top">
        <span className="label-caps">Clinical question</span>
        <div className="mode-toggle">
          {(["standard", "compare"] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`mode-btn${mode === m ? " active" : ""}`}
            >
              {m === "standard" ? "Standard" : "Compare treatments"}
            </button>
          ))}
        </div>
      </div>

      <Textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") onRun();
        }}
        placeholder="Ask a PICO-style question: population, intervention, comparison, outcome..."
      />

      {mode === "compare" && (
        <div className="field" style={{ marginTop: "0.75rem" }}>
          <span className="label-caps">Comparison question</span>
          <Textarea
            compare
            value={compareQuestion}
            onChange={(e) => setCompareQuestion(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") onRun();
            }}
            placeholder="e.g. In the same population, how effective is duloxetine?"
          />
        </div>
      )}

      {!question && (
        <div className="examples">
          <span className="label-caps">
            <Sparkles className="icon-sm" /> Try an example
          </span>
          <div className="example-grid">
            {EXAMPLES.map((ex) => (
              <button key={ex} type="button" onClick={() => setQuestion(ex)} className="example-btn">
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="filter-grid">
        <div className="field">
          <span className="label-caps">Study type</span>
          <Select
            value={filters.studyType}
            onChange={(e) => setFilters({ ...filters, studyType: e.target.value as StudyType })}
            options={STUDY_TYPES}
          />
        </div>
        <div className="field">
          <span className="label-caps">Publication years</span>
          <div className="year-row">
            <Input
              inputMode="numeric"
              placeholder="From"
              value={filters.yearFrom}
              onChange={(e) => setFilters({ ...filters, yearFrom: e.target.value.slice(0, 4) })}
            />
            <span className="muted">-</span>
            <Input
              inputMode="numeric"
              placeholder="To"
              value={filters.yearTo}
              onChange={(e) => setFilters({ ...filters, yearTo: e.target.value.slice(0, 4) })}
            />
          </div>
        </div>
        <div className="field">
          <span className="label-caps">Max papers</span>
          <Select
            value={String(filters.maxPapers)}
            onChange={(e) => setFilters({ ...filters, maxPapers: Number(e.target.value) })}
            options={[3, 5, 7, 10].map((n) => ({ value: String(n), label: `${n} papers` }))}
          />
        </div>
        <div style={{ display: "flex", alignItems: "flex-end" }}>
          <label className="trials-toggle">
            <span>
              Ongoing trials
              <small>ClinicalTrials.gov</small>
            </span>
            <Switch
              checked={filters.includeTrials}
              onCheckedChange={(v) => setFilters({ ...filters, includeTrials: v })}
            />
          </label>
        </div>
      </div>

      <div className="composer-actions">
        <Button size="lg" onClick={onRun} disabled={busy}>
          {busy ? <Loader2 className="icon spin" /> : <Microscope className="icon" />}
          {busy ? "Researching…" : "Run Research"}
        </Button>
        <span className="muted" style={{ fontSize: "0.75rem" }}>
          ⌘/Ctrl + Enter · searches PubMed, then synthesizes with citations
        </span>
      </div>
    </section>
  );
}
