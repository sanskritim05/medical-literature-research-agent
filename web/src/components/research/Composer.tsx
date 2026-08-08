import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
    <section className="panel p-5 sm:p-6">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <span className="label-caps">Clinical question</span>
        <div className="flex rounded-md border border-border bg-muted p-0.5">
          {(["standard", "compare"] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`rounded-[5px] px-3 py-1.5 text-xs font-medium transition-colors ${
                mode === m
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
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
        className="min-h-[120px] resize-y bg-card text-base leading-relaxed"
      />

      {mode === "compare" && (
        <div className="mt-3">
          <span className="label-caps">Comparison question</span>
          <Textarea
            value={compareQuestion}
            onChange={(e) => setCompareQuestion(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") onRun();
            }}
            placeholder="e.g. In the same population, how effective is duloxetine?"
            className="mt-1.5 min-h-[90px] resize-y bg-card"
          />
        </div>
      )}

      {!question && (
        <div className="mt-4 space-y-2">
          <span className="label-caps flex items-center gap-1.5">
            <Sparkles className="h-3 w-3" /> Try an example
          </span>
          <div className="grid gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => setQuestion(ex)}
                className="rounded-md border border-border bg-surface px-3 py-2 text-left text-sm text-muted-foreground transition-colors hover:border-ring hover:text-foreground"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="mt-5 grid gap-3 border-t border-border pt-4 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <span className="label-caps">Study type</span>
          <Select
            value={filters.studyType}
            onValueChange={(v) => setFilters({ ...filters, studyType: v as StudyType })}
          >
            <SelectTrigger className="mt-1.5 bg-card">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {STUDY_TYPES.map((s) => (
                <SelectItem key={s.value} value={s.value}>
                  {s.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <span className="label-caps">Publication years</span>
          <div className="mt-1.5 flex items-center gap-2">
            <Input
              inputMode="numeric"
              placeholder="From"
              value={filters.yearFrom}
              onChange={(e) => setFilters({ ...filters, yearFrom: e.target.value.slice(0, 4) })}
              className="bg-card"
            />
            <span className="text-muted-foreground">-</span>
            <Input
              inputMode="numeric"
              placeholder="To"
              value={filters.yearTo}
              onChange={(e) => setFilters({ ...filters, yearTo: e.target.value.slice(0, 4) })}
              className="bg-card"
            />
          </div>
        </div>
        <div>
          <span className="label-caps">Max papers</span>
          <Select
            value={String(filters.maxPapers)}
            onValueChange={(v) => setFilters({ ...filters, maxPapers: Number(v) })}
          >
            <SelectTrigger className="mt-1.5 bg-card">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[3, 5, 7, 10].map((n) => (
                <SelectItem key={n} value={String(n)}>
                  {n} papers
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-end">
          <label className="flex w-full items-center justify-between gap-3 rounded-md border border-border bg-surface px-3 py-2.5">
            <span className="text-sm">
              Ongoing trials
              <span className="block text-xs text-muted-foreground">ClinicalTrials.gov</span>
            </span>
            <Switch
              checked={filters.includeTrials}
              onCheckedChange={(v) => setFilters({ ...filters, includeTrials: v })}
            />
          </label>
        </div>
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <Button size="lg" onClick={onRun} disabled={busy} className="gap-2">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Microscope className="h-4 w-4" />}
          {busy ? "Researching…" : "Run Research"}
        </Button>
        <span className="text-xs text-muted-foreground">
          ⌘/Ctrl + Enter · searches PubMed, then synthesizes with citations
        </span>
      </div>
    </section>
  );
}
