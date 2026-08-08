import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, BookOpen, Stethoscope } from "lucide-react";
import { Composer } from "@/components/research/Composer";
import { ProgressTimeline, type Step } from "@/components/research/ProgressTimeline";
import { AnswerPanel } from "@/components/research/AnswerPanel";
import {
  CachedCard,
  EmptyNote,
  ReferenceCard,
  TrialCard,
} from "@/components/research/EvidenceCards";
import { HistorySidebar } from "@/components/research/HistorySidebar";
import { exportResultPdf } from "@/lib/pdf";
import { runResearchStream, simplifyAnswer } from "@/lib/api";
import {
  DEFAULT_FILTERS,
  type Filters,
  type Mode,
  type ResearchResult,
} from "@/lib/research-types";

const HISTORY_KEY = "evidentia.history";
const SESSION_KEY = "evidentia.session";

function baseSteps(mode: Mode, includeTrials: boolean): Step[] {
  const steps: Step[] = [
    { id: "context", label: "Building session context", detail: "Framing the PICO question", state: "pending" },
    { id: "pubmed", label: "Searching PubMed", detail: "Retrieving peer-reviewed abstracts", state: "pending" },
  ];
  steps.push({
    id: "trials",
    label: "Scanning ClinicalTrials.gov",
    detail: includeTrials ? "Looking for ongoing studies" : "Ongoing trials disabled",
    state: includeTrials ? "pending" : "skipped",
  });
  steps.push({
    id: "compare",
    label: "Retrieving comparison evidence",
    detail: mode === "compare" ? "Second literature pass" : "Standard mode",
    state: mode === "compare" ? "pending" : "skipped",
  });
  steps.push(
    { id: "synthesis", label: "Synthesizing evidence", detail: "Weighing findings and citations", state: "pending" },
    { id: "done", label: "Report ready", detail: "Confidence scored and referenced", state: "pending" },
  );
  return steps;
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [compareQuestion, setCompareQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("standard");
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const [steps, setSteps] = useState<Step[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [simplifying, setSimplifying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [history, setHistory] = useState<ResearchResult[]>([]);
  const [sessionId, setSessionId] = useState("");

  useEffect(() => {
    try {
      const stored = localStorage.getItem(HISTORY_KEY);
      if (stored) setHistory(JSON.parse(stored));
    } catch {
      /* ignore */
    }
    let sid = localStorage.getItem(SESSION_KEY);
    if (!sid) {
      sid = crypto.randomUUID();
      localStorage.setItem(SESSION_KEY, sid);
    }
    setSessionId(sid);
  }, []);

  const persist = useCallback((next: ResearchResult[]) => {
    setHistory(next);
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(next.slice(0, 12)));
    } catch {
      /* quota */
    }
  }, []);

  const mark = (id: string, state: Step["state"], detail?: string) =>
    setSteps((prev) =>
      prev
        ? prev.map((s) => (s.id === id ? { ...s, state, detail: detail ?? s.detail } : s))
        : prev,
    );

  const onRun = async () => {
    if (busy) return;
    if (question.trim().length < 5) {
      setError("Enter a clinical question of at least 5 characters.");
      return;
    }
    if (mode === "compare" && compareQuestion.trim().length < 5) {
      setError("Compare mode needs a second question to run head-to-head.");
      return;
    }
    setError(null);
    setBusy(true);
    setResult(null);
    setSteps(baseSteps(mode, filters.includeTrials));

    try {
      const next = await runResearchStream({
        question,
        mode,
        filters,
        sessionId,
        ...(mode === "compare" ? { compareQuestion } : {}),
        onActivity: (activity) => {
          mark(activity.step, "active", activity.detail);
          if (activity.step === "done") {
            mark("done", "done", activity.detail);
            return;
          }
          // Mark prior pending steps done once a later step becomes active.
          setSteps((prev) => {
            if (!prev) return prev;
            const activeIndex = prev.findIndex((s) => s.id === activity.step);
            return prev.map((s, index) => {
              if (s.state === "skipped") return s;
              if (index < activeIndex && s.state !== "done") {
                return { ...s, state: "done" as const };
              }
              if (s.id === activity.step) {
                return { ...s, state: "active" as const, detail: activity.detail };
              }
              return s;
            });
          });
        },
      });
      setResult(next);
      if (next.sessionId && next.sessionId !== sessionId) {
        setSessionId(next.sessionId);
        localStorage.setItem(SESSION_KEY, next.sessionId);
      }
      persist([next, ...history].slice(0, 12));
      setSteps(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Research failed. Please try again.");
      setSteps(null);
    } finally {
      setBusy(false);
    }
  };

  const onSimplify = async () => {
    if (!result) return;
    setSimplifying(true);
    setError(null);
    try {
      const simplified = await simplifyAnswer(result.question, result.synthesis.answer);
      const updated = { ...result, simplified };
      setResult(updated);
      persist(history.map((h) => (h.createdAt === updated.createdAt ? updated : h)));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not simplify the answer.");
    } finally {
      setSimplifying(false);
    }
  };

  const onSelectHistory = (item: ResearchResult) => {
    setResult(item);
    setQuestion(item.question);
    setCompareQuestion(item.compareQuestion ?? "");
    setMode(item.mode);
    setFilters(item.filters);
    setError(null);
    setSteps(null);
  };

  const onClearHistory = () => {
    persist([]);
    setResult(null);
    const sid = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, sid);
    setSessionId(sid);
  };

  const filterSummary = useMemo(() => {
    if (!result) return "";
    const f = result.filters;
    return [
      f.studyType === "any" ? "any study type" : f.studyType,
      f.yearFrom || f.yearTo ? `${f.yearFrom || "..."}-${f.yearTo || "..."}` : "all years",
      `max ${f.maxPapers} papers`,
      f.includeTrials ? "trials included" : "trials excluded",
      result.mode === "compare" ? "compare mode" : "standard mode",
    ].join(" · ");
  }, [result]);

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-4 px-5 py-5 sm:px-8">
          <span className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Stethoscope className="h-5 w-5" />
          </span>
          <div className="min-w-0">
            <h1 className="font-serif text-2xl leading-none">Evidentia</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Ask a clinical question. We search PubMed and ClinicalTrials.gov, then synthesize a
              cited, confidence-rated answer.
            </p>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-5 py-6 sm:px-8 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="space-y-6">
          <Composer
            question={question}
            setQuestion={setQuestion}
            compareQuestion={compareQuestion}
            setCompareQuestion={setCompareQuestion}
            mode={mode}
            setMode={setMode}
            filters={filters}
            setFilters={setFilters}
            busy={busy}
            onRun={onRun}
          />

          {error && (
            <div
              role="alert"
              className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {steps && <ProgressTimeline steps={steps} />}

          {!result && !steps && (
            <section className="panel flex flex-col items-center gap-2 px-6 py-14 text-center">
              <BookOpen className="h-6 w-6 text-muted-foreground" />
              <h2 className="font-serif text-xl">No results yet</h2>
              <p className="max-w-md text-sm text-muted-foreground">
                Ask a PICO-style question above. Each answer comes with numbered references you can
                trace back to the original abstracts.
              </p>
            </section>
          )}

          {result && (
            <>
              <AnswerPanel
                result={result}
                busy={busy}
                simplifying={simplifying}
                exporting={false}
                onSimplify={onSimplify}
                onExport={() => exportResultPdf(result)}
              />

              <section className="space-y-3">
                <h3 className="font-serif text-lg">References</h3>
                {result.references.length === 0 ? (
                  <EmptyNote>No references matched this question and filter set.</EmptyNote>
                ) : (
                  result.references.map((r) => <ReferenceCard key={`${r.pmid}-${r.index}`} reference={r} />)
                )}
              </section>

              {result.mode === "compare" && (
                <section className="space-y-3">
                  <h3 className="font-serif text-lg">Comparison references</h3>
                  {result.comparisonReferences.length === 0 ? (
                    <EmptyNote>No comparison references were found.</EmptyNote>
                  ) : (
                    result.comparisonReferences.map((r) => (
                      <ReferenceCard key={`cmp-${r.pmid}-${r.index}`} reference={r} />
                    ))
                  )}
                </section>
              )}

              <section className="space-y-3">
                <h3 className="font-serif text-lg">Ongoing clinical trials</h3>
                {result.trials.length === 0 ? (
                  <EmptyNote>
                    No ongoing trials to show. Either none matched or trials were excluded.
                  </EmptyNote>
                ) : (
                  result.trials.map((t) => <TrialCard key={t.nctId} trial={t} />)
                )}
              </section>

              <section className="space-y-3">
                <h3 className="font-serif text-lg">Similar cached abstracts</h3>
                {result.cached.length === 0 ? (
                  <EmptyNote>No similar abstracts from earlier searches yet.</EmptyNote>
                ) : (
                  result.cached.map((c) => <CachedCard key={c.pmid} item={c} />)
                )}
              </section>

              <section className="panel px-5 py-4 text-xs text-muted-foreground">
                <span className="label-caps">Search overview</span>
                <p className="mt-1.5">{filterSummary}</p>
                <p className="mt-1 font-mono">session {result.sessionId}</p>
              </section>
            </>
          )}

          <p className="rounded-md border border-caution/40 bg-caution/10 px-4 py-3 text-xs leading-5 text-foreground/80">
            <strong>Medical disclaimer:</strong> Evidentia summarizes published literature using AI
            and is for informational and research purposes only. It is not medical advice,
            diagnosis, or treatment, and it may miss or misread evidence. Always verify against the
            primary sources and apply clinical judgement.
          </p>
        </div>

        <HistorySidebar
          history={history}
          sessionId={sessionId}
          activeAt={result?.createdAt ?? null}
          onSelect={onSelectHistory}
          onClear={onClearHistory}
        />
      </main>
    </div>
  );
}
