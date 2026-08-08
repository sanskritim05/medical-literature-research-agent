import { Button } from "@/components/ui/button";
import { FileDown, Loader2, WandSparkles } from "lucide-react";
import type { ResearchResult } from "@/lib/research-types";
import { withoutEmDashes } from "@/lib/utils";

function CitedText({ text }: { text: string }) {
  const parts = withoutEmDashes(text).split(/(\[\d+(?:\s*,\s*\d+)*\])/g);
  return (
    <>
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+(?:\s*,\s*\d+)*)\]$/);
        if (!m) return <span key={i}>{part}</span>;
        return (
          <span key={i} className="inline-flex gap-0.5 align-baseline">
            {m[1]!.split(",").map((n) => {
              const num = n.trim();
              return (
                <a
                  key={num}
                  href={`#ref-${num}`}
                  onClick={(e) => {
                    e.preventDefault();
                    document
                      .getElementById(`ref-${num}`)
                      ?.scrollIntoView({ behavior: "smooth", block: "center" });
                  }}
                  className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full border border-primary/30 bg-accent px-1.5 font-mono text-[11px] font-semibold text-primary transition-colors hover:bg-primary hover:text-primary-foreground"
                >
                  {num}
                </a>
              );
            })}
          </span>
        );
      })}
    </>
  );
}

function ConfidencePill({ level, score }: { level: string; score: number }) {
  const tone =
    level === "High"
      ? "border-signal/40 bg-signal/12 text-signal"
      : level === "Moderate"
        ? "border-caution/50 bg-caution/15 text-caution"
        : "border-destructive/40 bg-destructive/10 text-destructive";
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold ${tone}`}
    >
      {level} confidence
      <span className="font-mono opacity-80">{score}/100</span>
    </span>
  );
}

interface Props {
  result: ResearchResult;
  busy: boolean;
  simplifying: boolean;
  exporting: boolean;
  onSimplify: () => void;
  onExport: () => void;
}

export function AnswerPanel({ result, busy, simplifying, exporting, onSimplify, onExport }: Props) {
  const { synthesis } = result;
  return (
    <section className="panel overflow-hidden">
      <div className="border-b border-border bg-surface px-5 py-4 sm:px-6">
        <span className="label-caps">Synthesized answer</span>
        <h2 className="mt-1 font-serif text-xl leading-snug">{result.question}</h2>
        {result.compareQuestion && (
          <p className="mt-1 text-sm text-muted-foreground">
            Compared against: {result.compareQuestion}
          </p>
        )}
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <ConfidencePill level={synthesis.confidence.level} score={synthesis.confidence.score} />
          <p className="max-w-2xl text-xs text-muted-foreground">
            {withoutEmDashes(synthesis.confidence.rationale)}
          </p>
        </div>
      </div>

      <div className="space-y-4 px-5 py-5 sm:px-6">
        {synthesis.answer.split(/\n{1,}/).filter(Boolean).map((para, i) => (
          <p key={i} className="text-[15px] leading-7">
            <CitedText text={para} />
          </p>
        ))}

        {synthesis.key_takeaways.length > 0 && (
          <div className="rounded-md border border-border bg-surface p-4">
            <span className="label-caps">Key takeaways</span>
            <ul className="mt-2 space-y-1.5">
              {synthesis.key_takeaways.map((t, i) => (
                <li key={i} className="flex gap-2 text-sm leading-6">
                  <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                  <span>
                    <CitedText text={t} />
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {synthesis.plain_language_summary && (
          <div className="rounded-md border border-accent bg-accent/40 p-4">
            <span className="label-caps">Plain-language summary</span>
            <p className="mt-1.5 text-sm leading-6">{withoutEmDashes(synthesis.plain_language_summary)}</p>
          </div>
        )}

        {result.simplified && (
          <div className="rounded-md border border-primary/25 bg-card p-4">
            <span className="label-caps">Simplified for patients</span>
            <div className="mt-2 space-y-2">
              {result.simplified.split(/\n+/).filter(Boolean).map((p, i) => (
                <p key={i} className="text-sm leading-6">
                  {withoutEmDashes(p)}
                </p>
              ))}
            </div>
          </div>
        )}

        <div className="flex flex-wrap gap-2 border-t border-border pt-4">
          <Button variant="outline" onClick={onSimplify} disabled={busy || simplifying} className="gap-2">
            {simplifying ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <WandSparkles className="h-4 w-4" />
            )}
            Simplify this answer
          </Button>
          <Button variant="outline" onClick={onExport} disabled={busy || exporting} className="gap-2">
            <FileDown className="h-4 w-4" />
            Export PDF
          </Button>
        </div>
      </div>
    </section>
  );
}
