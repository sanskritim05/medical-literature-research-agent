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
          <span key={i} className="cite-group">
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
                  className="cite-chip"
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
      ? "confidence-high"
      : level === "Moderate"
        ? "confidence-moderate"
        : "confidence-low";
  return (
    <span className={`confidence-pill ${tone}`}>
      {level} confidence
      <span className="mono" style={{ opacity: 0.8 }}>
        {score}/100
      </span>
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
    <section className="panel answer-panel">
      <div className="answer-header">
        <span className="label-caps">Synthesized answer</span>
        <h2>{result.question}</h2>
        {result.compareQuestion && (
          <p className="compare-note">Compared against: {result.compareQuestion}</p>
        )}
        <div className="confidence-row">
          <ConfidencePill level={synthesis.confidence.level} score={synthesis.confidence.score} />
          <p>{withoutEmDashes(synthesis.confidence.rationale)}</p>
        </div>
      </div>

      <div className="answer-body">
        {synthesis.answer
          .split(/\n{1,}/)
          .filter(Boolean)
          .map((para, i) => (
            <p key={i}>
              <CitedText text={para} />
            </p>
          ))}

        {synthesis.key_takeaways.length > 0 && (
          <div className="info-box">
            <span className="label-caps">Key takeaways</span>
            <ul className="takeaway-list">
              {synthesis.key_takeaways.map((t, i) => (
                <li key={i}>
                  <span className="takeaway-dot" />
                  <span>
                    <CitedText text={t} />
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {synthesis.plain_language_summary && (
          <div className="info-box accent">
            <span className="label-caps">Plain-language summary</span>
            <p style={{ margin: "0.375rem 0 0", fontSize: "0.875rem", lineHeight: 1.5 }}>
              {withoutEmDashes(synthesis.plain_language_summary)}
            </p>
          </div>
        )}

        {result.simplified && (
          <div className="info-box primary">
            <span className="label-caps">Simplified for patients</span>
            <div style={{ marginTop: "0.5rem", display: "grid", gap: "0.5rem" }}>
              {result.simplified
                .split(/\n+/)
                .filter(Boolean)
                .map((p, i) => (
                  <p key={i} style={{ margin: 0, fontSize: "0.875rem", lineHeight: 1.5 }}>
                    {withoutEmDashes(p)}
                  </p>
                ))}
            </div>
          </div>
        )}

        <div className="answer-actions">
          <Button variant="outline" onClick={onSimplify} disabled={busy || simplifying}>
            {simplifying ? <Loader2 className="icon spin" /> : <WandSparkles className="icon" />}
            Simplify this answer
          </Button>
          <Button variant="outline" onClick={onExport} disabled={busy || exporting}>
            <FileDown className="icon" />
            Export PDF
          </Button>
        </div>
      </div>
    </section>
  );
}
