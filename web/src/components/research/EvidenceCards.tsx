import { ExternalLink, FlaskConical, Layers } from "lucide-react";
import type { CachedAbstract, Reference, Trial } from "@/lib/research-types";

export function EmptyNote({ children }: { children: React.ReactNode }) {
  return <p className="empty-note">{children}</p>;
}

export function ReferenceCard({ reference }: { reference: Reference }) {
  const r = reference;
  return (
    <article id={`ref-${r.index}`} className="panel ref-card">
      <div className="ref-row">
        <span className="ref-index">{r.index}</span>
        <div className="ref-body">
          <h4>{r.title}</h4>
          <p className="ref-authors">{r.authors}</p>
          <div className="ref-meta">
            <span className="muted" style={{ fontStyle: "italic" }}>
              {r.journal} · {r.year}
            </span>
            <span className="chip">PMID {r.pmid}</span>
            <span className="chip chip-accent">{r.studyType}</span>
          </div>
          <p className="ref-summary">{r.summary}</p>
          {r.highlights.length > 0 && (
            <ul className="highlight-list">
              {r.highlights.map((h, i) => (
                <li key={i}>“{h}”</li>
              ))}
            </ul>
          )}
          <a href={r.url} target="_blank" rel="noreferrer" className="ext-link">
            View on PubMed <ExternalLink className="icon-sm" />
          </a>
        </div>
      </div>
    </article>
  );
}

export function TrialCard({ trial }: { trial: Trial }) {
  return (
    <article className="panel trial-card">
      <div className="trial-title-row">
        <FlaskConical className="icon text-signal" style={{ marginTop: "0.125rem" }} />
        <h4>{trial.title}</h4>
      </div>
      <div className="trial-meta">
        <span className="chip">{trial.nctId}</span>
        <span className="status-pill">{trial.status}</span>
        <span className="muted">{trial.phase}</span>
        {trial.enrollment !== null && <span className="muted">n = {trial.enrollment}</span>}
      </div>
      {trial.interventions.length > 0 && (
        <p className="trial-line">
          <span className="muted">Interventions: </span>
          {trial.interventions.join(", ")}
        </p>
      )}
      {trial.conditions.length > 0 && (
        <p className="trial-line">
          <span className="muted">Conditions: </span>
          {trial.conditions.join(", ")}
        </p>
      )}
      <p className="trial-line muted" style={{ fontSize: "0.75rem" }}>
        {trial.locations}
      </p>
      <a href={trial.url} target="_blank" rel="noreferrer" className="ext-link">
        View on ClinicalTrials.gov <ExternalLink className="icon-sm" />
      </a>
    </article>
  );
}

export function CachedCard({ item }: { item: CachedAbstract }) {
  return (
    <a href={item.url} target="_blank" rel="noreferrer" className="panel cached-card">
      <Layers className="icon muted" style={{ marginTop: "0.125rem" }} />
      <div className="cached-copy">
        <p>{item.title}</p>
        <p className="meta">
          PMID {item.pmid} · retrieved for “{item.question.slice(0, 60)}
          {item.question.length > 60 ? "…" : ""}”
        </p>
      </div>
      <span className="similarity">{(item.similarity * 100).toFixed(0)}%</span>
    </a>
  );
}
