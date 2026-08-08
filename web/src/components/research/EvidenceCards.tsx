import { ExternalLink, FlaskConical, Layers } from "lucide-react";
import type { CachedAbstract, Reference, Trial } from "@/lib/research-types";

export function EmptyNote({ children }: { children: React.ReactNode }) {
  return (
    <p className="rounded-md border border-dashed border-border bg-surface px-4 py-6 text-center text-sm text-muted-foreground">
      {children}
    </p>
  );
}

export function ReferenceCard({ reference }: { reference: Reference }) {
  const r = reference;
  return (
    <article
      id={`ref-${r.index}`}
      className="panel scroll-mt-24 p-4 transition-colors target:border-primary"
    >
      <div className="flex gap-3">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary font-mono text-xs font-semibold text-primary-foreground">
          {r.index}
        </span>
        <div className="min-w-0 flex-1">
          <h4 className="font-serif text-base leading-snug">{r.title}</h4>
          <p className="mt-1 text-xs text-muted-foreground">{r.authors}</p>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
            <span className="italic text-muted-foreground">
              {r.journal} · {r.year}
            </span>
            <span className="rounded border border-border bg-surface px-1.5 py-0.5 font-mono">
              PMID {r.pmid}
            </span>
            <span className="rounded border border-accent bg-accent/50 px-1.5 py-0.5 text-accent-foreground">
              {r.studyType}
            </span>
          </div>
          <p className="mt-3 text-sm leading-6 text-foreground/90">{r.summary}</p>
          {r.highlights.length > 0 && (
            <ul className="mt-3 space-y-1.5 border-l-2 border-signal/50 pl-3">
              {r.highlights.map((h, i) => (
                <li key={i} className="text-sm italic leading-6 text-muted-foreground">
                  “{h}”
                </li>
              ))}
            </ul>
          )}
          <a
            href={r.url}
            target="_blank"
            rel="noreferrer"
            className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
          >
            View on PubMed <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      </div>
    </article>
  );
}

export function TrialCard({ trial }: { trial: Trial }) {
  return (
    <article className="panel p-4">
      <div className="flex items-start gap-2">
        <FlaskConical className="mt-0.5 h-4 w-4 shrink-0 text-signal" />
        <h4 className="font-serif text-base leading-snug">{trial.title}</h4>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        <span className="rounded border border-border bg-surface px-1.5 py-0.5 font-mono">
          {trial.nctId}
        </span>
        <span className="rounded-full border border-signal/40 bg-signal/10 px-2 py-0.5 capitalize text-signal">
          {trial.status}
        </span>
        <span className="text-muted-foreground">{trial.phase}</span>
        {trial.enrollment !== null && (
          <span className="text-muted-foreground">n = {trial.enrollment}</span>
        )}
      </div>
      {trial.interventions.length > 0 && (
        <p className="mt-2 text-sm">
          <span className="text-muted-foreground">Interventions: </span>
          {trial.interventions.join(", ")}
        </p>
      )}
      {trial.conditions.length > 0 && (
        <p className="mt-1 text-sm">
          <span className="text-muted-foreground">Conditions: </span>
          {trial.conditions.join(", ")}
        </p>
      )}
      <p className="mt-1 text-xs text-muted-foreground">{trial.locations}</p>
      <a
        href={trial.url}
        target="_blank"
        rel="noreferrer"
        className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
      >
        View on ClinicalTrials.gov <ExternalLink className="h-3 w-3" />
      </a>
    </article>
  );
}

export function CachedCard({ item }: { item: CachedAbstract }) {
  return (
    <a
      href={item.url}
      target="_blank"
      rel="noreferrer"
      className="panel flex items-start gap-3 p-3 transition-colors hover:border-ring"
    >
      <Layers className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
      <div className="min-w-0">
        <p className="text-sm leading-snug">{item.title}</p>
        <p className="mt-1 text-xs text-muted-foreground">
          PMID {item.pmid} · retrieved for “{item.question.slice(0, 60)}
          {item.question.length > 60 ? "…" : ""}”
        </p>
      </div>
      <span className="ml-auto shrink-0 rounded-full border border-border bg-surface px-2 py-0.5 font-mono text-xs">
        {(item.similarity * 100).toFixed(0)}%
      </span>
    </a>
  );
}
