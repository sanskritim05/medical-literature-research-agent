import { Check, Loader2, Circle } from "lucide-react";

export interface Step {
  id: string;
  label: string;
  detail: string;
  state: "pending" | "active" | "done" | "skipped";
}

export function ProgressTimeline({ steps }: { steps: Step[] }) {
  return (
    <section className="panel progress">
      <div className="progress-title">
        <Loader2 className="icon spin text-primary" />
        <span>Tracing through the evidence…</span>
      </div>
      <ol className="progress-list">
        {steps.map((s) => (
          <li key={s.id} className="progress-item">
            <span>
              {s.state === "done" ? (
                <Check className="icon text-signal" />
              ) : s.state === "active" ? (
                <Loader2 className="icon spin text-primary" />
              ) : (
                <Circle className="icon text-muted-soft" />
              )}
            </span>
            <div>
              <p
                className={`progress-label${
                  s.state === "pending" || s.state === "skipped" ? " pending" : ""
                }`}
              >
                {s.label}
                {s.state === "skipped" && <span className="skip-tag">skipped</span>}
              </p>
              <p className="progress-detail">{s.detail}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
