import { Check, Loader2, Circle } from "lucide-react";

export interface Step {
  id: string;
  label: string;
  detail: string;
  state: "pending" | "active" | "done" | "skipped";
}

export function ProgressTimeline({ steps }: { steps: Step[] }) {
  return (
    <section className="panel p-5 sm:p-6">
      <div className="mb-4 flex items-center gap-2">
        <Loader2 className="h-4 w-4 animate-spin text-primary" />
        <span className="font-serif text-lg">Tracing through the evidence…</span>
      </div>
      <ol className="space-y-3">
        {steps.map((s) => (
          <li key={s.id} className="flex items-start gap-3">
            <span className="mt-0.5">
              {s.state === "done" ? (
                <Check className="h-4 w-4 text-signal" />
              ) : s.state === "active" ? (
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
              ) : (
                <Circle className="h-4 w-4 text-muted-foreground/40" />
              )}
            </span>
            <div className="min-w-0">
              <p
                className={`text-sm font-medium ${
                  s.state === "pending" || s.state === "skipped"
                    ? "text-muted-foreground"
                    : "text-foreground"
                }`}
              >
                {s.label}
                {s.state === "skipped" && (
                  <span className="ml-2 text-xs text-muted-foreground">skipped</span>
                )}
              </p>
              <p className="text-xs text-muted-foreground">{s.detail}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
