import { History, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ResearchResult } from "@/lib/research-types";

interface Props {
  history: ResearchResult[];
  sessionId: string;
  activeAt: number | null;
  onSelect: (item: ResearchResult) => void;
  onClear: () => void;
}

export function HistorySidebar({ history, sessionId, activeAt, onSelect, onClear }: Props) {
  return (
    <aside className="panel sticky top-6 flex max-h-[calc(100vh-3rem)] flex-col p-4">
      <div className="flex items-center justify-between">
        <span className="label-caps flex items-center gap-1.5">
          <History className="h-3.5 w-3.5" /> Recent questions
        </span>
        {history.length > 0 && (
          <Button variant="ghost" size="sm" className="h-7 gap-1 px-2 text-xs" onClick={onClear}>
            <Trash2 className="h-3 w-3" /> Clear
          </Button>
        )}
      </div>

      <div className="mt-3 min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
        {history.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Your last 12 questions are kept in this browser so you can revisit results instantly.
          </p>
        ) : (
          history.map((h) => (
            <button
              key={h.createdAt}
              type="button"
              onClick={() => onSelect(h)}
              className={`w-full rounded-md border px-3 py-2 text-left transition-colors ${
                activeAt === h.createdAt
                  ? "border-primary bg-accent/50"
                  : "border-border bg-surface hover:border-ring"
              }`}
            >
              <p className="line-clamp-3 text-sm leading-snug">{h.question}</p>
              <p className="mt-1 text-[11px] text-muted-foreground">
                {new Date(h.createdAt).toLocaleString()} ·{" "}
                {h.mode === "compare" ? "Compare" : "Standard"} · {h.references.length} refs
              </p>
            </button>
          ))
        )}
      </div>

      <p className="mt-3 border-t border-border pt-3 font-mono text-[11px] text-muted-foreground">
        session {sessionId.slice(0, 12)}
      </p>
    </aside>
  );
}
