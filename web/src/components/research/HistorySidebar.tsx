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
    <aside className="panel sidebar">
      <div className="sidebar-top">
        <span className="label-caps">
          <History className="icon-sm" /> Recent questions
        </span>
        {history.length > 0 && (
          <Button variant="ghost" size="sm" onClick={onClear}>
            <Trash2 className="icon-sm" /> Clear
          </Button>
        )}
      </div>

      <div className="history-list">
        {history.length === 0 ? (
          <p className="history-empty">
            Your last 12 questions are kept in this browser so you can revisit results instantly.
          </p>
        ) : (
          history.map((h) => (
            <button
              key={h.createdAt}
              type="button"
              onClick={() => onSelect(h)}
              className={`history-item${activeAt === h.createdAt ? " active" : ""}`}
            >
              <p>{h.question}</p>
              <p className="meta">
                {new Date(h.createdAt).toLocaleString()} ·{" "}
                {h.mode === "compare" ? "Compare" : "Standard"} · {h.references.length} refs
              </p>
            </button>
          ))
        )}
      </div>

      <p className="session-foot">session {sessionId.slice(0, 12)}</p>
    </aside>
  );
}
