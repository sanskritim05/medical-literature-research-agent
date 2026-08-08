import jsPDF from "jspdf";
import type { ResearchResult } from "./research-types";
import { withoutEmDashes } from "./utils";

export function exportResultPdf(result: ResearchResult) {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const margin = 48;
  const width = doc.internal.pageSize.getWidth() - margin * 2;
  const bottom = doc.internal.pageSize.getHeight() - margin;
  let y = margin;

  const write = (text: string, size: number, style: "normal" | "bold" | "italic" = "normal", gap = 6) => {
    doc.setFont("helvetica", style);
    doc.setFontSize(size);
    const lines = doc.splitTextToSize(withoutEmDashes(text), width) as string[];
    lines.forEach((line) => {
      if (y > bottom) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin, y);
      y += size * 1.35;
    });
    y += gap;
  };

  write("Clinical Evidence Report", 20, "bold", 2);
  write(new Date(result.createdAt).toLocaleString() + `  ·  session ${result.sessionId}`, 9, "italic", 12);

  write("Question", 12, "bold", 2);
  write(result.question, 11);
  if (result.compareQuestion) {
    write("Comparison question", 12, "bold", 2);
    write(result.compareQuestion, 11);
  }

  write(
    `Confidence: ${result.synthesis.confidence.level} (${result.synthesis.confidence.score}/100) - ${result.synthesis.confidence.rationale}`,
    10,
    "italic",
    10,
  );

  write("Synthesized answer", 12, "bold", 2);
  result.synthesis.answer.split(/\n+/).filter(Boolean).forEach((p) => write(p, 11));

  if (result.synthesis.plain_language_summary) {
    write("Plain-language summary", 12, "bold", 2);
    write(result.synthesis.plain_language_summary, 11);
  }
  if (result.simplified) {
    write("Simplified for patients", 12, "bold", 2);
    write(result.simplified, 11);
  }

  const refBlock = (title: string, refs: typeof result.references) => {
    if (!refs.length) return;
    write(title, 12, "bold", 4);
    refs.forEach((r) => {
      write(`[${r.index}] ${r.title}`, 10, "bold", 2);
      write(`${r.authors} - ${r.journal} (${r.year}) · PMID ${r.pmid} · ${r.studyType}`, 9, "italic", 2);
      write(r.summary, 9);
    });
  };
  refBlock("References", result.references);
  refBlock("Comparison references", result.comparisonReferences);

  if (result.trials.length) {
    write("Ongoing clinical trials", 12, "bold", 4);
    result.trials.forEach((t) => {
      write(`${t.nctId} - ${t.title}`, 10, "bold", 2);
      write(`${t.status} · ${t.phase} · ${t.interventions.join(", ")}`, 9, "italic", 6);
    });
  }

  write(
    "Disclaimer: This report is an AI-generated synthesis of published literature for informational purposes only. It is not medical advice, diagnosis, or treatment. Verify all findings against primary sources and clinical judgement.",
    8,
    "italic",
  );

  doc.save(`evidence-report-${result.createdAt}.pdf`);
}
