import type { TextareaHTMLAttributes } from "react";

interface Props extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  compare?: boolean;
}

export function Textarea({ className = "", compare = false, ...props }: Props) {
  return (
    <textarea
      className={["textarea", compare ? "compare" : "", className].filter(Boolean).join(" ")}
      {...props}
    />
  );
}
