import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "outline" | "ghost";
type Size = "default" | "lg" | "sm";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  children: ReactNode;
}

export function Button({
  variant = "primary",
  size = "default",
  className = "",
  children,
  ...props
}: Props) {
  const classes = [
    "btn",
    variant === "primary" ? "btn-primary" : "",
    variant === "outline" ? "btn-outline" : "",
    variant === "ghost" ? "btn-ghost" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <button className={classes} {...props}>
      {children}
    </button>
  );
}
