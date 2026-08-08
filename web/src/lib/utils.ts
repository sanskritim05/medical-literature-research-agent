import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function withoutEmDashes(text: string): string {
  return text.replace(/[\u2014\u2013]/g, "-");
}
