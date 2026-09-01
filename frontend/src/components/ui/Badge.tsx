import type { ReactNode } from "react";

type Tone = "neutral" | "success" | "danger" | "warning" | "accent";

const TONE_CLASSES: Record<Tone, string> = {
  neutral: "bg-zinc-800 text-zinc-300 ring-zinc-700",
  success: "bg-emerald-950 text-emerald-400 ring-emerald-800",
  danger: "bg-rose-950 text-rose-400 ring-rose-800",
  warning: "bg-amber-950 text-amber-400 ring-amber-800",
  accent: "bg-indigo-950 text-indigo-300 ring-indigo-800",
};

export function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${TONE_CLASSES[tone]}`}
    >
      {children}
    </span>
  );
}
