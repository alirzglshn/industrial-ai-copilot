import type { Pipeline } from "../types";

const OPTIONS: { value: Pipeline; label: string; hint: string }[] = [
  { value: "fixed", label: "Direct", hint: "Always search, then answer" },
  { value: "agent", label: "Agent", hint: "Decides which tools it needs" },
];

export function PipelineToggle({
  value,
  onChange,
}: {
  value: Pipeline;
  onChange: (value: Pipeline) => void;
}) {
  return (
    <div className="inline-flex rounded-lg bg-zinc-900 p-1 ring-1 ring-inset ring-zinc-800">
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          onClick={() => onChange(option.value)}
          title={option.hint}
          className={`rounded-md px-3 py-1 text-sm font-medium transition-colors ${
            value === option.value
              ? "bg-accent text-zinc-950"
              : "text-zinc-400 hover:text-zinc-200"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
