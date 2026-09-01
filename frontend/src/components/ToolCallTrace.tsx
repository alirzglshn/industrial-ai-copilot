import { useState } from "react";

export function ToolCallTrace({ toolCalls }: { toolCalls: string[] }) {
  const [expanded, setExpanded] = useState(false);
  if (toolCalls.length === 0) return null;

  return (
    <div className="mt-2 rounded-lg bg-zinc-900/60 ring-1 ring-inset ring-zinc-800">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-xs text-zinc-400 hover:text-zinc-200"
      >
        <svg
          className={`h-3 w-3 shrink-0 transition-transform ${expanded ? "rotate-90" : ""}`}
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path d="M6 4l8 6-8 6V4z" />
        </svg>
        agent used {toolCalls.length} tool{toolCalls.length === 1 ? "" : "s"}
      </button>
      {expanded && (
        <ul className="space-y-1 px-2.5 pb-2.5 font-mono text-[11px] text-zinc-500">
          {toolCalls.map((call, i) => (
            <li key={i} className="break-all">
              {call}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
