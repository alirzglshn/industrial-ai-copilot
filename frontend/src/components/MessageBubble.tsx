import type { ChatMessage } from "../types";
import { Badge } from "./ui/Badge";
import { Spinner } from "./ui/Spinner";
import { CitationChip } from "./CitationChip";
import { ToolCallTrace } from "./ToolCallTrace";

function GroundingBadges({ result }: { result: NonNullable<ChatMessage["result"]> }) {
  if (result.insufficient_evidence) {
    return <Badge tone="warning">insufficient evidence — declined to guess</Badge>;
  }
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Badge tone={result.grounded ? "success" : "danger"}>
        {result.grounded ? "grounded" : "not grounded"}
      </Badge>
      <Badge tone="neutral">faithfulness {result.faithfulness.toFixed(2)}</Badge>
      {result.unsupported_pages.length > 0 && (
        <Badge tone="danger">invented pages: {result.unsupported_pages.join(", ")}</Badge>
      )}
    </div>
  );
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-2xl rounded-2xl px-4 py-3 ${
          isUser ? "bg-accent text-zinc-950" : "bg-zinc-900 text-zinc-100 ring-1 ring-inset ring-zinc-800"
        }`}
      >
        {!isUser && message.pipeline && (
          <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
            {message.pipeline === "agent" ? "Agent" : "Direct"}
          </div>
        )}

        <p className="whitespace-pre-wrap text-sm leading-relaxed">
          {message.text}
          {message.streaming && <Spinner className="ml-1.5 inline h-3.5 w-3.5 text-zinc-500" />}
        </p>

        {!isUser && message.toolCalls && <ToolCallTrace toolCalls={message.toolCalls} />}

        {!isUser && message.result && !message.streaming && (
          <div className="mt-2.5 space-y-2 border-t border-zinc-800 pt-2.5">
            <GroundingBadges result={message.result} />
            {message.result.citations.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {message.result.citations.map((citation, i) => (
                  <CitationChip key={i} citation={citation} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
