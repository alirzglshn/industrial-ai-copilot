import { useEffect, useRef, useState } from "react";
import type { ChatMessage, DocumentSummary, Pipeline } from "../types";
import { Button } from "./ui/Button";
import { MessageBubble } from "./MessageBubble";
import { PipelineToggle } from "./PipelineToggle";

interface Props {
  messages: ChatMessage[];
  busy: boolean;
  error: string | null;
  documents: DocumentSummary[];
  onAsk: (question: string, pipeline: Pipeline, documentId: string | null) => void;
  onNewConversation: () => void;
}

export function ChatPanel({ messages, busy, error, documents, onAsk, onNewConversation }: Props) {
  const [question, setQuestion] = useState("");
  const [pipeline, setPipeline] = useState<Pipeline>("fixed");
  const [documentId, setDocumentId] = useState<string>("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const submit = () => {
    const trimmed = question.trim();
    if (!trimmed || busy) return;
    onAsk(trimmed, pipeline, documentId || null);
    setQuestion("");
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-zinc-800 px-6 py-4">
        <div>
          <h1 className="text-sm font-semibold text-zinc-100">Ask the manuals</h1>
          <p className="text-xs text-zinc-500">
            {pipeline === "agent"
              ? "The agent decides which tools it needs — search, page lookup, calculator, or metadata."
              : "Retrieves relevant passages and diagrams, then answers only from what it finds."}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <PipelineToggle value={pipeline} onChange={setPipeline} />
          <Button variant="ghost" onClick={onNewConversation}>
            New chat
          </Button>
        </div>
      </header>

      <div className="flex-1 space-y-4 overflow-y-auto px-6 py-6">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center text-sm text-zinc-600">
            {documents.length === 0
              ? "Upload a manual to get started."
              : "Ask a question about an uploaded manual."}
          </div>
        )}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {error && (
          <div className="rounded-lg bg-rose-950/50 px-3 py-2 text-sm text-rose-400 ring-1 ring-inset ring-rose-900">
            {error}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-zinc-800 p-4">
        <div className="mb-2 flex items-center gap-2">
          <label className="text-xs text-zinc-500">Scope:</label>
          <select
            value={documentId}
            onChange={(e) => setDocumentId(e.target.value)}
            className="rounded-md bg-zinc-900 px-2 py-1 text-xs text-zinc-300 ring-1 ring-inset ring-zinc-800 focus:outline-none focus:ring-accent"
          >
            <option value="">All manuals</option>
            {documents.map((doc) => (
              <option key={doc.id} value={doc.id}>
                {doc.filename}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-end gap-2">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder="Ask a question about an uploaded manual…"
            className="max-h-40 flex-1 resize-none rounded-lg bg-zinc-900 px-3 py-2.5 text-sm text-zinc-100 placeholder-zinc-600 ring-1 ring-inset ring-zinc-800 focus:outline-none focus:ring-2 focus:ring-accent"
          />
          <Button variant="primary" onClick={submit} disabled={busy || !question.trim()}>
            {busy ? "Answering…" : "Ask"}
          </Button>
        </div>
      </div>
    </div>
  );
}
