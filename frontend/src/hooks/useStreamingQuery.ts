import { useCallback, useState } from "react";
import { ApiError, streamQuery } from "../api/client";
import type { ChatMessage, ConversationDetail, Pipeline } from "../types";

function fromHistory(detail: ConversationDetail): ChatMessage[] {
  return detail.messages.map((message) => ({
    id: message.id,
    role: message.role,
    text: message.text,
    pipeline: message.pipeline ?? undefined,
    toolCalls: message.tool_calls ?? undefined,
    result:
      message.role === "assistant"
        ? {
            answer: message.text,
            citations: message.citations ?? [],
            insufficient_evidence: message.insufficient_evidence ?? false,
            unsupported_pages: message.unsupported_pages ?? [],
            grounded: message.grounded ?? true,
            faithfulness: message.faithfulness ?? 1,
            tool_calls: message.tool_calls ?? [],
            conversation_id: detail.id,
          }
        : undefined,
  }));
}

export function useStreamingQuery() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startNew = useCallback(() => {
    setMessages([]);
    setConversationId(null);
    setError(null);
  }, []);

  const loadConversation = useCallback((detail: ConversationDetail) => {
    setConversationId(detail.id);
    setMessages(fromHistory(detail));
    setError(null);
  }, []);

  const ask = useCallback(
    async (question: string, pipeline: Pipeline, documentId: string | null) => {
      setBusy(true);
      setError(null);

      const userMessage: ChatMessage = { id: crypto.randomUUID(), role: "user", text: question };
      const assistantId = crypto.randomUUID();
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        text: "",
        pipeline,
        streaming: true,
      };
      setMessages((prev) => [...prev, userMessage, assistantMessage]);

      const patchAssistant = (patch: Partial<ChatMessage>) =>
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, ...patch } : m)));

      try {
        for await (const evt of streamQuery({ question, pipeline, documentId, conversationId })) {
          if (evt.event === "token") {
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, text: m.text + evt.data.text } : m))
            );
          } else if (evt.event === "tool_calls") {
            patchAssistant({ toolCalls: evt.data.tool_calls });
          } else if (evt.event === "result") {
            setConversationId(evt.data.conversation_id);
            patchAssistant({ streaming: false, text: evt.data.answer, result: evt.data });
          }
        }
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Something went wrong. Is the API running?");
        patchAssistant({ streaming: false });
      } finally {
        setBusy(false);
      }
    },
    [conversationId]
  );

  return { messages, conversationId, busy, error, ask, startNew, loadConversation };
}
