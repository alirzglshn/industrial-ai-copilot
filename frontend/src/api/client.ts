import type {
  ConversationDetail,
  ConversationSummary,
  DocumentSummary,
  DocumentUploadResponse,
  Pipeline,
  QueryResult,
} from "../types";

// paths stay relative, vite's dev proxy and nginx both forward these prefixes to the api

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(body.detail ?? "Request failed", response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}

export async function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/documents/upload", { method: "POST", body: form });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(body.detail ?? "Upload failed", response.status);
  }
  return response.json();
}

export function listDocuments(): Promise<DocumentSummary[]> {
  return request("/documents");
}

export function listConversations(): Promise<ConversationSummary[]> {
  return request("/conversations");
}

export function getConversation(id: string): Promise<ConversationDetail> {
  return request(`/conversations/${id}`);
}

export function deleteConversation(id: string): Promise<void> {
  return request(`/conversations/${id}`, { method: "DELETE" });
}

export function imageFileUrl(documentId: string, imageId: string): string {
  return `/documents/${documentId}/images/${imageId}/file`;
}

export function pagePreviewUrl(documentId: string, pageNumber: number): string {
  return `/documents/${documentId}/pages/${pageNumber}/preview`;
}

// streaming

export type StreamEvent =
  | { event: "token"; data: { text: string } }
  | { event: "tool_calls"; data: { tool_calls: string[] } }
  | { event: "result"; data: QueryResult };

interface StreamQueryOptions {
  question: string;
  documentId?: string | null;
  conversationId?: string | null;
  pipeline: Pipeline;
  signal?: AbortSignal;
}

// reading a post'd sse response as an async generator of parsed events
export async function* streamQuery(options: StreamQueryOptions): AsyncGenerator<StreamEvent> {
  const path = options.pipeline === "agent" ? "/agent/query/stream" : "/query/stream";
  const body = {
    question: options.question,
    document_id: options.documentId,
    conversation_id: options.conversationId,
  };

  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: options.signal,
  });

  if (!response.ok || !response.body) {
    const errBody = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(errBody.detail ?? "Streaming request failed", response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseFrame(frame);
      if (event) yield event;
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function parseFrame(frame: string): StreamEvent | null {
  const lines = frame.split("\n");
  const eventLine = lines.find((line) => line.startsWith("event: "));
  const dataLine = lines.find((line) => line.startsWith("data: "));
  if (!eventLine || !dataLine) return null;
  const event = eventLine.slice("event: ".length);
  const data = JSON.parse(dataLine.slice("data: ".length));
  return { event, data } as StreamEvent;
}
