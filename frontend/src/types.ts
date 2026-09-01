export interface DocumentSummary {
  id: string;
  filename: string;
  status: string;
  page_count: number;
  uploaded_at: string;
}

export interface DocumentUploadResponse extends DocumentSummary {
  chunk_count: number;
  image_count: number;
  indexed_chunks: number;
  indexed_images: number;
}

export interface Citation {
  kind: "text" | "image";
  document_id: string;
  page_number: number;
  chunk_id: string | null;
  image_id: string | null;
  image_path: string | null;
}

export interface QueryResult {
  answer: string;
  citations: Citation[];
  insufficient_evidence: boolean;
  unsupported_pages: number[];
  grounded: boolean;
  faithfulness: number;
  tool_calls: string[];
  conversation_id: string | null;
}

export type Pipeline = "fixed" | "agent";

export interface MessageOut {
  id: string;
  role: "user" | "assistant";
  text: string;
  created_at: string;
  pipeline: Pipeline | null;
  citations: Citation[] | null;
  insufficient_evidence: boolean | null;
  grounded: boolean | null;
  faithfulness: number | null;
  unsupported_pages: number[] | null;
  tool_calls: string[] | null;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  message_count: number;
}

export interface ConversationDetail extends ConversationSummary {
  messages: MessageOut[];
}

// a chat message as rendered client-side, assistant text streams in until result attaches
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  pipeline?: Pipeline;
  streaming?: boolean;
  toolCalls?: string[];
  result?: QueryResult;
}
