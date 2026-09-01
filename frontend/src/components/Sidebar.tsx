import type { ConversationSummary, DocumentSummary } from "../types";
import { DocumentList } from "./DocumentList";
import { DocumentUpload } from "./DocumentUpload";
import { ConversationList } from "./ConversationList";

interface Props {
  documents: DocumentSummary[];
  uploading: boolean;
  onUpload: (file: File) => Promise<void>;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
}

export function Sidebar({
  documents,
  uploading,
  onUpload,
  conversations,
  activeConversationId,
  onSelectConversation,
  onDeleteConversation,
}: Props) {
  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-zinc-800 bg-zinc-950">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-4">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-accent text-sm font-bold text-zinc-950">
          IC
        </div>
        <span className="text-sm font-semibold text-zinc-100">Industrial AI Copilot</span>
      </div>

      <div className="flex-1 space-y-6 overflow-y-auto px-4 py-4">
        <section>
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">
            Manuals
          </h2>
          <DocumentUpload uploading={uploading} onUpload={onUpload} />
          <div className="mt-3">
            <DocumentList documents={documents} />
          </div>
        </section>

        <section>
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">
            History
          </h2>
          <ConversationList
            conversations={conversations}
            activeId={activeConversationId}
            onSelect={onSelectConversation}
            onDelete={onDeleteConversation}
          />
        </section>
      </div>

      <div className="border-t border-zinc-800 px-4 py-3 text-[11px] text-zinc-600">
        Runs entirely locally — no data leaves this machine.
      </div>
    </aside>
  );
}
