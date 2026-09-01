import { getConversation } from "./api/client";
import { useConversations } from "./hooks/useConversations";
import { useDocuments } from "./hooks/useDocuments";
import { useStreamingQuery } from "./hooks/useStreamingQuery";
import { ChatPanel } from "./components/ChatPanel";
import { Sidebar } from "./components/Sidebar";

export default function App() {
  const { documents, uploading, upload } = useDocuments();
  const { conversations, refresh: refreshConversations, remove: removeConversation } =
    useConversations();
  const { messages, conversationId, busy, error, ask, startNew, loadConversation } =
    useStreamingQuery();

  const handleAsk = async (question: string, pipeline: "fixed" | "agent", documentId: string | null) => {
    await ask(question, pipeline, documentId);
    refreshConversations();
  };

  const handleSelectConversation = async (id: string) => {
    const detail = await getConversation(id);
    loadConversation(detail);
  };

  const handleDeleteConversation = async (id: string) => {
    await removeConversation(id);
    if (id === conversationId) startNew();
  };

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100">
      <Sidebar
        documents={documents}
        uploading={uploading}
        onUpload={upload}
        conversations={conversations}
        activeConversationId={conversationId}
        onSelectConversation={handleSelectConversation}
        onDeleteConversation={handleDeleteConversation}
      />
      <main className="flex-1 overflow-hidden">
        <ChatPanel
          messages={messages}
          busy={busy}
          error={error}
          documents={documents}
          onAsk={handleAsk}
          onNewConversation={startNew}
        />
      </main>
    </div>
  );
}
