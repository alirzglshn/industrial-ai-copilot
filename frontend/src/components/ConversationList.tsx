import type { ConversationSummary } from "../types";

interface Props {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export function ConversationList({ conversations, activeId, onSelect, onDelete }: Props) {
  if (conversations.length === 0) {
    return <p className="px-1 text-xs text-zinc-600">No conversations yet.</p>;
  }

  return (
    <ul className="space-y-1">
      {conversations.map((conversation) => (
        <li
          key={conversation.id}
          className={`group flex items-center gap-1 rounded-lg px-2.5 py-2 text-sm ${
            conversation.id === activeId
              ? "bg-zinc-800 text-zinc-100"
              : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
          }`}
        >
          <button onClick={() => onSelect(conversation.id)} className="flex-1 truncate text-left">
            {conversation.title}
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(conversation.id);
            }}
            className="hidden shrink-0 text-zinc-600 hover:text-rose-400 group-hover:block"
            title="Delete conversation"
          >
            <svg className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm4-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </li>
      ))}
    </ul>
  );
}
