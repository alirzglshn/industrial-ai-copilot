import { useCallback, useEffect, useState } from "react";
import { deleteConversation, listConversations } from "../api/client";
import type { ConversationSummary } from "../types";

export function useConversations() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setConversations(await listConversations());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const remove = useCallback(
    async (id: string) => {
      await deleteConversation(id);
      await refresh();
    },
    [refresh]
  );

  return { conversations, loading, refresh, remove };
}
