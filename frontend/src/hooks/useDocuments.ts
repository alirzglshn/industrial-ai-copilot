import { useCallback, useEffect, useState } from "react";
import { ApiError, listDocuments, uploadDocument } from "../api/client";
import type { DocumentSummary } from "../types";

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setDocuments(await listDocuments());
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load documents");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const upload = useCallback(
    async (file: File) => {
      setUploading(true);
      setError(null);
      try {
        await uploadDocument(file);
        await refresh();
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Upload failed");
        throw err;
      } finally {
        setUploading(false);
      }
    },
    [refresh]
  );

  return { documents, loading, uploading, error, refresh, upload };
}
