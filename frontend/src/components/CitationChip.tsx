import { useState } from "react";
import { imageFileUrl } from "../api/client";
import type { Citation } from "../types";
import { PagePreviewModal } from "./PagePreviewModal";

export function CitationChip({ citation }: { citation: Citation }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-1.5 rounded-md bg-zinc-800 px-2 py-1 text-xs text-zinc-300 ring-1 ring-inset ring-zinc-700 transition-colors hover:bg-zinc-700 hover:text-zinc-100"
        title="View source page"
      >
        {citation.kind === "image" && citation.image_id ? (
          <img
            src={imageFileUrl(citation.document_id, citation.image_id)}
            alt=""
            className="h-4 w-4 rounded-sm object-cover"
          />
        ) : (
          <svg className="h-3.5 w-3.5 text-zinc-500" viewBox="0 0 20 20" fill="currentColor">
            <path d="M4 2a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2V7.414A2 2 0 0017.414 6L14 2.586A2 2 0 0012.586 2H4z" />
          </svg>
        )}
        page {citation.page_number}
      </button>
      {open && (
        <PagePreviewModal
          documentId={citation.document_id}
          pageNumber={citation.page_number}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}
