import { useEffect } from "react";
import { pagePreviewUrl } from "../api/client";
import { Button } from "./ui/Button";

interface Props {
  documentId: string;
  pageNumber: number;
  onClose: () => void;
}

export function PagePreviewModal({ documentId, pageNumber, onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6"
      onClick={onClose}
    >
      <div
        className="flex max-h-full max-w-3xl flex-col overflow-hidden rounded-xl bg-zinc-900 shadow-2xl ring-1 ring-zinc-800"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
          <span className="text-sm font-medium text-zinc-300">Page {pageNumber} — source</span>
          <Button variant="ghost" onClick={onClose}>
            Close
          </Button>
        </div>
        <div className="overflow-auto p-4">
          <img
            src={pagePreviewUrl(documentId, pageNumber)}
            alt={`Page ${pageNumber}`}
            className="w-full rounded-lg bg-white"
          />
        </div>
      </div>
    </div>
  );
}
