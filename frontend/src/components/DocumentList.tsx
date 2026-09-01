import type { DocumentSummary } from "../types";
import { Badge } from "./ui/Badge";

const STATUS_TONE: Record<string, "success" | "warning" | "danger" | "neutral"> = {
  indexed: "success",
  parsed: "warning",
  parsing: "neutral",
  failed: "danger",
};

export function DocumentList({ documents }: { documents: DocumentSummary[] }) {
  if (documents.length === 0) {
    return <p className="px-1 text-xs text-zinc-600">No manuals uploaded yet.</p>;
  }

  return (
    <ul className="space-y-1.5">
      {documents.map((doc) => (
        <li
          key={doc.id}
          className="rounded-lg px-2.5 py-2 ring-1 ring-inset ring-zinc-800/60 hover:ring-zinc-700"
        >
          <div className="truncate text-sm text-zinc-200" title={doc.filename}>
            {doc.filename}
          </div>
          <div className="mt-1 flex items-center gap-1.5">
            <Badge tone={STATUS_TONE[doc.status] ?? "neutral"}>{doc.status}</Badge>
            <span className="text-[11px] text-zinc-600">{doc.page_count} pages</span>
          </div>
        </li>
      ))}
    </ul>
  );
}
