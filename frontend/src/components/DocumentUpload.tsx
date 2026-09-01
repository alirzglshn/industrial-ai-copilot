import { useRef, useState } from "react";
import { Button } from "./ui/Button";
import { Spinner } from "./ui/Spinner";

interface Props {
  uploading: boolean;
  onUpload: (file: File) => Promise<void>;
}

export function DocumentUpload({ uploading, onUpload }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = (file: File | undefined) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) return;
    onUpload(file).catch(() => {});
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFile(e.dataTransfer.files[0]);
      }}
      className={`rounded-lg border border-dashed p-4 text-center transition-colors ${
        dragging ? "border-accent bg-accent/5" : "border-zinc-800"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      {uploading ? (
        <div className="flex items-center justify-center gap-2 py-2 text-sm text-zinc-400">
          <Spinner /> Ingesting…
        </div>
      ) : (
        <>
          <p className="mb-2 text-xs text-zinc-500">Drop a PDF manual here, or</p>
          <Button variant="secondary" onClick={() => inputRef.current?.click()}>
            Choose file
          </Button>
        </>
      )}
    </div>
  );
}
