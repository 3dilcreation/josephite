"use client";

import { useState } from "react";

/** Secrets stay hidden until asked for, so a shared screen doesn't leak them. */
export function SecretReveal({ endpoint, secret }: { endpoint: string; secret: string }) {
  const [shown, setShown] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);

  async function copy(value: string, label: string) {
    await navigator.clipboard.writeText(value);
    setCopied(label);
    setTimeout(() => setCopied(null), 1500);
  }

  return (
    <div className="space-y-1 text-xs">
      <div className="flex items-center gap-2">
        <code className="max-w-[18rem] truncate rounded bg-ink-50 px-1.5 py-0.5">{endpoint}</code>
        <button type="button" onClick={() => copy(endpoint, "url")} className="font-medium text-brand-600">
          {copied === "url" ? "Copied" : "Copy"}
        </button>
      </div>
      <div className="flex items-center gap-2">
        <code className="max-w-[18rem] truncate rounded bg-ink-50 px-1.5 py-0.5">
          {shown ? secret : "••••••••••••••••"}
        </code>
        <button type="button" onClick={() => setShown((value) => !value)} className="font-medium text-brand-600">
          {shown ? "Hide" : "Show"}
        </button>
        <button type="button" onClick={() => copy(secret, "secret")} className="font-medium text-brand-600">
          {copied === "secret" ? "Copied" : "Copy"}
        </button>
      </div>
    </div>
  );
}
