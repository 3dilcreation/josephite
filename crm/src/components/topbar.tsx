"use client";

import { useRouter } from "next/navigation";
import { ROLE_LABELS } from "@/lib/rbac";
import type { CurrentUser } from "@/lib/auth";

export function Topbar({ user, unread }: { user: CurrentUser; unread: number }) {
  const router = useRouter();

  async function signOut() {
    await fetch("/api/auth/logout", { method: "POST" });
    router.push("/login");
    router.refresh();
  }

  return (
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-ink-100 bg-white px-4 py-3 sm:px-6">
      <div className="min-w-0">
        <p className="truncate text-sm font-semibold text-ink-900">{user.name}</p>
        <p className="truncate text-xs text-ink-500">
          {ROLE_LABELS[user.role]}
          {unread > 0 ? ` · ${unread} unread` : ""}
        </p>
      </div>
      <button type="button" onClick={signOut} className="btn btn-ghost">
        Sign out
      </button>
    </header>
  );
}
