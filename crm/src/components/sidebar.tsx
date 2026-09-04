"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import type { Role } from "@prisma/client";
import { canManageUsers } from "@/lib/rbac";

const MAIN = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/leads", label: "Leads" },
  { href: "/orders", label: "Orders" },
  { href: "/production", label: "Production board" },
  { href: "/customers", label: "Customers" },
  { href: "/payments", label: "Payments" },
];

const ADMIN = [
  { href: "/admin/users", label: "Users & roles" },
  { href: "/admin/departments", label: "Departments" },
  { href: "/admin/branches", label: "Branches & franchises" },
  { href: "/admin/integrations", label: "Integrations" },
];

export function Sidebar({ role, unread }: { role: Role; unread: number }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const link = (item: { href: string; label: string }) => {
    const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
    return (
      <Link
        key={item.href}
        href={item.href}
        onClick={() => setOpen(false)}
        className={`block rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
          active ? "bg-white/10 text-white" : "text-ink-300 hover:bg-white/5 hover:text-white"
        }`}
      >
        {item.label}
      </Link>
    );
  };

  return (
    <>
      <div className="flex items-center justify-between bg-ink-950 px-4 py-3 lg:hidden">
        <span className="text-sm font-semibold text-white">3DIL CRM</span>
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          className="rounded-lg border border-white/20 px-3 py-1 text-sm text-white"
          aria-expanded={open}
        >
          Menu
        </button>
      </div>

      <aside
        className={`${open ? "block" : "hidden"} w-full shrink-0 bg-ink-950 px-3 pb-6 lg:block lg:w-60 lg:px-3 lg:py-4`}
      >
        <div className="mb-6 hidden items-center gap-2 px-2 lg:flex">
          <span className="rounded-lg bg-white p-1.5">
            <Image src="/logo.png" alt="" width={26} height={21} />
          </span>
          <span className="text-sm font-semibold leading-tight text-white">
            3DIL CREATION
            <span className="block text-xs font-normal text-ink-500">CRM</span>
          </span>
        </div>

        <nav className="space-y-1">{MAIN.map(link)}</nav>

        <Link
          href="/notifications"
          onClick={() => setOpen(false)}
          className="mt-1 flex items-center justify-between rounded-lg px-3 py-2 text-sm font-medium text-ink-300 hover:bg-white/5 hover:text-white"
        >
          Notifications
          {unread > 0 ? (
            <span className="rounded-full bg-brand-500 px-1.5 text-xs font-semibold text-white">
              {unread > 99 ? "99+" : unread}
            </span>
          ) : null}
        </Link>

        {canManageUsers(role) ? (
          <>
            <p className="mt-6 px-3 text-xs font-semibold uppercase tracking-wide text-ink-500">
              Administration
            </p>
            <nav className="mt-2 space-y-1">{ADMIN.map(link)}</nav>
          </>
        ) : null}
      </aside>
    </>
  );
}
