import Link from "next/link";

export function PageHeader({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink-900">{title}</h1>
        {subtitle ? <p className="mt-1 text-sm text-ink-500">{subtitle}</p> : null}
      </div>
      {action}
    </div>
  );
}

export function StatCard({
  label,
  value,
  hint,
  href,
  tone = "neutral",
}: {
  label: string;
  value: string;
  hint?: string;
  href?: string;
  tone?: "neutral" | "warn" | "danger" | "good";
}) {
  const accent = {
    neutral: "text-ink-900",
    warn: "text-amber-600",
    danger: "text-red-600",
    good: "text-emerald-600",
  }[tone];

  const body = (
    <div className="card h-full p-4 transition-shadow hover:shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-ink-500">{label}</p>
      <p className={`mt-2 text-2xl font-semibold tabular-nums ${accent}`}>{value}</p>
      {hint ? <p className="mt-1 text-xs text-ink-500">{hint}</p> : null}
    </div>
  );

  return href ? <Link href={href}>{body}</Link> : body;
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="px-4 py-14 text-center">
      <p className="text-sm font-medium text-ink-700">{title}</p>
      {hint ? <p className="mt-1 text-sm text-ink-500">{hint}</p> : null}
    </div>
  );
}

export function Th({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <th
      className={`whitespace-nowrap px-3 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-ink-500 ${className}`}
    >
      {children}
    </th>
  );
}

export function Td({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-3 py-2.5 align-middle text-sm text-ink-800 ${className}`}>{children}</td>;
}
