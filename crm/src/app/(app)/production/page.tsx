import Link from "next/link";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { scopeWhere } from "@/lib/rbac";
import { formatMoney, formatDate, relativeDays, pluralDays, humanise } from "@/lib/format";
import { PageHeader } from "@/components/ui";
import { PriorityBadge, PaymentBadge } from "@/components/badges";

export const dynamic = "force-dynamic";

// The stages a job actually passes through on the floor. DRAFT, cancelled and
// delivered work is deliberately absent — this board is only what is live.
const COLUMNS = [
  "CONFIRMED",
  "DESIGN",
  "PRINTING",
  "POST_PROCESSING",
  "QUALITY_CHECK",
  "READY",
  "SHIPPED",
] as const;

export default async function ProductionPage() {
  const user = await requireUser();

  const orders = await db.order.findMany({
    where: { AND: [scopeWhere(user), { status: { in: [...COLUMNS] } }] },
    orderBy: [{ priority: "desc" }, { dueDate: { sort: "asc", nulls: "last" } }],
    include: {
      customer: { select: { name: true } },
      assignedTo: { select: { name: true } },
      items: { select: { technology: true, material: true, quantity: true } },
    },
  });

  const byStatus = new Map<string, typeof orders>();
  for (const status of COLUMNS) byStatus.set(status, []);
  for (const order of orders) byStatus.get(order.status)?.push(order);

  return (
    <>
      <PageHeader
        title="Production board"
        subtitle={`${orders.length} live jobs · urgent first, then earliest deadline`}
      />

      <div className="scroll-x pb-4">
        <div className="flex min-w-max gap-3">
          {COLUMNS.map((status) => {
            const column = byStatus.get(status) ?? [];
            return (
              <section key={status} className="w-64 shrink-0">
                <div className="mb-2 flex items-center justify-between px-1">
                  <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-500">
                    {humanise(status)}
                  </h2>
                  <span className="rounded-full bg-ink-100 px-2 text-xs font-semibold tabular-nums text-ink-700">
                    {column.length}
                  </span>
                </div>

                <div className="space-y-2">
                  {column.length === 0 ? (
                    <p className="rounded-lg border border-dashed border-ink-300 px-3 py-6 text-center text-xs text-ink-500">
                      Empty
                    </p>
                  ) : (
                    column.map((order) => {
                      const days = relativeDays(order.dueDate);
                      const late = days !== null && days < 0;
                      const soon = days !== null && days >= 0 && days <= 2;
                      return (
                        <Link
                          key={order.id}
                          href={`/orders/${order.id}`}
                          className="card block p-3 transition-shadow hover:shadow-md"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <span className="text-sm font-semibold text-ink-900">{order.orderNo}</span>
                            <PriorityBadge value={order.priority} />
                          </div>
                          <p className="mt-1 truncate text-xs text-ink-700">{order.customer.name}</p>
                          <p className="mt-1 truncate text-xs text-ink-500">
                            {order.items
                              .map((item) =>
                                [item.technology, item.material].filter(Boolean).join(" "),
                              )
                              .filter(Boolean)
                              .join(", ") || `${order.items.length} item(s)`}
                          </p>
                          <div className="mt-2 flex flex-wrap items-center gap-1.5">
                            <PaymentBadge value={order.paymentStatus} />
                            <span className="text-xs tabular-nums text-ink-500">
                              {formatMoney(order.total, order.currency)}
                            </span>
                          </div>
                          <p
                            className={`mt-2 text-xs ${
                              late ? "font-semibold text-red-600" : soon ? "text-amber-600" : "text-ink-500"
                            }`}
                          >
                            {order.dueDate
                              ? late
                                ? `${pluralDays(Math.abs(days!))} late`
                                : `Due ${formatDate(order.dueDate)}`
                              : "No deadline set"}
                          </p>
                          <p className="mt-1 truncate text-xs text-ink-500">
                            {order.assignedTo?.name ?? "Unassigned"}
                          </p>
                        </Link>
                      );
                    })
                  )}
                </div>
              </section>
            );
          })}
        </div>
      </div>

      <p className="mt-2 text-xs text-ink-500">
        Open a job to move it to the next stage. Everyone watching that order is notified.
      </p>
    </>
  );
}
