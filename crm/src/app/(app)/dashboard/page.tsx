import Link from "next/link";
import type { Prisma } from "@prisma/client";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { scopeWhere } from "@/lib/rbac";
import { formatMoney, formatDate, toNumber, humanise } from "@/lib/format";
import { PageHeader, StatCard, EmptyState, Th, Td } from "@/components/ui";
import { OrderStatusBadge, PaymentBadge, PriorityBadge, LeadStatusBadge } from "@/components/badges";

export const dynamic = "force-dynamic";

const OPEN_LEAD_STATUSES = ["NEW", "CONTACTED", "QUALIFIED", "QUOTED", "NEGOTIATION"] as const;
const LIVE_ORDER_STATUSES = [
  "CONFIRMED",
  "DESIGN",
  "PRINTING",
  "POST_PROCESSING",
  "QUALITY_CHECK",
  "READY",
] as const;

export default async function DashboardPage() {
  const user = await requireUser();
  const scope = scopeWhere(user);
  // Every count below is `scope AND <filter>`; spreading would drop the scope's OR.
  const scoped = (extra: Prisma.OrderWhereInput = {}): Prisma.OrderWhereInput => ({ AND: [scope, extra] });
  const scopedLead = (extra: Prisma.LeadWhereInput = {}): Prisma.LeadWhereInput => ({
    AND: [scope as Prisma.LeadWhereInput, extra],
  });

  const startOfMonth = new Date();
  startOfMonth.setDate(1);
  startOfMonth.setHours(0, 0, 0, 0);

  const in3Days = new Date();
  in3Days.setDate(in3Days.getDate() + 3);

  const [
    openLeads,
    liveOrders,
    dueSoon,
    overdueOrders,
    unpaid,
    monthRevenue,
    recentLeads,
    recentOrders,
    byStatus,
  ] = await Promise.all([
    db.lead.count({ where: scopedLead({ status: { in: [...OPEN_LEAD_STATUSES] } }) }),
    db.order.count({ where: scoped({ status: { in: [...LIVE_ORDER_STATUSES] } }) }),
    db.order.count({
      where: scoped({
        status: { in: [...LIVE_ORDER_STATUSES] },
        dueDate: { gte: new Date(), lte: in3Days },
      }),
    }),
    db.order.count({
      where: scoped({ status: { in: [...LIVE_ORDER_STATUSES] }, dueDate: { lt: new Date() } }),
    }),
    db.order.aggregate({
      where: scoped({ paymentStatus: { in: ["UNPAID", "PARTIAL", "OVERDUE"] }, status: { not: "CANCELLED" } }),
      _sum: { total: true, amountPaid: true },
    }),
    db.order.aggregate({
      where: scoped({ createdAt: { gte: startOfMonth }, status: { not: "CANCELLED" } }),
      _sum: { total: true },
      _count: true,
    }),
    db.lead.findMany({
      where: scope as Prisma.LeadWhereInput,
      orderBy: { createdAt: "desc" },
      take: 6,
      include: { assignedTo: { select: { name: true } } },
    }),
    db.order.findMany({
      where: scope,
      orderBy: { createdAt: "desc" },
      take: 6,
      include: { customer: { select: { name: true } } },
    }),
    db.order.groupBy({
      by: ["status"],
      where: scoped({ status: { in: [...LIVE_ORDER_STATUSES] } }),
      _count: true,
    }),
  ]);

  const outstanding = toNumber(unpaid._sum.total) - toNumber(unpaid._sum.amountPaid);

  return (
    <>
      <PageHeader
        title={`Good day, ${user.name.split(" ")[0]}`}
        subtitle="Everything needing your attention, in one view."
        action={
          <div className="flex gap-2">
            <Link href="/leads/new" className="btn btn-ghost">
              New lead
            </Link>
            <Link href="/orders/new" className="btn btn-accent">
              New order
            </Link>
          </div>
        }
      />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
        <StatCard label="Open leads" value={String(openLeads)} href="/leads" />
        <StatCard label="Live orders" value={String(liveOrders)} href="/production" />
        <StatCard
          label="Due in 3 days"
          value={String(dueSoon)}
          tone={dueSoon > 0 ? "warn" : "neutral"}
          href="/orders?status=CONFIRMED"
        />
        <StatCard
          label="Overdue"
          value={String(overdueOrders)}
          tone={overdueOrders > 0 ? "danger" : "good"}
          href="/orders"
        />
        <StatCard
          label="Outstanding"
          value={formatMoney(outstanding)}
          tone={outstanding > 0 ? "warn" : "good"}
          hint="Billed but not collected"
          href="/payments"
        />
        <StatCard
          label="This month"
          value={formatMoney(monthRevenue._sum.total)}
          hint={`${monthRevenue._count} orders`}
          tone="good"
        />
      </div>

      <section className="mt-6 card p-4">
        <h2 className="text-sm font-semibold text-ink-900">Work in progress</h2>
        {byStatus.length === 0 ? (
          <p className="mt-3 text-sm text-ink-500">Nothing on the floor right now.</p>
        ) : (
          <div className="mt-3 flex flex-wrap gap-2">
            {byStatus.map((row) => (
              <Link
                key={row.status}
                href={`/orders?status=${row.status}`}
                className="rounded-lg border border-ink-100 px-3 py-2 text-sm hover:bg-ink-50"
              >
                <span className="font-semibold tabular-nums">{row._count}</span>{" "}
                <span className="text-ink-500">{humanise(row.status)}</span>
              </Link>
            ))}
          </div>
        )}
      </section>

      <div className="mt-6 grid gap-4 xl:grid-cols-2">
        <section className="card">
          <div className="flex items-center justify-between border-b border-ink-100 px-4 py-3">
            <h2 className="text-sm font-semibold text-ink-900">Latest leads</h2>
            <Link href="/leads" className="text-sm font-medium text-brand-600">
              View all
            </Link>
          </div>
          {recentLeads.length === 0 ? (
            <EmptyState title="No leads yet" hint="They will appear here as your channels send them in." />
          ) : (
            <div className="scroll-x">
              <table className="w-full">
                <thead className="border-b border-ink-100">
                  <tr>
                    <Th>Enquiry</Th>
                    <Th>Status</Th>
                    <Th>Priority</Th>
                    <Th>Owner</Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink-100">
                  {recentLeads.map((lead) => (
                    <tr key={lead.id} className="hover:bg-ink-50">
                      <Td>
                        <Link href={`/leads/${lead.id}`} className="font-medium text-ink-900 hover:underline">
                          {lead.title}
                        </Link>
                        <span className="block text-xs text-ink-500">
                          {lead.contactName ?? lead.company ?? "—"} · {humanise(lead.sourceKind)}
                        </span>
                      </Td>
                      <Td>
                        <LeadStatusBadge value={lead.status} />
                      </Td>
                      <Td>
                        <PriorityBadge value={lead.priority} />
                      </Td>
                      <Td className="whitespace-nowrap text-ink-500">
                        {lead.assignedTo?.name ?? "Unassigned"}
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="card">
          <div className="flex items-center justify-between border-b border-ink-100 px-4 py-3">
            <h2 className="text-sm font-semibold text-ink-900">Latest orders</h2>
            <Link href="/orders" className="text-sm font-medium text-brand-600">
              View all
            </Link>
          </div>
          {recentOrders.length === 0 ? (
            <EmptyState title="No orders yet" hint="Create one, or connect a sales channel." />
          ) : (
            <div className="scroll-x">
              <table className="w-full">
                <thead className="border-b border-ink-100">
                  <tr>
                    <Th>Order</Th>
                    <Th>Stage</Th>
                    <Th>Payment</Th>
                    <Th>Due</Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink-100">
                  {recentOrders.map((order) => (
                    <tr key={order.id} className="hover:bg-ink-50">
                      <Td>
                        <Link href={`/orders/${order.id}`} className="font-medium text-ink-900 hover:underline">
                          {order.orderNo}
                        </Link>
                        <span className="block text-xs text-ink-500">
                          {order.customer.name} · {formatMoney(order.total, order.currency)}
                        </span>
                      </Td>
                      <Td>
                        <OrderStatusBadge value={order.status} />
                      </Td>
                      <Td>
                        <PaymentBadge value={order.paymentStatus} />
                      </Td>
                      <Td className="whitespace-nowrap text-ink-500">{formatDate(order.dueDate)}</Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </>
  );
}
