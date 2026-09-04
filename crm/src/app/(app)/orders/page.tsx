import Link from "next/link";
import { OrderStatus, PaymentStatus, Priority, Channel, SourceKind, type Prisma } from "@prisma/client";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { scopeWhere } from "@/lib/rbac";
import { formatMoney, formatDate, toNumber, relativeDays, pluralDays, humanise } from "@/lib/format";
import { PageHeader, EmptyState, Th, Td } from "@/components/ui";
import { OrderStatusBadge, PaymentBadge, PriorityBadge, ChannelBadge } from "@/components/badges";
import { FilterBar } from "@/components/filter-bar";
import { enumOptions } from "@/components/forms";

export const dynamic = "force-dynamic";

const PAGE_SIZE = 25;

export default async function OrdersPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | undefined>>;
}) {
  const params = await searchParams;
  const user = await requireUser();

  const page = Math.max(1, Number(params.page ?? 1));

  // Scope first, then filters, all under one AND — so the search box's own OR
  // can never widen what this user is allowed to see.
  const conditions: Prisma.OrderWhereInput[] = [scopeWhere(user)];

  if (params.status) conditions.push({ status: params.status as never });
  if (params.paymentStatus) conditions.push({ paymentStatus: params.paymentStatus as never });
  if (params.priority) conditions.push({ priority: params.priority as never });
  if (params.channel) conditions.push({ channel: params.channel as never });
  if (params.sourceKind) conditions.push({ sourceKind: params.sourceKind as never });
  if (params.from || params.to) {
    conditions.push({
      createdAt: {
        ...(params.from ? { gte: new Date(params.from) } : {}),
        ...(params.to ? { lte: new Date(`${params.to}T23:59:59`) } : {}),
      },
    });
  }
  if (params.q) {
    conditions.push({
      OR: [
        { orderNo: { contains: params.q, mode: "insensitive" } },
        { trackingNumber: { contains: params.q, mode: "insensitive" } },
        { customer: { name: { contains: params.q, mode: "insensitive" } } },
        { customer: { phone: { contains: params.q } } },
      ],
    });
  }

  const where: Prisma.OrderWhereInput = { AND: conditions };

  const [orders, total, totals] = await Promise.all([
    db.order.findMany({
      where,
      // Soonest deadline first; orders with no deadline sink to the bottom.
      orderBy: [{ dueDate: { sort: "asc", nulls: "last" } }, { createdAt: "desc" }],
      skip: (page - 1) * PAGE_SIZE,
      take: PAGE_SIZE,
      include: {
        customer: { select: { id: true, name: true } },
        assignedTo: { select: { name: true } },
      },
    }),
    db.order.count({ where }),
    db.order.aggregate({ where, _sum: { total: true, amountPaid: true } }),
  ]);

  const outstanding = toNumber(totals._sum.total) - toNumber(totals._sum.amountPaid);
  const queryString = (nextPage: number) => {
    const next = new URLSearchParams(
      Object.entries(params).filter(([, value]) => value) as [string, string][],
    );
    next.set("page", String(nextPage));
    return `?${next.toString()}`;
  };
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <>
      <PageHeader
        title="Orders"
        subtitle={`${total} orders · ${formatMoney(totals._sum.total)} billed · ${formatMoney(outstanding)} outstanding`}
        action={
          <Link href="/orders/new" className="btn btn-accent">
            New order
          </Link>
        }
      />

      <FilterBar
        searchPlaceholder="Order no, customer, tracking…"
        filters={[
          { name: "status", label: "Stage", options: enumOptions(OrderStatus) },
          { name: "paymentStatus", label: "Payment", options: enumOptions(PaymentStatus) },
          { name: "priority", label: "Priority", options: enumOptions(Priority) },
          { name: "channel", label: "Channel", options: enumOptions(Channel) },
          { name: "sourceKind", label: "Source", options: enumOptions(SourceKind) },
        ]}
      />

      <div className="card">
        {orders.length === 0 ? (
          <EmptyState title="No orders match these filters" />
        ) : (
          <div className="scroll-x">
            <table className="w-full">
              <thead className="border-b border-ink-100">
                <tr>
                  <Th>Order</Th>
                  <Th>Stage</Th>
                  <Th>Priority</Th>
                  <Th>Channel</Th>
                  <Th>Total</Th>
                  <Th>Payment</Th>
                  <Th>Due</Th>
                  <Th>Owner</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-100">
                {orders.map((order) => {
                  const days = relativeDays(order.dueDate);
                  const late =
                    days !== null &&
                    days < 0 &&
                    !["DELIVERED", "SHIPPED", "CANCELLED"].includes(order.status);
                  return (
                    <tr key={order.id} className="hover:bg-ink-50">
                      <Td>
                        <Link
                          href={`/orders/${order.id}`}
                          className="font-medium text-ink-900 hover:underline"
                        >
                          {order.orderNo}
                        </Link>
                        <span className="block text-xs text-ink-500">
                          {order.customer.name} · {humanise(order.sourceKind)}
                        </span>
                      </Td>
                      <Td>
                        <OrderStatusBadge value={order.status} />
                      </Td>
                      <Td>
                        <PriorityBadge value={order.priority} />
                      </Td>
                      <Td>
                        <ChannelBadge value={order.channel} />
                      </Td>
                      <Td className="whitespace-nowrap tabular-nums">
                        {formatMoney(order.total, order.currency)}
                      </Td>
                      <Td>
                        <PaymentBadge value={order.paymentStatus} />
                        {toNumber(order.amountPaid) > 0 &&
                        toNumber(order.amountPaid) < toNumber(order.total) ? (
                          <span className="block text-xs text-ink-500">
                            {formatMoney(toNumber(order.total) - toNumber(order.amountPaid))} left
                          </span>
                        ) : null}
                      </Td>
                      <Td className="whitespace-nowrap">
                        <span className={late ? "font-semibold text-red-600" : "text-ink-500"}>
                          {formatDate(order.dueDate)}
                        </span>
                        {late ? (
                          <span className="block text-xs text-red-600">{pluralDays(Math.abs(days!))} late</span>
                        ) : null}
                      </Td>
                      <Td className="whitespace-nowrap text-ink-500">
                        {order.assignedTo?.name ?? "Unassigned"}
                      </Td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {pages > 1 ? (
        <div className="mt-4 flex items-center justify-between text-sm">
          <span className="text-ink-500">
            Page {page} of {pages}
          </span>
          <div className="flex gap-2">
            {page > 1 ? (
              <Link href={queryString(page - 1)} className="btn btn-ghost">
                Previous
              </Link>
            ) : null}
            {page < pages ? (
              <Link href={queryString(page + 1)} className="btn btn-ghost">
                Next
              </Link>
            ) : null}
          </div>
        </div>
      ) : null}
    </>
  );
}
