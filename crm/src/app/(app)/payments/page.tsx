import Link from "next/link";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { scopeWhere } from "@/lib/rbac";
import { formatMoney, formatDate, toNumber, humanise, relativeDays } from "@/lib/format";
import { PageHeader, StatCard, EmptyState, Th, Td } from "@/components/ui";
import { PaymentBadge, OrderStatusBadge } from "@/components/badges";

export const dynamic = "force-dynamic";

export default async function PaymentsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | undefined>>;
}) {
  const params = await searchParams;
  const user = await requireUser();
  const scope = scopeWhere(user);

  const from = params.from ? new Date(params.from) : null;
  const to = params.to ? new Date(`${params.to}T23:59:59`) : null;

  const [receivable, received, recentPayments, monthTotal] = await Promise.all([
    db.order.findMany({
      where: {
        AND: [scope, { status: { not: "CANCELLED" }, paymentStatus: { in: ["UNPAID", "PARTIAL", "OVERDUE"] } }],
      },
      orderBy: [{ paymentDueDate: { sort: "asc", nulls: "last" } }],
      take: 100,
      include: { customer: { select: { id: true, name: true } } },
    }),
    db.payment.aggregate({
      where: {
        order: scope,
        ...(from || to ? { paidAt: { ...(from ? { gte: from } : {}), ...(to ? { lte: to } : {}) } } : {}),
      },
      _sum: { amount: true },
      _count: true,
    }),
    db.payment.findMany({
      where: {
        order: scope,
        ...(from || to ? { paidAt: { ...(from ? { gte: from } : {}), ...(to ? { lte: to } : {}) } } : {}),
      },
      orderBy: { paidAt: "desc" },
      take: 50,
      include: {
        order: { select: { id: true, orderNo: true, customer: { select: { name: true } } } },
        recordedBy: { select: { name: true } },
      },
    }),
    db.payment.aggregate({
      where: {
        order: scope,
        paidAt: { gte: new Date(new Date().getFullYear(), new Date().getMonth(), 1) },
      },
      _sum: { amount: true },
    }),
  ]);

  const outstanding = receivable.reduce(
    (sum, order) => sum + toNumber(order.total) - toNumber(order.amountPaid),
    0,
  );
  const overdue = receivable.filter(
    (order) => order.paymentDueDate && order.paymentDueDate < new Date(),
  );
  const overdueAmount = overdue.reduce(
    (sum, order) => sum + toNumber(order.total) - toNumber(order.amountPaid),
    0,
  );

  return (
    <>
      <PageHeader title="Payments" subtitle="What is owed, what came in, and what is late." />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard
          label="Outstanding"
          value={formatMoney(outstanding)}
          hint={`${receivable.length} open invoices`}
          tone={outstanding > 0 ? "warn" : "good"}
        />
        <StatCard
          label="Overdue"
          value={formatMoney(overdueAmount)}
          hint={`${overdue.length} past due date`}
          tone={overdueAmount > 0 ? "danger" : "good"}
        />
        <StatCard label="Collected this month" value={formatMoney(monthTotal._sum.amount)} tone="good" />
        <StatCard
          label="In selected range"
          value={formatMoney(received._sum.amount)}
          hint={`${received._count} payments`}
        />
      </div>

      <form className="my-4 flex flex-wrap items-end gap-2" action="/payments">
        <div>
          <label className="label" htmlFor="from">
            From
          </label>
          <input id="from" name="from" type="date" defaultValue={params.from ?? ""} className="input" />
        </div>
        <div>
          <label className="label" htmlFor="to">
            To
          </label>
          <input id="to" name="to" type="date" defaultValue={params.to ?? ""} className="input" />
        </div>
        <button type="submit" className="btn btn-primary">
          Apply
        </button>
        <Link href="/payments" className="btn btn-ghost">
          Reset
        </Link>
      </form>

      <div className="grid gap-4 xl:grid-cols-2">
        <section className="card">
          <h2 className="border-b border-ink-100 px-4 py-3 text-sm font-semibold text-ink-900">
            Money owed to you
          </h2>
          {receivable.length === 0 ? (
            <EmptyState title="Everything is collected" />
          ) : (
            <div className="scroll-x">
              <table className="w-full">
                <thead className="border-b border-ink-100">
                  <tr>
                    <Th>Order</Th>
                    <Th>Stage</Th>
                    <Th>Balance</Th>
                    <Th>Status</Th>
                    <Th>Due</Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink-100">
                  {receivable.map((order) => {
                    const days = relativeDays(order.paymentDueDate);
                    const late = days !== null && days < 0;
                    return (
                      <tr key={order.id} className="hover:bg-ink-50">
                        <Td>
                          <Link href={`/orders/${order.id}`} className="font-medium text-brand-600">
                            {order.orderNo}
                          </Link>
                          <span className="block text-xs text-ink-500">{order.customer.name}</span>
                        </Td>
                        <Td>
                          <OrderStatusBadge value={order.status} />
                        </Td>
                        <Td className="whitespace-nowrap tabular-nums font-medium">
                          {formatMoney(toNumber(order.total) - toNumber(order.amountPaid))}
                        </Td>
                        <Td>
                          <PaymentBadge value={order.paymentStatus} />
                        </Td>
                        <Td className="whitespace-nowrap">
                          <span className={late ? "font-semibold text-red-600" : "text-ink-500"}>
                            {formatDate(order.paymentDueDate)}
                          </span>
                        </Td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="card">
          <h2 className="border-b border-ink-100 px-4 py-3 text-sm font-semibold text-ink-900">
            Payments received
          </h2>
          {recentPayments.length === 0 ? (
            <EmptyState title="No payments in this range" />
          ) : (
            <div className="scroll-x">
              <table className="w-full">
                <thead className="border-b border-ink-100">
                  <tr>
                    <Th>Date</Th>
                    <Th>Order</Th>
                    <Th>Amount</Th>
                    <Th>Method</Th>
                    <Th>By</Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink-100">
                  {recentPayments.map((payment) => (
                    <tr key={payment.id} className="hover:bg-ink-50">
                      <Td className="whitespace-nowrap">{formatDate(payment.paidAt)}</Td>
                      <Td>
                        <Link href={`/orders/${payment.order.id}`} className="font-medium text-brand-600">
                          {payment.order.orderNo}
                        </Link>
                        <span className="block text-xs text-ink-500">{payment.order.customer.name}</span>
                      </Td>
                      <Td className="whitespace-nowrap tabular-nums font-medium text-emerald-700">
                        {formatMoney(payment.amount)}
                      </Td>
                      <Td className="text-ink-500">{humanise(payment.method)}</Td>
                      <Td className="text-ink-500">{payment.recordedBy?.name ?? "System"}</Td>
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
