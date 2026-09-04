import Link from "next/link";
import { notFound } from "next/navigation";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { customerScopeWhere, canWrite } from "@/lib/rbac";
import { addNote } from "@/actions/misc";
import { formatMoney, formatDate, formatDateTime, toNumber } from "@/lib/format";
import { PageHeader, StatCard, Th, Td } from "@/components/ui";
import { OrderStatusBadge, PaymentBadge, LeadStatusBadge } from "@/components/badges";

export const dynamic = "force-dynamic";

export default async function CustomerDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const user = await requireUser();

  const customer = await db.customer.findFirst({
    where: { AND: [{ id }, customerScopeWhere(user)] },
    include: {
      owner: { select: { name: true } },
      orders: { orderBy: { createdAt: "desc" } },
      leads: { orderBy: { createdAt: "desc" } },
    },
  });

  if (!customer) notFound();

  const notes = await db.note.findMany({
    where: { entityType: "CUSTOMER", entityId: id },
    orderBy: { createdAt: "desc" },
    include: { author: { select: { name: true } } },
  });

  const billed = customer.orders
    .filter((order) => order.status !== "CANCELLED")
    .reduce((sum, order) => sum + toNumber(order.total), 0);
  const collected = customer.orders.reduce((sum, order) => sum + toNumber(order.amountPaid), 0);

  return (
    <>
      <PageHeader
        title={customer.name}
        subtitle={[customer.company, customer.phone, customer.email].filter(Boolean).join(" · ")}
        action={
          <Link href="/customers" className="btn btn-ghost">
            Back
          </Link>
        }
      />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="Lifetime value" value={formatMoney(billed)} tone="good" />
        <StatCard
          label="Outstanding"
          value={formatMoney(billed - collected)}
          tone={billed - collected > 0 ? "warn" : "neutral"}
        />
        <StatCard label="Orders" value={String(customer.orders.length)} />
        <StatCard label="Account owner" value={customer.owner?.name ?? "—"} />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          <section className="card">
            <h2 className="border-b border-ink-100 px-4 py-3 text-sm font-semibold text-ink-900">
              Orders
            </h2>
            {customer.orders.length === 0 ? (
              <p className="px-4 py-6 text-sm text-ink-500">No orders yet.</p>
            ) : (
              <div className="scroll-x">
                <table className="w-full">
                  <thead className="border-b border-ink-100">
                    <tr>
                      <Th>Order</Th>
                      <Th>Stage</Th>
                      <Th>Total</Th>
                      <Th>Payment</Th>
                      <Th>Placed</Th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-ink-100">
                    {customer.orders.map((order) => (
                      <tr key={order.id} className="hover:bg-ink-50">
                        <Td>
                          <Link href={`/orders/${order.id}`} className="font-medium text-brand-600">
                            {order.orderNo}
                          </Link>
                        </Td>
                        <Td>
                          <OrderStatusBadge value={order.status} />
                        </Td>
                        <Td className="whitespace-nowrap tabular-nums">{formatMoney(order.total)}</Td>
                        <Td>
                          <PaymentBadge value={order.paymentStatus} />
                        </Td>
                        <Td className="whitespace-nowrap text-ink-500">{formatDate(order.createdAt)}</Td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="card">
            <h2 className="border-b border-ink-100 px-4 py-3 text-sm font-semibold text-ink-900">
              Enquiries
            </h2>
            {customer.leads.length === 0 ? (
              <p className="px-4 py-6 text-sm text-ink-500">No enquiries recorded.</p>
            ) : (
              <ul className="divide-y divide-ink-100">
                {customer.leads.map((lead) => (
                  <li key={lead.id} className="flex flex-wrap items-center justify-between gap-2 px-4 py-3">
                    <Link href={`/leads/${lead.id}`} className="text-sm font-medium text-brand-600">
                      {lead.title}
                    </Link>
                    <div className="flex items-center gap-2">
                      <LeadStatusBadge value={lead.status} />
                      <span className="text-xs text-ink-500">{formatDate(lead.createdAt)}</span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>

        <div className="space-y-4">
          <section className="card p-5">
            <h2 className="mb-3 text-sm font-semibold text-ink-900">Details</h2>
            <dl className="space-y-2 text-sm">
              <Row label="GSTIN" value={customer.gstin} />
              <Row label="Address" value={customer.address} />
              <Row
                label="City"
                value={[customer.city, customer.state, customer.pincode].filter(Boolean).join(", ")}
              />
              <Row label="Added" value={formatDate(customer.createdAt)} />
            </dl>
          </section>

          <section className="card p-5">
            <h2 className="mb-3 text-sm font-semibold text-ink-900">Notes</h2>
            {canWrite(user.role) ? (
              <form action={addNote} className="mb-4 space-y-2">
                <input type="hidden" name="entityType" value="CUSTOMER" />
                <input type="hidden" name="entityId" value={customer.id} />
                <input name="body" required placeholder="Prefers PETG, pays on delivery…" className="input" />
                <button type="submit" className="btn btn-primary w-full">
                  Add note
                </button>
              </form>
            ) : null}
            {notes.length === 0 ? (
              <p className="text-sm text-ink-500">No notes yet.</p>
            ) : (
              <ul className="space-y-3">
                {notes.map((note) => (
                  <li key={note.id} className="border-l-2 border-ink-100 pl-3">
                    <p className="text-sm text-ink-800">{note.body}</p>
                    <p className="mt-0.5 text-xs text-ink-500">
                      {note.author?.name ?? "System"} · {formatDateTime(note.createdAt)}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </div>
    </>
  );
}

function Row({ label, value }: { label: string; value?: string | null }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-ink-500">{label}</dt>
      <dd className="text-right text-ink-800">{value || "—"}</dd>
    </div>
  );
}
