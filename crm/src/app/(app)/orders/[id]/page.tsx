import Link from "next/link";
import { notFound } from "next/navigation";
import { OrderStatus, Priority, PaymentMethod, PrintTechnology } from "@prisma/client";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { scopeWhere, canWrite } from "@/lib/rbac";
import { updateOrder, addOrderItem, deleteOrderItem, recordPayment } from "@/actions/orders";
import { addNote } from "@/actions/misc";
import { formatMoney, formatDate, formatDateTime, toNumber, humanise } from "@/lib/format";
import { PageHeader, Th, Td } from "@/components/ui";
import { OrderStatusBadge, PaymentBadge, PriorityBadge, ChannelBadge } from "@/components/badges";
import { Field, TextArea, Select, enumOptions } from "@/components/forms";

export const dynamic = "force-dynamic";

export default async function OrderDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const user = await requireUser();

  const order = await db.order.findFirst({
    where: { AND: [{ id }, scopeWhere(user)] },
    include: {
      customer: true,
      assignedTo: { select: { name: true } },
      department: { select: { name: true } },
      source: { select: { name: true } },
      lead: { select: { id: true, title: true } },
      items: { orderBy: { id: "asc" } },
      payments: {
        orderBy: { paidAt: "desc" },
        include: { recordedBy: { select: { name: true } } },
      },
    },
  });

  if (!order) notFound();

  const [members, departments, notes, activity] = await Promise.all([
    db.user.findMany({
      where: { orgId: user.orgId, isActive: true },
      orderBy: { name: "asc" },
      select: { id: true, name: true },
    }),
    db.department.findMany({ where: { orgId: user.orgId }, orderBy: { name: "asc" } }),
    db.note.findMany({
      where: { entityType: "ORDER", entityId: id },
      orderBy: { createdAt: "desc" },
      include: { author: { select: { name: true } } },
    }),
    db.activity.findMany({
      where: { entityType: { in: ["ORDER", "PAYMENT"] }, entityId: id },
      orderBy: { createdAt: "desc" },
      take: 25,
    }),
  ]);

  const editable = canWrite(user.role);
  const balance = toNumber(order.total) - toNumber(order.amountPaid);

  return (
    <>
      <PageHeader
        title={order.orderNo}
        subtitle={`${order.customer.name} · created ${formatDateTime(order.createdAt)} via ${
          order.source?.name ?? humanise(order.sourceKind)
        }`}
        action={
          <Link href="/orders" className="btn btn-ghost">
            Back
          </Link>
        }
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          <section className="card p-5">
            <div className="mb-4 flex flex-wrap gap-2">
              <OrderStatusBadge value={order.status} />
              <PaymentBadge value={order.paymentStatus} />
              <PriorityBadge value={order.priority} />
              <ChannelBadge value={order.channel} />
            </div>

            <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
              <Detail label="Total" value={formatMoney(order.total, order.currency)} />
              <Detail label="Received" value={formatMoney(order.amountPaid, order.currency)} />
              <Detail
                label="Balance"
                value={formatMoney(balance, order.currency)}
                tone={balance > 0 ? "danger" : "good"}
              />
              <Detail label="Delivery due" value={formatDate(order.dueDate)} />
              <Detail label="Payment due" value={formatDate(order.paymentDueDate)} />
              <Detail label="Delivered" value={formatDate(order.deliveredAt)} />
              <Detail label="Owner" value={order.assignedTo?.name ?? "Unassigned"} />
              <Detail label="Department" value={order.department?.name} />
              <Detail label="Tracking" value={order.trackingNumber} />
            </dl>

            <div className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
              <div className="rounded-lg bg-ink-50 p-3">
                <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-500">
                  Customer
                </p>
                <Link href={`/customers/${order.customer.id}`} className="font-medium text-brand-600">
                  {order.customer.name}
                </Link>
                <p className="text-ink-700">
                  {[order.customer.company, order.customer.phone, order.customer.email]
                    .filter(Boolean)
                    .join(" · ") || "—"}
                </p>
              </div>
              <div className="rounded-lg bg-ink-50 p-3">
                <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-500">
                  Ship to
                </p>
                <p className="text-ink-700">{order.shippingAddress || order.customer.address || "—"}</p>
              </div>
            </div>

            {order.externalUrl ? (
              <p className="mt-3 text-sm">
                <a href={order.externalUrl} target="_blank" rel="noreferrer" className="text-brand-600">
                  Open on {humanise(order.sourceKind)} ↗
                </a>
              </p>
            ) : null}
            {order.lead ? (
              <p className="mt-2 text-sm text-ink-500">
                From lead{" "}
                <Link href={`/leads/${order.lead.id}`} className="text-brand-600">
                  {order.lead.title}
                </Link>
              </p>
            ) : null}
          </section>

          <section className="card">
            <h2 className="border-b border-ink-100 px-4 py-3 text-sm font-semibold text-ink-900">
              Line items
            </h2>
            <div className="scroll-x">
              <table className="w-full">
                <thead className="border-b border-ink-100">
                  <tr>
                    <Th>Item</Th>
                    <Th>Process</Th>
                    <Th>Qty</Th>
                    <Th>Unit</Th>
                    <Th>Line total</Th>
                    {editable ? <Th> </Th> : null}
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink-100">
                  {order.items.map((item) => (
                    <tr key={item.id}>
                      <Td>
                        <span className="font-medium text-ink-900">{item.name}</span>
                        <span className="block text-xs text-ink-500">
                          {[item.description, item.fileName].filter(Boolean).join(" · ") || "—"}
                        </span>
                      </Td>
                      <Td className="text-ink-500">
                        {[item.technology, item.material, item.colour].filter(Boolean).join(" / ") || "—"}
                        {item.printHours ? (
                          <span className="block text-xs">{Number(item.printHours)} h print</span>
                        ) : null}
                      </Td>
                      <Td className="tabular-nums">{item.quantity}</Td>
                      <Td className="whitespace-nowrap tabular-nums">{formatMoney(item.unitPrice)}</Td>
                      <Td className="whitespace-nowrap tabular-nums font-medium">
                        {formatMoney(toNumber(item.unitPrice) * item.quantity)}
                      </Td>
                      {editable ? (
                        <Td>
                          <form action={deleteOrderItem}>
                            <input type="hidden" name="itemId" value={item.id} />
                            <button type="submit" className="text-xs font-medium text-red-600">
                              Remove
                            </button>
                          </form>
                        </Td>
                      ) : null}
                    </tr>
                  ))}
                </tbody>
                <tfoot className="border-t border-ink-100 text-sm">
                  <SummaryRow label="Subtotal" value={formatMoney(order.subtotal)} span={editable ? 6 : 5} />
                  <SummaryRow label="Tax" value={formatMoney(order.taxAmount)} span={editable ? 6 : 5} />
                  <SummaryRow label="Shipping" value={formatMoney(order.shipping)} span={editable ? 6 : 5} />
                  <SummaryRow label="Discount" value={`− ${formatMoney(order.discount)}`} span={editable ? 6 : 5} />
                  <SummaryRow label="Total" value={formatMoney(order.total)} span={editable ? 6 : 5} strong />
                </tfoot>
              </table>
            </div>

            {editable ? (
              <form action={addOrderItem} className="space-y-3 border-t border-ink-100 p-4">
                <input type="hidden" name="orderId" value={order.id} />
                <p className="text-xs font-semibold uppercase tracking-wide text-ink-500">Add item</p>
                <div className="grid gap-3 sm:grid-cols-3">
                  <Field label="Item name" name="name" required />
                  <Select
                    label="Technology"
                    name="technology"
                    includeBlank
                    options={enumOptions(PrintTechnology)}
                  />
                  <Field label="Material" name="material" />
                </div>
                <div className="grid gap-3 sm:grid-cols-4">
                  <Field label="Colour" name="colour" />
                  <Field label="Quantity" name="quantity" type="number" defaultValue={1} />
                  <Field label="Unit price (₹)" name="unitPrice" type="number" step="0.01" defaultValue={0} />
                  <Field label="Print hours" name="printHours" type="number" step="0.1" />
                </div>
                <button type="submit" className="btn btn-primary">
                  Add item
                </button>
              </form>
            ) : null}
          </section>

          <section className="card">
            <h2 className="border-b border-ink-100 px-4 py-3 text-sm font-semibold text-ink-900">
              Payments
            </h2>
            {order.payments.length === 0 ? (
              <p className="px-4 py-6 text-sm text-ink-500">Nothing received yet.</p>
            ) : (
              <div className="scroll-x">
                <table className="w-full">
                  <thead className="border-b border-ink-100">
                    <tr>
                      <Th>Date</Th>
                      <Th>Amount</Th>
                      <Th>Method</Th>
                      <Th>Reference</Th>
                      <Th>Recorded by</Th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-ink-100">
                    {order.payments.map((payment) => (
                      <tr key={payment.id}>
                        <Td className="whitespace-nowrap">{formatDate(payment.paidAt)}</Td>
                        <Td className="whitespace-nowrap tabular-nums font-medium">
                          {formatMoney(payment.amount)}
                        </Td>
                        <Td>{humanise(payment.method)}</Td>
                        <Td className="text-ink-500">{payment.reference ?? "—"}</Td>
                        <Td className="text-ink-500">{payment.recordedBy?.name ?? "System"}</Td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {editable && balance > 0 ? (
              <form action={recordPayment} className="grid gap-3 border-t border-ink-100 p-4 sm:grid-cols-4">
                <input type="hidden" name="orderId" value={order.id} />
                <Field
                  label="Amount (₹)"
                  name="amount"
                  type="number"
                  step="0.01"
                  required
                  defaultValue={balance}
                />
                <Select label="Method" name="method" options={enumOptions(PaymentMethod)} defaultValue="UPI" />
                <Field label="Reference" name="reference" placeholder="UTR / txn id" />
                <div className="flex items-end">
                  <button type="submit" className="btn btn-accent w-full">
                    Record payment
                  </button>
                </div>
              </form>
            ) : null}
          </section>

          <section className="card p-5">
            <h2 className="mb-3 text-sm font-semibold text-ink-900">Notes</h2>
            {editable ? (
              <form action={addNote} className="mb-4 flex gap-2">
                <input type="hidden" name="entityType" value="ORDER" />
                <input type="hidden" name="entityId" value={order.id} />
                <input name="body" required placeholder="Support failure, reprinting part 2…" className="input" />
                <button type="submit" className="btn btn-primary">
                  Add
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

          <section className="card p-5">
            <h2 className="mb-3 text-sm font-semibold text-ink-900">History</h2>
            {activity.length === 0 ? (
              <p className="text-sm text-ink-500">Nothing recorded yet.</p>
            ) : (
              <ul className="space-y-2 text-sm">
                {activity.map((entry) => (
                  <li key={entry.id} className="flex flex-wrap justify-between gap-2">
                    <span className="text-ink-800">{entry.summary}</span>
                    <span className="text-xs text-ink-500">{formatDateTime(entry.createdAt)}</span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>

        {editable ? (
          <form action={updateOrder} className="card h-fit space-y-4 p-5">
            <input type="hidden" name="id" value={order.id} />
            <h2 className="text-sm font-semibold text-ink-900">Update</h2>
            <Select label="Stage" name="status" options={enumOptions(OrderStatus)} defaultValue={order.status} />
            <Select
              label="Priority"
              name="priority"
              options={enumOptions(Priority)}
              defaultValue={order.priority}
            />
            <Select
              label="Assign to"
              name="assignedToId"
              includeBlank
              blankLabel="— unassigned —"
              options={members.map((m) => ({ value: m.id, label: m.name }))}
              defaultValue={order.assignedToId}
            />
            <Select
              label="Department"
              name="departmentId"
              includeBlank
              options={departments.map((d) => ({ value: d.id, label: d.name }))}
              defaultValue={order.departmentId}
            />
            <Field
              label="Delivery due"
              name="dueDate"
              type="date"
              defaultValue={order.dueDate?.toISOString().slice(0, 10)}
            />
            <Field
              label="Payment due"
              name="paymentDueDate"
              type="date"
              defaultValue={order.paymentDueDate?.toISOString().slice(0, 10)}
            />
            <Field label="Tracking number" name="trackingNumber" defaultValue={order.trackingNumber} />
            <div className="grid grid-cols-3 gap-2">
              <Field label="Tax" name="taxAmount" type="number" step="0.01" defaultValue={Number(order.taxAmount)} />
              <Field label="Ship" name="shipping" type="number" step="0.01" defaultValue={Number(order.shipping)} />
              <Field label="Disc." name="discount" type="number" step="0.01" defaultValue={Number(order.discount)} />
            </div>
            <button type="submit" className="btn btn-primary w-full">
              Save changes
            </button>
          </form>
        ) : null}
      </div>
    </>
  );
}

function Detail({
  label,
  value,
  tone,
}: {
  label: string;
  value?: string | null;
  tone?: "danger" | "good";
}) {
  const colour = tone === "danger" ? "text-red-600" : tone === "good" ? "text-emerald-600" : "text-ink-800";
  return (
    <div>
      <dt className="text-xs font-semibold uppercase tracking-wide text-ink-500">{label}</dt>
      <dd className={`font-medium ${colour}`}>{value || "—"}</dd>
    </div>
  );
}

function SummaryRow({
  label,
  value,
  span,
  strong,
}: {
  label: string;
  value: string;
  span: number;
  strong?: boolean;
}) {
  return (
    <tr>
      <td colSpan={span - 1} className={`px-3 py-1.5 text-right ${strong ? "font-semibold" : "text-ink-500"}`}>
        {label}
      </td>
      <td className={`px-3 py-1.5 tabular-nums ${strong ? "font-semibold" : "text-ink-700"}`}>{value}</td>
    </tr>
  );
}
