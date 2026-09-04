import Link from "next/link";
import { notFound } from "next/navigation";
import { LeadStatus, Priority, type Prisma } from "@prisma/client";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { scopeWhere, canWrite } from "@/lib/rbac";
import { updateLead, convertLeadToOrder } from "@/actions/leads";
import { addNote } from "@/actions/misc";
import { formatMoney, formatDateTime, formatDate, humanise } from "@/lib/format";
import { PageHeader } from "@/components/ui";
import { LeadStatusBadge, PriorityBadge, ChannelBadge, Badge } from "@/components/badges";
import { Field, Select, enumOptions } from "@/components/forms";

export const dynamic = "force-dynamic";

export default async function LeadDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const user = await requireUser();

  const lead = await db.lead.findFirst({
    where: { AND: [{ id }, scopeWhere(user) as Prisma.LeadWhereInput] },
    include: {
      assignedTo: { select: { name: true } },
      department: { select: { name: true } },
      customer: { select: { id: true, name: true } },
      source: { select: { name: true } },
      orders: { select: { id: true, orderNo: true, status: true } },
    },
  });

  if (!lead) notFound();

  const [members, departments, notes, activity] = await Promise.all([
    db.user.findMany({
      where: { orgId: user.orgId, isActive: true },
      orderBy: { name: "asc" },
      select: { id: true, name: true },
    }),
    db.department.findMany({ where: { orgId: user.orgId }, orderBy: { name: "asc" } }),
    db.note.findMany({
      where: { entityType: "LEAD", entityId: id },
      orderBy: { createdAt: "desc" },
      include: { author: { select: { name: true } } },
    }),
    db.activity.findMany({
      where: { entityType: "LEAD", entityId: id },
      orderBy: { createdAt: "desc" },
      take: 20,
    }),
  ]);

  const editable = canWrite(user.role);

  return (
    <>
      <PageHeader
        title={lead.title}
        subtitle={`Received ${formatDateTime(lead.createdAt)} via ${lead.source?.name ?? humanise(lead.sourceKind)}`}
        action={
          <div className="flex flex-wrap gap-2">
            <Link href="/leads" className="btn btn-ghost">
              Back
            </Link>
            {editable && lead.status !== "LOST" && lead.orders.length === 0 ? (
              <form action={convertLeadToOrder}>
                <input type="hidden" name="id" value={lead.id} />
                <button type="submit" className="btn btn-accent">
                  Convert to order
                </button>
              </form>
            ) : null}
          </div>
        }
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          <section className="card p-5">
            <div className="mb-4 flex flex-wrap gap-2">
              <LeadStatusBadge value={lead.status} />
              <PriorityBadge value={lead.priority} />
              <ChannelBadge value={lead.channel} />
              {lead.estimatedValue ? <Badge tone="green">{formatMoney(lead.estimatedValue)}</Badge> : null}
            </div>

            <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
              <Detail label="Contact" value={lead.contactName} />
              <Detail label="Company" value={lead.company} />
              <Detail label="Phone" value={lead.phone} />
              <Detail label="Email" value={lead.email} />
              <Detail label="Owner" value={lead.assignedTo?.name ?? "Unassigned"} />
              <Detail label="Department" value={lead.department?.name} />
              <Detail label="Expected close" value={formatDate(lead.expectedCloseDate)} />
              <Detail label="Customer record" value={lead.customer?.name} />
            </dl>

            {lead.requirement ? (
              <div className="mt-4 rounded-lg bg-ink-50 p-3 text-sm text-ink-800">
                <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-500">
                  Requirement
                </p>
                {lead.requirement}
              </div>
            ) : null}

            {lead.orders.length > 0 ? (
              <p className="mt-4 text-sm text-ink-500">
                Converted to{" "}
                {lead.orders.map((order) => (
                  <Link key={order.id} href={`/orders/${order.id}`} className="font-medium text-brand-600">
                    {order.orderNo}
                  </Link>
                ))}
              </p>
            ) : null}
          </section>

          <section className="card p-5">
            <h2 className="mb-3 text-sm font-semibold text-ink-900">Notes</h2>
            {editable ? (
              <form action={addNote} className="mb-4 flex gap-2">
                <input type="hidden" name="entityType" value="LEAD" />
                <input type="hidden" name="entityId" value={lead.id} />
                <input
                  name="body"
                  required
                  placeholder="Called, sent quote, waiting on drawings…"
                  className="input"
                />
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
          <form action={updateLead} className="card h-fit space-y-4 p-5">
            <input type="hidden" name="id" value={lead.id} />
            <h2 className="text-sm font-semibold text-ink-900">Update</h2>
            <Select label="Status" name="status" options={enumOptions(LeadStatus)} defaultValue={lead.status} />
            <Select
              label="Priority"
              name="priority"
              options={enumOptions(Priority)}
              defaultValue={lead.priority}
            />
            <Select
              label="Assign to"
              name="assignedToId"
              includeBlank
              blankLabel="— unassigned —"
              options={members.map((m) => ({ value: m.id, label: m.name }))}
              defaultValue={lead.assignedToId}
            />
            <Select
              label="Department"
              name="departmentId"
              includeBlank
              options={departments.map((d) => ({ value: d.id, label: d.name }))}
              defaultValue={lead.departmentId}
            />
            <Field
              label="Estimated value (₹)"
              name="estimatedValue"
              type="number"
              step="0.01"
              defaultValue={lead.estimatedValue ? Number(lead.estimatedValue) : undefined}
            />
            <Field
              label="Expected close date"
              name="expectedCloseDate"
              type="date"
              defaultValue={lead.expectedCloseDate?.toISOString().slice(0, 10)}
            />
            <Field label="Lost reason" name="lostReason" defaultValue={lead.lostReason} />
            <button type="submit" className="btn btn-primary w-full">
              Save changes
            </button>
          </form>
        ) : null}
      </div>
    </>
  );
}

function Detail({ label, value }: { label: string; value?: string | null }) {
  return (
    <div>
      <dt className="text-xs font-semibold uppercase tracking-wide text-ink-500">{label}</dt>
      <dd className="text-ink-800">{value || "—"}</dd>
    </div>
  );
}
