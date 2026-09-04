import Link from "next/link";
import { LeadStatus, Priority, SourceKind, Channel, type Prisma } from "@prisma/client";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { scopeWhere } from "@/lib/rbac";
import { formatMoney, formatDate, humanise } from "@/lib/format";
import { PageHeader, EmptyState, Th, Td } from "@/components/ui";
import { LeadStatusBadge, PriorityBadge, ChannelBadge } from "@/components/badges";
import { FilterBar } from "@/components/filter-bar";
import { enumOptions } from "@/components/forms";

export const dynamic = "force-dynamic";

const PAGE_SIZE = 25;

export default async function LeadsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | undefined>>;
}) {
  const params = await searchParams;
  const user = await requireUser();

  const page = Math.max(1, Number(params.page ?? 1));

  // Scope first, then filters, all under one AND — so the search box's own OR
  // can never widen what this user is allowed to see.
  const conditions: Prisma.LeadWhereInput[] = [scopeWhere(user) as Prisma.LeadWhereInput];

  if (params.status) conditions.push({ status: params.status as never });
  if (params.priority) conditions.push({ priority: params.priority as never });
  if (params.sourceKind) conditions.push({ sourceKind: params.sourceKind as never });
  if (params.channel) conditions.push({ channel: params.channel as never });
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
        { title: { contains: params.q, mode: "insensitive" } },
        { contactName: { contains: params.q, mode: "insensitive" } },
        { company: { contains: params.q, mode: "insensitive" } },
        { phone: { contains: params.q } },
        { email: { contains: params.q, mode: "insensitive" } },
      ],
    });
  }

  const where: Prisma.LeadWhereInput = { AND: conditions };

  const [leads, total] = await Promise.all([
    db.lead.findMany({
      where,
      // Urgent first, then oldest untouched — the order a sales desk should work.
      orderBy: [{ priority: "desc" }, { lastActivityAt: "asc" }],
      skip: (page - 1) * PAGE_SIZE,
      take: PAGE_SIZE,
      include: {
        assignedTo: { select: { name: true } },
        department: { select: { name: true } },
      },
    }),
    db.lead.count({ where }),
  ]);

  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const queryString = (nextPage: number) => {
    const next = new URLSearchParams(
      Object.entries(params).filter(([, value]) => value) as [string, string][],
    );
    next.set("page", String(nextPage));
    return `?${next.toString()}`;
  };

  return (
    <>
      <PageHeader
        title="Leads"
        subtitle={`${total} enquir${total === 1 ? "y" : "ies"} · sorted by priority, then longest untouched`}
        action={
          <Link href="/leads/new" className="btn btn-accent">
            New lead
          </Link>
        }
      />

      <FilterBar
        searchPlaceholder="Title, contact, phone, email…"
        filters={[
          { name: "status", label: "Status", options: enumOptions(LeadStatus) },
          { name: "priority", label: "Priority", options: enumOptions(Priority) },
          { name: "channel", label: "Channel", options: enumOptions(Channel) },
          { name: "sourceKind", label: "Source", options: enumOptions(SourceKind) },
        ]}
      />

      <div className="card">
        {leads.length === 0 ? (
          <EmptyState title="No leads match these filters" hint="Try widening the date range." />
        ) : (
          <div className="scroll-x">
            <table className="w-full">
              <thead className="border-b border-ink-100">
                <tr>
                  <Th>Enquiry</Th>
                  <Th>Status</Th>
                  <Th>Priority</Th>
                  <Th>Channel</Th>
                  <Th>Source</Th>
                  <Th>Value</Th>
                  <Th>Owner</Th>
                  <Th>Received</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-100">
                {leads.map((lead) => (
                  <tr key={lead.id} className="hover:bg-ink-50">
                    <Td>
                      <Link href={`/leads/${lead.id}`} className="font-medium text-ink-900 hover:underline">
                        {lead.title}
                      </Link>
                      <span className="block text-xs text-ink-500">
                        {[lead.contactName, lead.company, lead.phone].filter(Boolean).join(" · ") || "—"}
                      </span>
                    </Td>
                    <Td>
                      <LeadStatusBadge value={lead.status} />
                    </Td>
                    <Td>
                      <PriorityBadge value={lead.priority} />
                    </Td>
                    <Td>
                      <ChannelBadge value={lead.channel} />
                    </Td>
                    <Td className="whitespace-nowrap text-ink-500">{humanise(lead.sourceKind)}</Td>
                    <Td className="whitespace-nowrap tabular-nums">
                      {lead.estimatedValue ? formatMoney(lead.estimatedValue) : "—"}
                    </Td>
                    <Td className="whitespace-nowrap text-ink-500">
                      {lead.assignedTo?.name ?? lead.department?.name ?? "Unassigned"}
                    </Td>
                    <Td className="whitespace-nowrap text-ink-500">{formatDate(lead.createdAt)}</Td>
                  </tr>
                ))}
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
