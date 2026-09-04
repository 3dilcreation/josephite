import { SourceKind } from "@prisma/client";
import { requireRole } from "@/lib/auth";
import { db } from "@/lib/db";
import { createIntegrationSource, toggleIntegrationSource } from "@/actions/misc";
import { formatDateTime, humanise } from "@/lib/format";
import { PageHeader, Th, Td } from "@/components/ui";
import { Badge } from "@/components/badges";
import { Field, Select, enumOptions } from "@/components/forms";
import { SecretReveal } from "./secret-reveal";

export const dynamic = "force-dynamic";

export default async function IntegrationsPage() {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");
  const appUrl = process.env.APP_URL ?? "https://crm.3dilcreation.com";

  const [sources, departments, branches, recentEvents] = await Promise.all([
    db.integrationSource.findMany({
      where: { orgId: admin.orgId },
      orderBy: { createdAt: "asc" },
      include: { _count: { select: { leads: true, orders: true, events: true } } },
    }),
    db.department.findMany({ where: { orgId: admin.orgId }, orderBy: { name: "asc" } }),
    db.branch.findMany({ where: { orgId: admin.orgId }, orderBy: { name: "asc" } }),
    db.webhookEvent.findMany({
      where: { source: { orgId: admin.orgId } },
      orderBy: { createdAt: "desc" },
      take: 25,
      include: { source: { select: { name: true } } },
    }),
  ]);

  return (
    <>
      <PageHeader
        title="Integrations"
        subtitle="Each channel gets its own URL and secret. Point the platform at it and leads or orders land in the CRM."
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          <div className="card">
            <div className="scroll-x">
              <table className="w-full">
                <thead className="border-b border-ink-100">
                  <tr>
                    <Th>Channel</Th>
                    <Th>Endpoint & secret</Th>
                    <Th>Received</Th>
                    <Th>Last event</Th>
                    <Th> </Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink-100">
                  {sources.length === 0 ? (
                    <tr>
                      <Td className="py-8 text-center text-ink-500">
                        No channels connected yet.
                      </Td>
                    </tr>
                  ) : (
                    sources.map((source) => (
                      <tr key={source.id}>
                        <Td>
                          <span className="font-medium text-ink-900">{source.name}</span>
                          <span className="mt-0.5 block">
                            <Badge tone={source.isActive ? "green" : "neutral"}>
                              {source.isActive ? "Active" : "Paused"}
                            </Badge>{" "}
                            <span className="text-xs text-ink-500">{humanise(source.kind)}</span>
                          </span>
                        </Td>
                        <Td>
                          <SecretReveal
                            endpoint={`${appUrl}/api/ingest/${source.key}`}
                            secret={source.secret}
                          />
                        </Td>
                        <Td className="whitespace-nowrap text-ink-500">
                          {source._count.leads} leads · {source._count.orders} orders
                          <span className="block text-xs">{source._count.events} events</span>
                        </Td>
                        <Td className="whitespace-nowrap text-ink-500">
                          {source.lastEventAt ? formatDateTime(source.lastEventAt) : "Never"}
                        </Td>
                        <Td>
                          <form action={toggleIntegrationSource}>
                            <input type="hidden" name="id" value={source.id} />
                            <button type="submit" className="text-xs font-medium text-brand-600">
                              {source.isActive ? "Pause" : "Resume"}
                            </button>
                          </form>
                        </Td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <h2 className="border-b border-ink-100 px-4 py-3 text-sm font-semibold text-ink-900">
              Recent inbound events
            </h2>
            {recentEvents.length === 0 ? (
              <p className="px-4 py-6 text-sm text-ink-500">
                Nothing received yet. Send a test payload to any endpoint above.
              </p>
            ) : (
              <ul className="divide-y divide-ink-100">
                {recentEvents.map((event) => (
                  <li key={event.id} className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5">
                    <span className="text-sm text-ink-800">
                      {event.source.name}
                      {event.error ? <span className="block text-xs text-red-600">{event.error}</span> : null}
                    </span>
                    <span className="flex items-center gap-2">
                      <Badge
                        tone={
                          event.status === "PROCESSED"
                            ? "green"
                            : event.status === "FAILED"
                              ? "red"
                              : "neutral"
                        }
                      >
                        {humanise(event.status)}
                      </Badge>
                      <span className="text-xs text-ink-500">{formatDateTime(event.createdAt)}</span>
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        <form action={createIntegrationSource} className="card h-fit space-y-3 p-5">
          <h2 className="text-sm font-semibold text-ink-900">Connect a channel</h2>
          <Field label="Name" name="name" required placeholder="3dilcreation.com quote form" />
          <Select label="Platform" name="kind" options={enumOptions(SourceKind)} defaultValue="WEBSITE" />
          <Select
            label="Route to department"
            name="defaultDepartmentId"
            includeBlank
            blankLabel="— notify admins —"
            options={departments.map((d) => ({ value: d.id, label: d.name }))}
          />
          <Select
            label="Route to branch"
            name="defaultBranchId"
            includeBlank
            blankLabel="— head office —"
            options={branches.map((b) => ({ value: b.id, label: b.name }))}
          />
          <button type="submit" className="btn btn-accent w-full">
            Generate endpoint
          </button>
          <p className="text-xs text-ink-500">
            Sign every request with <code>x-3dil-signature</code>: the HMAC-SHA256 of the raw JSON body,
            keyed with the secret.
          </p>
        </form>
      </div>
    </>
  );
}
