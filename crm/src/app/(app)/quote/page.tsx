import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { customerScopeWhere } from "@/lib/rbac";
import { pricingFor } from "@/lib/quote/service";
import { toNumber } from "@/lib/format";
import { PageHeader } from "@/components/ui";
import { QuoteForm } from "./quote-form";

export const dynamic = "force-dynamic";

export default async function QuotePage() {
  const user = await requireUser();

  const [materials, machines, customers, pricing] = await Promise.all([
    db.material.findMany({
      where: { orgId: user.orgId, isActive: true },
      orderBy: [{ technology: "asc" }, { name: "asc" }],
    }),
    db.machine.findMany({
      where: { orgId: user.orgId, isActive: true },
      orderBy: { name: "asc" },
    }),
    db.customer.findMany({
      where: customerScopeWhere(user),
      orderBy: { name: "asc" },
      take: 500,
      select: { id: true, name: true, company: true },
    }),
    pricingFor(user.orgId),
  ]);

  if (materials.length === 0) {
    return (
      <>
        <PageHeader title="Instant quote" />
        <div className="card p-6">
          <p className="text-sm text-ink-700">
            Add at least one material first — its density and cost per gram are what turn a model&rsquo;s
            volume into a price.
          </p>
          <a href="/admin/materials" className="btn btn-accent mt-4">
            Set up materials
          </a>
        </div>
      </>
    );
  }

  return (
    <>
      <PageHeader
        title="Instant quote"
        subtitle="Upload an STL and get a costed price in seconds. Change the settings to re-price without re-uploading."
      />
      <QuoteForm
        materials={materials.map((m) => ({
          id: m.id,
          name: m.name,
          technology: m.technology,
          colour: m.colour,
          costPerGram: toNumber(m.costPerGram),
          stockGrams: toNumber(m.stockGrams),
        }))}
        machines={machines.map((m) => ({
          id: m.id,
          name: m.name,
          technology: m.technology,
          status: m.status,
        }))}
        customers={customers}
        marginPercent={toNumber(pricing.marginPercent)}
      />
    </>
  );
}
