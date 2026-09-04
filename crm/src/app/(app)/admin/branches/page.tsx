import { BranchType } from "@prisma/client";
import { requireRole } from "@/lib/auth";
import { db } from "@/lib/db";
import { createBranch } from "@/actions/misc";
import { formatMoney, toNumber, humanise } from "@/lib/format";
import { PageHeader, Th, Td } from "@/components/ui";
import { Badge } from "@/components/badges";
import { Field, Select, enumOptions } from "@/components/forms";

export const dynamic = "force-dynamic";

export default async function BranchesPage() {
  const admin = await requireRole("SUPER_ADMIN");

  const branches = await db.branch.findMany({
    where: { orgId: admin.orgId },
    orderBy: [{ type: "asc" }, { name: "asc" }],
    include: { _count: { select: { users: true, customers: true, orders: true } } },
  });

  // One grouped aggregate rather than a query per branch, so this page stays
  // fast when the franchise count grows.
  const revenue = await db.order.groupBy({
    by: ["branchId"],
    where: { orgId: admin.orgId, status: { not: "CANCELLED" } },
    _sum: { total: true, amountPaid: true },
  });
  const revenueByBranch = new Map(revenue.map((row) => [row.branchId, row._sum]));

  return (
    <>
      <PageHeader
        title="Branches & franchises"
        subtitle="Every location reports into one place. Franchise rows also carry a royalty rate."
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="card xl:col-span-2">
          <div className="scroll-x">
            <table className="w-full">
              <thead className="border-b border-ink-100">
                <tr>
                  <Th>Location</Th>
                  <Th>Type</Th>
                  <Th>Staff</Th>
                  <Th>Orders</Th>
                  <Th>Billed</Th>
                  <Th>Collected</Th>
                  <Th>Royalty due</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-100">
                {branches.map((branch) => {
                  const sums = revenueByBranch.get(branch.id);
                  const billed = toNumber(sums?.total);
                  const collected = toNumber(sums?.amountPaid);
                  const royalty = branch.royaltyPercent
                    ? (billed * toNumber(branch.royaltyPercent)) / 100
                    : null;
                  return (
                    <tr key={branch.id}>
                      <Td>
                        <span className="font-medium text-ink-900">{branch.name}</span>
                        <span className="block text-xs text-ink-500">
                          {[branch.code, branch.city, branch.state].filter(Boolean).join(" · ")}
                        </span>
                      </Td>
                      <Td>
                        <Badge tone={branch.type === "FRANCHISE" ? "purple" : "neutral"}>
                          {humanise(branch.type)}
                        </Badge>
                      </Td>
                      <Td className="tabular-nums">{branch._count.users}</Td>
                      <Td className="tabular-nums">{branch._count.orders}</Td>
                      <Td className="whitespace-nowrap tabular-nums">{formatMoney(billed)}</Td>
                      <Td className="whitespace-nowrap tabular-nums">{formatMoney(collected)}</Td>
                      <Td className="whitespace-nowrap tabular-nums">
                        {royalty === null ? "—" : formatMoney(royalty)}
                      </Td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <form action={createBranch} className="card h-fit space-y-3 p-5">
          <h2 className="text-sm font-semibold text-ink-900">Add location</h2>
          <Field label="Name" name="name" required placeholder="3DIL Coimbatore" />
          <Field label="Short code" name="code" required placeholder="CBE" />
          <Select label="Type" name="type" options={enumOptions(BranchType)} defaultValue="BRANCH" />
          <div className="grid grid-cols-2 gap-2">
            <Field label="City" name="city" />
            <Field label="State" name="state" />
          </div>
          <Field label="Phone" name="phone" type="tel" />
          <Field label="Email" name="email" type="email" />
          <Field
            label="Royalty % (franchise only)"
            name="royaltyPercent"
            type="number"
            step="0.01"
            placeholder="e.g. 8"
          />
          <button type="submit" className="btn btn-accent w-full">
            Create location
          </button>
        </form>
      </div>
    </>
  );
}
