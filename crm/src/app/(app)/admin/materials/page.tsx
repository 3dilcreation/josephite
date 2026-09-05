import { PrintTechnology } from "@prisma/client";
import { requireRole } from "@/lib/auth";
import { db } from "@/lib/db";
import { createMaterial, updateMaterialStock } from "@/actions/quotes";
import { formatMoney, toNumber, humanise } from "@/lib/format";
import { PageHeader, Th, Td, EmptyState } from "@/components/ui";
import { Badge } from "@/components/badges";
import { Field, Select, enumOptions } from "@/components/forms";

export const dynamic = "force-dynamic";

export default async function MaterialsPage() {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");

  const materials = await db.material.findMany({
    where: { orgId: admin.orgId },
    orderBy: [{ technology: "asc" }, { name: "asc" }],
  });

  const lowStock = materials.filter(
    (m) => toNumber(m.reorderLevelGrams) > 0 && toNumber(m.stockGrams) <= toNumber(m.reorderLevelGrams),
  );

  return (
    <>
      <PageHeader
        title="Materials"
        subtitle="Density turns a model's volume into grams; cost per gram turns grams into rupees. Every quote depends on these two numbers."
      />

      {lowStock.length > 0 ? (
        <p className="mb-4 rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-800">
          Running low: {lowStock.map((m) => m.name).join(", ")}. Reorder before the next batch.
        </p>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="card xl:col-span-2">
          {materials.length === 0 ? (
            <EmptyState title="No materials yet" hint="Add your most-used filament or resin first." />
          ) : (
            <div className="scroll-x">
              <table className="w-full">
                <thead className="border-b border-ink-100">
                  <tr>
                    <Th>Material</Th>
                    <Th>Density</Th>
                    <Th>Cost/g</Th>
                    <Th>Waste</Th>
                    <Th>In stock</Th>
                    <Th>Adjust</Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink-100">
                  {materials.map((material) => {
                    const stock = toNumber(material.stockGrams);
                    const low =
                      toNumber(material.reorderLevelGrams) > 0 &&
                      stock <= toNumber(material.reorderLevelGrams);
                    return (
                      <tr key={material.id}>
                        <Td>
                          <span className="font-medium text-ink-900">{material.name}</span>
                          <span className="block text-xs text-ink-500">
                            {[humanise(material.technology), material.brand, material.colour]
                              .filter(Boolean)
                              .join(" · ")}
                          </span>
                        </Td>
                        <Td className="whitespace-nowrap tabular-nums text-ink-500">
                          {toNumber(material.densityGramsPerCm3).toFixed(2)} g/cm³
                        </Td>
                        <Td className="whitespace-nowrap tabular-nums">
                          {formatMoney(material.costPerGram)}
                        </Td>
                        <Td className="tabular-nums text-ink-500">{toNumber(material.wastePercent)}%</Td>
                        <Td className="whitespace-nowrap tabular-nums">
                          {stock.toFixed(0)} g
                          {low ? (
                            <span className="ml-1">
                              <Badge tone="red">Low</Badge>
                            </span>
                          ) : null}
                        </Td>
                        <Td>
                          <form action={updateMaterialStock} className="flex gap-1">
                            <input type="hidden" name="id" value={material.id} />
                            <input
                              name="deltaGrams"
                              type="number"
                              step="1"
                              placeholder="±g"
                              required
                              className="input w-24 py-1 text-xs"
                            />
                            <button type="submit" className="btn btn-ghost px-2 py-1 text-xs">
                              Apply
                            </button>
                          </form>
                        </Td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <form action={createMaterial} className="card h-fit space-y-3 p-5">
          <h2 className="text-sm font-semibold text-ink-900">Add material</h2>
          <Field label="Name" name="name" required placeholder="PLA — Black" />
          <Select label="Process" name="technology" options={enumOptions(PrintTechnology)} defaultValue="FDM" />
          <div className="grid grid-cols-2 gap-2">
            <Field label="Brand" name="brand" />
            <Field label="Colour" name="colour" />
          </div>
          <Field
            label="Density (g/cm³)"
            name="densityGramsPerCm3"
            type="number"
            step="0.01"
            defaultValue={1.24}
          />
          <Field label="Cost per gram (₹)" name="costPerGram" type="number" step="0.01" defaultValue={1.5} />
          <Field label="Waste allowance %" name="wastePercent" type="number" step="0.5" defaultValue={10} />
          <div className="grid grid-cols-2 gap-2">
            <Field label="Stock (g)" name="stockGrams" type="number" defaultValue={0} />
            <Field label="Reorder at (g)" name="reorderLevelGrams" type="number" defaultValue={0} />
          </div>
          <button type="submit" className="btn btn-accent w-full">
            Add material
          </button>
        </form>
      </div>
    </>
  );
}
