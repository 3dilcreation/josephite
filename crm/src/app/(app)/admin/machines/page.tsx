import { PrintTechnology, MachineStatus } from "@prisma/client";
import { requireRole } from "@/lib/auth";
import { db } from "@/lib/db";
import { createMachine, setMachineStatus } from "@/actions/quotes";
import { formatMoney, toNumber, humanise } from "@/lib/format";
import { PageHeader, Th, Td, EmptyState } from "@/components/ui";
import { Badge } from "@/components/badges";
import { Field, Select, enumOptions } from "@/components/forms";

export const dynamic = "force-dynamic";

const TONES: Record<string, "green" | "blue" | "amber" | "neutral"> = {
  IDLE: "green",
  PRINTING: "blue",
  MAINTENANCE: "amber",
  OFFLINE: "neutral",
};

export default async function MachinesPage() {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");

  const machines = await db.machine.findMany({
    where: { orgId: admin.orgId },
    orderBy: [{ technology: "asc" }, { name: "asc" }],
  });

  return (
    <>
      <PageHeader
        title="Machines"
        subtitle="Hourly rate and throughput drive the machine-time half of every quote. Calibrate cm³/hour from real jobs — the manufacturer's figure is optimistic."
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="card xl:col-span-2">
          {machines.length === 0 ? (
            <EmptyState title="No machines yet" hint="Quotes fall back to default rates until you add one." />
          ) : (
            <div className="scroll-x">
              <table className="w-full">
                <thead className="border-b border-ink-100">
                  <tr>
                    <Th>Machine</Th>
                    <Th>Build volume</Th>
                    <Th>Rate</Th>
                    <Th>Throughput</Th>
                    <Th>Status</Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink-100">
                  {machines.map((machine) => (
                    <tr key={machine.id}>
                      <Td>
                        <span className="font-medium text-ink-900">{machine.name}</span>
                        <span className="block text-xs text-ink-500">
                          {[humanise(machine.technology), machine.model].filter(Boolean).join(" · ")}
                        </span>
                      </Td>
                      <Td className="whitespace-nowrap text-ink-500">
                        {machine.buildXmm
                          ? `${machine.buildXmm} × ${machine.buildYmm} × ${machine.buildZmm} mm`
                          : "—"}
                      </Td>
                      <Td className="whitespace-nowrap tabular-nums">
                        {formatMoney(machine.hourlyRate)}/h
                      </Td>
                      <Td className="whitespace-nowrap tabular-nums text-ink-500">
                        {toNumber(machine.cm3PerHour)} cm³/h
                        <span className="block text-xs">
                          at {toNumber(machine.referenceLayerMm)} mm layers
                        </span>
                      </Td>
                      <Td>
                        <form action={setMachineStatus} className="flex items-center gap-2">
                          <input type="hidden" name="id" value={machine.id} />
                          <Badge tone={TONES[machine.status]}>{humanise(machine.status)}</Badge>
                          <select
                            name="status"
                            defaultValue={machine.status}
                            className="input w-32 py-1 text-xs"
                          >
                            {Object.values(MachineStatus).map((status) => (
                              <option key={status} value={status}>
                                {humanise(status)}
                              </option>
                            ))}
                          </select>
                          <button type="submit" className="btn btn-ghost px-2 py-1 text-xs">
                            Set
                          </button>
                        </form>
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <form action={createMachine} className="card h-fit space-y-3 p-5">
          <h2 className="text-sm font-semibold text-ink-900">Add machine</h2>
          <Field label="Name" name="name" required placeholder="Bambu P1S #1" />
          <Select label="Process" name="technology" options={enumOptions(PrintTechnology)} defaultValue="FDM" />
          <Field label="Model" name="model" />
          <div className="grid grid-cols-3 gap-2">
            <Field label="X mm" name="buildXmm" type="number" />
            <Field label="Y mm" name="buildYmm" type="number" />
            <Field label="Z mm" name="buildZmm" type="number" />
          </div>
          <Field label="Hourly rate (₹)" name="hourlyRate" type="number" step="1" defaultValue={60} />
          <Field label="Throughput (cm³/h)" name="cm3PerHour" type="number" step="0.5" defaultValue={15} />
          <Field
            label="Reference layer (mm)"
            name="referenceLayerMm"
            type="number"
            step="0.01"
            defaultValue={0.2}
          />
          <button type="submit" className="btn btn-accent w-full">
            Add machine
          </button>
        </form>
      </div>
    </>
  );
}
