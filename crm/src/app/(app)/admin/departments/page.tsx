import { requireRole } from "@/lib/auth";
import { db } from "@/lib/db";
import { createDepartment } from "@/actions/misc";
import { PageHeader, Th, Td } from "@/components/ui";
import { Field, TextArea } from "@/components/forms";

export const dynamic = "force-dynamic";

export default async function DepartmentsPage() {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");

  const departments = await db.department.findMany({
    where: { orgId: admin.orgId },
    orderBy: { name: "asc" },
    include: {
      _count: { select: { members: true, leads: true, orders: true } },
    },
  });

  return (
    <>
      <PageHeader
        title="Departments"
        subtitle="Work is routed to a department; anyone in it sees and can act on it."
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="card xl:col-span-2">
          <div className="scroll-x">
            <table className="w-full">
              <thead className="border-b border-ink-100">
                <tr>
                  <Th>Department</Th>
                  <Th>People</Th>
                  <Th>Open leads</Th>
                  <Th>Orders</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-100">
                {departments.map((department) => (
                  <tr key={department.id}>
                    <Td>
                      <span className="inline-flex items-center gap-2">
                        <span
                          className="inline-block size-2.5 rounded-full"
                          style={{ background: department.colour }}
                        />
                        <span className="font-medium text-ink-900">{department.name}</span>
                      </span>
                      <span className="block text-xs text-ink-500">{department.description ?? "—"}</span>
                    </Td>
                    <Td className="tabular-nums">{department._count.members}</Td>
                    <Td className="tabular-nums">{department._count.leads}</Td>
                    <Td className="tabular-nums">{department._count.orders}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <form action={createDepartment} className="card h-fit space-y-3 p-5">
          <h2 className="text-sm font-semibold text-ink-900">Add department</h2>
          <Field label="Name" name="name" required placeholder="Production" />
          <TextArea label="Description" name="description" rows={2} />
          <Field label="Colour" name="colour" type="color" defaultValue="#ef6820" />
          <button type="submit" className="btn btn-accent w-full">
            Create department
          </button>
        </form>
      </div>
    </>
  );
}
