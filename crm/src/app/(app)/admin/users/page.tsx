import { Role } from "@prisma/client";
import { requireRole } from "@/lib/auth";
import { db } from "@/lib/db";
import { ROLE_LABELS } from "@/lib/rbac";
import { createUser, updateUser } from "@/actions/misc";
import { formatDateTime } from "@/lib/format";
import { PageHeader, Th, Td } from "@/components/ui";
import { Badge } from "@/components/badges";
import { Field, Select } from "@/components/forms";

export const dynamic = "force-dynamic";

export default async function UsersPage() {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");

  const [users, departments, branches] = await Promise.all([
    db.user.findMany({
      where: { orgId: admin.orgId },
      orderBy: [{ isActive: "desc" }, { name: "asc" }],
      include: {
        branch: { select: { name: true } },
        departments: { include: { department: { select: { id: true, name: true } } } },
      },
    }),
    db.department.findMany({ where: { orgId: admin.orgId }, orderBy: { name: "asc" } }),
    db.branch.findMany({ where: { orgId: admin.orgId }, orderBy: { name: "asc" } }),
  ]);

  // A branch admin cannot promote anyone to super admin, so the option is not
  // offered — the server action enforces the same rule.
  const roleOptions = Object.values(Role)
    .filter((role) => admin.role === "SUPER_ADMIN" || role !== "SUPER_ADMIN")
    .map((role) => ({ value: role, label: ROLE_LABELS[role] }));

  return (
    <>
      <PageHeader
        title="Users & roles"
        subtitle="Create logins, set what each person can see, and put them in departments."
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="card xl:col-span-2">
          <div className="scroll-x">
            <table className="w-full">
              <thead className="border-b border-ink-100">
                <tr>
                  <Th>Person</Th>
                  <Th>Role</Th>
                  <Th>Departments</Th>
                  <Th>Branch</Th>
                  <Th>Last sign in</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-100">
                {users.map((member) => (
                  <tr key={member.id} className={member.isActive ? "" : "opacity-50"}>
                    <Td>
                      <span className="font-medium text-ink-900">{member.name}</span>
                      <span className="block text-xs text-ink-500">{member.email}</span>
                    </Td>
                    <Td>
                      <Badge tone={member.role === "SUPER_ADMIN" ? "purple" : "neutral"}>
                        {ROLE_LABELS[member.role]}
                      </Badge>
                      {!member.isActive ? (
                        <span className="ml-1">
                          <Badge tone="red">Disabled</Badge>
                        </span>
                      ) : null}
                    </Td>
                    <Td className="text-ink-500">
                      {member.departments.map((d) => d.department.name).join(", ") || "—"}
                    </Td>
                    <Td className="text-ink-500">{member.branch?.name ?? "—"}</Td>
                    <Td className="whitespace-nowrap text-ink-500">
                      {member.lastLoginAt ? formatDateTime(member.lastLoginAt) : "Never"}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <details className="border-t border-ink-100 p-4">
            <summary className="cursor-pointer text-sm font-semibold text-ink-900">
              Edit an existing user
            </summary>
            <div className="mt-4 space-y-6">
              {users.map((member) => (
                <form
                  key={member.id}
                  action={updateUser}
                  className="grid gap-3 rounded-xl border border-ink-100 p-4 sm:grid-cols-2"
                >
                  <input type="hidden" name="id" value={member.id} />
                  <p className="text-sm font-semibold text-ink-900 sm:col-span-2">{member.email}</p>
                  <Field label="Name" name="name" defaultValue={member.name} />
                  <Field label="Phone" name="phone" defaultValue={member.phone} />
                  <Select label="Role" name="role" options={roleOptions} defaultValue={member.role} />
                  <Select
                    label="Account"
                    name="isActive"
                    options={[
                      { value: "true", label: "Active" },
                      { value: "false", label: "Disabled" },
                    ]}
                    defaultValue={String(member.isActive)}
                  />
                  <Field
                    label="Reset password (blank = keep)"
                    name="password"
                    type="password"
                    className="sm:col-span-2"
                  />
                  <fieldset className="sm:col-span-2">
                    <legend className="label">Departments</legend>
                    <div className="flex flex-wrap gap-3">
                      {departments.map((department) => (
                        <label key={department.id} className="flex items-center gap-1.5 text-sm">
                          <input
                            type="checkbox"
                            name="departmentIds"
                            value={department.id}
                            defaultChecked={member.departments.some(
                              (d) => d.departmentId === department.id,
                            )}
                          />
                          {department.name}
                        </label>
                      ))}
                    </div>
                  </fieldset>
                  <div className="sm:col-span-2">
                    <button type="submit" className="btn btn-primary">
                      Save
                    </button>
                  </div>
                </form>
              ))}
            </div>
          </details>
        </div>

        <form action={createUser} className="card h-fit space-y-3 p-5">
          <h2 className="text-sm font-semibold text-ink-900">Add a user</h2>
          <Field label="Full name" name="name" required />
          <Field label="Email" name="email" type="email" required />
          <Field label="Phone" name="phone" type="tel" />
          <Field
            label="Temporary password"
            name="password"
            type="password"
            required
            placeholder="At least 8 characters"
          />
          <Select label="Role" name="role" options={roleOptions} defaultValue="STAFF" />
          {admin.role === "SUPER_ADMIN" ? (
            <Select
              label="Branch"
              name="branchId"
              includeBlank
              blankLabel="— head office —"
              options={branches.map((b) => ({ value: b.id, label: b.name }))}
            />
          ) : null}
          <fieldset>
            <legend className="label">Departments</legend>
            <div className="space-y-1">
              {departments.length === 0 ? (
                <p className="text-sm text-ink-500">Create a department first.</p>
              ) : (
                departments.map((department) => (
                  <label key={department.id} className="flex items-center gap-2 text-sm">
                    <input type="checkbox" name="departmentIds" value={department.id} />
                    {department.name}
                  </label>
                ))
              )}
            </div>
          </fieldset>
          <button type="submit" className="btn btn-accent w-full">
            Create user
          </button>
        </form>
      </div>
    </>
  );
}
