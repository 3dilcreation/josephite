import Link from "next/link";
import { Priority, Channel, SourceKind } from "@prisma/client";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { createLead } from "@/actions/leads";
import { PageHeader } from "@/components/ui";
import { Field, TextArea, Select, enumOptions } from "@/components/forms";

export const dynamic = "force-dynamic";

export default async function NewLeadPage() {
  const user = await requireUser();

  const [members, departments] = await Promise.all([
    db.user.findMany({
      where: { orgId: user.orgId, isActive: true },
      orderBy: { name: "asc" },
      select: { id: true, name: true },
    }),
    db.department.findMany({ where: { orgId: user.orgId }, orderBy: { name: "asc" } }),
  ]);

  return (
    <>
      <PageHeader title="New lead" subtitle="Log a walk-in, a call, or an enquiry that arrived off-channel." />

      <form action={createLead} className="card max-w-3xl space-y-4 p-5">
        <Field label="Enquiry title" name="title" required placeholder="e.g. 40 × SLA jigs for assembly line" />

        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Contact name" name="contactName" />
          <Field label="Company" name="company" />
          <Field label="Phone" name="phone" type="tel" />
          <Field label="Email" name="email" type="email" />
        </div>

        <TextArea
          label="Requirement"
          name="requirement"
          rows={4}
          placeholder="Material, tolerance, finish, quantity, deadline…"
        />

        <div className="grid gap-4 sm:grid-cols-3">
          <Select label="Priority" name="priority" options={enumOptions(Priority)} defaultValue="MEDIUM" />
          <Select label="Channel" name="channel" options={enumOptions(Channel)} defaultValue="OFFLINE" />
          <Select label="Source" name="sourceKind" options={enumOptions(SourceKind)} defaultValue="PHONE" />
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Estimated value (₹)" name="estimatedValue" type="number" step="0.01" />
          <Field label="Expected close date" name="expectedCloseDate" type="date" />
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <Select
            label="Assign to"
            name="assignedToId"
            includeBlank
            blankLabel="— me (default) —"
            options={members.map((m) => ({ value: m.id, label: m.name }))}
          />
          <Select
            label="Department"
            name="departmentId"
            includeBlank
            options={departments.map((d) => ({ value: d.id, label: d.name }))}
          />
        </div>

        <div className="flex gap-2 pt-2">
          <button type="submit" className="btn btn-accent">
            Create lead
          </button>
          <Link href="/leads" className="btn btn-ghost">
            Cancel
          </Link>
        </div>
      </form>
    </>
  );
}
