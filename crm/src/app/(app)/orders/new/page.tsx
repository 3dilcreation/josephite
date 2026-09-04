import Link from "next/link";
import { OrderStatus, Priority, Channel, SourceKind, PrintTechnology } from "@prisma/client";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { customerScopeWhere } from "@/lib/rbac";
import { createOrder } from "@/actions/orders";
import { PageHeader } from "@/components/ui";
import { Field, TextArea, Select, enumOptions } from "@/components/forms";

export const dynamic = "force-dynamic";

export default async function NewOrderPage() {
  const user = await requireUser();

  const [customers, members, departments] = await Promise.all([
    db.customer.findMany({
      where: customerScopeWhere(user),
      orderBy: { name: "asc" },
      take: 500,
      select: { id: true, name: true, company: true, phone: true },
    }),
    db.user.findMany({
      where: { orgId: user.orgId, isActive: true },
      orderBy: { name: "asc" },
      select: { id: true, name: true },
    }),
    db.department.findMany({ where: { orgId: user.orgId }, orderBy: { name: "asc" } }),
  ]);

  if (customers.length === 0) {
    return (
      <>
        <PageHeader title="New order" />
        <div className="card p-6">
          <p className="text-sm text-ink-700">
            Add a customer first — an order always belongs to someone you can invoice and follow up
            with.
          </p>
          <Link href="/customers" className="btn btn-accent mt-4">
            Go to customers
          </Link>
        </div>
      </>
    );
  }

  return (
    <>
      <PageHeader
        title="New order"
        subtitle="One line item to start; add the rest from the order page."
      />

      <form action={createOrder} className="card max-w-3xl space-y-5 p-5">
        <Select
          label="Customer"
          name="customerId"
          options={customers.map((c) => ({
            value: c.id,
            label: [c.name, c.company, c.phone].filter(Boolean).join(" · "),
          }))}
        />

        <div className="grid gap-4 sm:grid-cols-3">
          <Select label="Stage" name="status" options={enumOptions(OrderStatus)} defaultValue="CONFIRMED" />
          <Select label="Priority" name="priority" options={enumOptions(Priority)} defaultValue="MEDIUM" />
          <Select label="Channel" name="channel" options={enumOptions(Channel)} defaultValue="OFFLINE" />
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          <Select label="Source" name="sourceKind" options={enumOptions(SourceKind)} defaultValue="WALK_IN" />
          <Field label="Delivery due date" name="dueDate" type="date" />
          <Field label="Payment due date" name="paymentDueDate" type="date" />
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

        <fieldset className="rounded-xl border border-ink-100 p-4">
          <legend className="px-1 text-xs font-semibold uppercase tracking-wide text-ink-500">
            First line item
          </legend>

          <div className="space-y-4">
            <Field label="Item name" name="itemName" required placeholder="e.g. Drone frame v3" />
            <TextArea label="Description" name="itemDescription" rows={2} />

            <div className="grid gap-4 sm:grid-cols-3">
              <Select
                label="Technology"
                name="technology"
                includeBlank
                options={enumOptions(PrintTechnology)}
              />
              <Field label="Material" name="material" placeholder="PLA / PETG / Nylon PA12" />
              <Field label="Colour" name="colour" />
            </div>

            <div className="grid gap-4 sm:grid-cols-4">
              <Field label="Quantity" name="quantity" type="number" defaultValue={1} />
              <Field label="Unit price (₹)" name="unitPrice" type="number" step="0.01" defaultValue={0} />
              <Field label="Weight (g)" name="weightGrams" type="number" step="0.01" />
              <Field label="Print hours" name="printHours" type="number" step="0.1" />
            </div>

            <Field label="Model file name" name="fileName" placeholder="frame_v3.stl" />
          </div>
        </fieldset>

        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Tax (₹)" name="taxAmount" type="number" step="0.01" defaultValue={0} />
          <Field label="Shipping (₹)" name="shipping" type="number" step="0.01" defaultValue={0} />
          <Field label="Discount (₹)" name="discount" type="number" step="0.01" defaultValue={0} />
        </div>

        <TextArea label="Shipping address" name="shippingAddress" rows={2} />

        <div className="flex gap-2 pt-2">
          <button type="submit" className="btn btn-accent">
            Create order
          </button>
          <Link href="/orders" className="btn btn-ghost">
            Cancel
          </Link>
        </div>
      </form>
    </>
  );
}
