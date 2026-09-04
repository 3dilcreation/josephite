import Link from "next/link";
import type { Prisma } from "@prisma/client";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { customerScopeWhere, canWrite } from "@/lib/rbac";
import { createCustomer } from "@/actions/misc";
import { formatDate } from "@/lib/format";
import { PageHeader, EmptyState, Th, Td } from "@/components/ui";
import { Field } from "@/components/forms";

export const dynamic = "force-dynamic";

export default async function CustomersPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | undefined>>;
}) {
  const params = await searchParams;
  const user = await requireUser();

  const conditions: Prisma.CustomerWhereInput[] = [customerScopeWhere(user)];
  if (params.q) {
    conditions.push({
      OR: [
        { name: { contains: params.q, mode: "insensitive" } },
        { company: { contains: params.q, mode: "insensitive" } },
        { phone: { contains: params.q } },
        { email: { contains: params.q, mode: "insensitive" } },
        { gstin: { contains: params.q, mode: "insensitive" } },
      ],
    });
  }
  const where: Prisma.CustomerWhereInput = { AND: conditions };

  const customers = await db.customer.findMany({
    where,
    orderBy: { updatedAt: "desc" },
    take: 100,
    include: { _count: { select: { orders: true, leads: true } } },
  });

  return (
    <>
      <PageHeader title="Customers" subtitle={`${customers.length} shown`} />

      <form className="mb-4 flex gap-2" action="/customers">
        <input
          name="q"
          defaultValue={params.q ?? ""}
          placeholder="Name, company, phone, email, GSTIN…"
          className="input max-w-md"
        />
        <button type="submit" className="btn btn-primary">
          Search
        </button>
      </form>

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="card xl:col-span-2">
          {customers.length === 0 ? (
            <EmptyState title="No customers yet" hint="Add one on the right, or let a channel create it." />
          ) : (
            <div className="scroll-x">
              <table className="w-full">
                <thead className="border-b border-ink-100">
                  <tr>
                    <Th>Customer</Th>
                    <Th>Contact</Th>
                    <Th>Orders</Th>
                    <Th>Leads</Th>
                    <Th>Added</Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink-100">
                  {customers.map((customer) => (
                    <tr key={customer.id} className="hover:bg-ink-50">
                      <Td>
                        <Link
                          href={`/customers/${customer.id}`}
                          className="font-medium text-ink-900 hover:underline"
                        >
                          {customer.name}
                        </Link>
                        <span className="block text-xs text-ink-500">{customer.company ?? "—"}</span>
                      </Td>
                      <Td className="text-ink-500">
                        {customer.phone ?? "—"}
                        <span className="block text-xs">{customer.email ?? ""}</span>
                      </Td>
                      <Td className="tabular-nums">{customer._count.orders}</Td>
                      <Td className="tabular-nums">{customer._count.leads}</Td>
                      <Td className="whitespace-nowrap text-ink-500">{formatDate(customer.createdAt)}</Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {canWrite(user.role) ? (
          <form action={createCustomer} className="card h-fit space-y-3 p-5">
            <h2 className="text-sm font-semibold text-ink-900">Add customer</h2>
            <Field label="Name" name="name" required />
            <Field label="Company" name="company" />
            <Field label="Phone" name="phone" type="tel" />
            <Field label="Email" name="email" type="email" />
            <Field label="GSTIN" name="gstin" />
            <Field label="Address" name="address" />
            <div className="grid grid-cols-3 gap-2">
              <Field label="City" name="city" />
              <Field label="State" name="state" />
              <Field label="PIN" name="pincode" />
            </div>
            <button type="submit" className="btn btn-accent w-full">
              Save customer
            </button>
          </form>
        ) : null}
      </div>
    </>
  );
}
