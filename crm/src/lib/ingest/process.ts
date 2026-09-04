import { createHmac, timingSafeEqual } from "node:crypto";
import { db } from "@/lib/db";
import { notify, logActivity } from "@/lib/notify";
import { nextOrderNumber, recalculateOrder } from "@/lib/orders";
import { normalizePayload } from "@/lib/ingest/normalize";
import { contactMatch } from "@/lib/contacts";
import { formatMoney } from "@/lib/format";
import type { IntegrationSource } from "@prisma/client";

export function sign(secret: string, body: string) {
  return createHmac("sha256", secret).update(body).digest("hex");
}

export function verifySignature(secret: string, body: string, provided: string | null) {
  if (!provided) return false;
  const expected = sign(secret, body);
  const a = Buffer.from(expected);
  const b = Buffer.from(provided.replace(/^sha256=/, ""));
  return a.length === b.length && timingSafeEqual(a, b);
}

/**
 * Find an existing customer by phone or email before creating one, so the same
 * person ordering from the website and from Amazon stays a single record with a
 * single order history — the whole point of centralising channels.
 */
async function upsertCustomer(
  orgId: string,
  branchId: string | null,
  input: { name: string; email?: string; phone?: string; address?: string },
) {
  const identifiers = contactMatch(input);

  if (identifiers) {
    const existing = await db.customer.findFirst({ where: { orgId, OR: identifiers } });
    if (existing) return existing;
  }

  return db.customer.create({
    data: {
      orgId,
      branchId,
      name: input.name,
      email: input.email,
      phone: input.phone,
      address: input.address,
    },
  });
}

export async function processInboundPayload(source: IntegrationSource, payload: Record<string, unknown>) {
  const config = (source.config ?? {}) as Record<string, unknown>;
  const normalized = normalizePayload(source.kind, payload);
  const branchId = source.defaultBranchId ?? null;
  const departmentId = source.defaultDepartmentId ?? null;

  if (normalized.type === "lead") {
    // Replays of the same webhook must not create duplicate leads.
    if (normalized.externalId) {
      const duplicate = await db.lead.findFirst({
        where: { sourceId: source.id, externalId: normalized.externalId },
      });
      if (duplicate) return { entityType: "LEAD" as const, entityId: duplicate.id, duplicate: true };
    }

    const customer =
      normalized.email || normalized.phone
        ? await upsertCustomer(source.orgId, branchId, {
            name: normalized.contactName ?? normalized.title,
            email: normalized.email,
            phone: normalized.phone,
          })
        : null;

    const lead = await db.lead.create({
      data: {
        orgId: source.orgId,
        branchId,
        departmentId,
        customerId: customer?.id,
        title: normalized.title,
        contactName: normalized.contactName,
        email: normalized.email,
        phone: normalized.phone,
        company: normalized.company,
        requirement: normalized.requirement,
        estimatedValue: normalized.estimatedValue,
        priority: normalized.priority ?? (config.defaultPriority as never) ?? "MEDIUM",
        channel: normalized.channel,
        sourceKind: source.kind,
        sourceId: source.id,
        externalId: normalized.externalId,
        externalUrl: normalized.externalUrl,
      },
    });

    await logActivity({
      orgId: source.orgId,
      entityType: "LEAD",
      entityId: lead.id,
      action: "lead.created",
      summary: `Lead received from ${source.name}`,
    });

    // Nobody is assigned yet, so alert whoever can act: the department's members,
    // or the branch admins if the source has no default department.
    const watchers = await db.user.findMany({
      where: departmentId
        ? { orgId: source.orgId, isActive: true, departments: { some: { departmentId } } }
        : { orgId: source.orgId, isActive: true, role: { in: ["SUPER_ADMIN", "ADMIN"] } },
      select: { id: true },
    });

    await notify({
      orgId: source.orgId,
      userIds: watchers.map((w) => w.id),
      type: "LEAD_ASSIGNED",
      title: `New lead from ${source.name}`,
      body: normalized.title,
      link: `/leads/${lead.id}`,
    });

    return { entityType: "LEAD" as const, entityId: lead.id, duplicate: false };
  }

  const duplicate = await db.order.findFirst({
    where: { sourceId: source.id, externalId: normalized.externalId },
  });
  if (duplicate) return { entityType: "ORDER" as const, entityId: duplicate.id, duplicate: true };

  const customer = await upsertCustomer(source.orgId, branchId, {
    name: normalized.contactName,
    email: normalized.email,
    phone: normalized.phone,
    address: normalized.shippingAddress,
  });

  const order = await db.order.create({
    data: {
      orgId: source.orgId,
      branchId,
      departmentId,
      orderNo: await nextOrderNumber(source.orgId),
      customerId: customer.id,
      status: "CONFIRMED",
      channel: normalized.channel,
      sourceKind: source.kind,
      sourceId: source.id,
      externalId: normalized.externalId,
      externalUrl: normalized.externalUrl,
      currency: normalized.currency ?? "INR",
      shippingAddress: normalized.shippingAddress,
      items: {
        create: normalized.items.map((item) => ({
          name: item.name,
          quantity: item.quantity,
          unitPrice: item.unitPrice,
          material: item.material,
          colour: item.colour,
          notes: item.notes,
        })),
      },
    },
  });

  if (normalized.amountPaid && normalized.amountPaid > 0) {
    await db.payment.create({
      data: {
        orderId: order.id,
        amount: normalized.amountPaid,
        method: "GATEWAY",
        reference: `${source.name} ${normalized.externalId}`,
      },
    });
  }

  await recalculateOrder(order.id);

  await logActivity({
    orgId: source.orgId,
    entityType: "ORDER",
    entityId: order.id,
    action: "order.created",
    summary: `Order imported from ${source.name}`,
  });

  const watchers = await db.user.findMany({
    where: departmentId
      ? { orgId: source.orgId, isActive: true, departments: { some: { departmentId } } }
      : { orgId: source.orgId, isActive: true, role: { in: ["SUPER_ADMIN", "ADMIN"] } },
    select: { id: true },
  });

  await notify({
    orgId: source.orgId,
    userIds: watchers.map((w) => w.id),
    type: "ORDER_ASSIGNED",
    title: `New ${source.name} order ${order.orderNo}`,
    body: `${normalized.items.length} item(s) · ${formatMoney(normalized.total, normalized.currency ?? "INR")}`,
    link: `/orders/${order.id}`,
  });

  return { entityType: "ORDER" as const, entityId: order.id, duplicate: false };
}
