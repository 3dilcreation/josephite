"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { z } from "zod";
import { db } from "@/lib/db";
import { requireUser } from "@/lib/auth";
import { canWrite } from "@/lib/rbac";
import { notify, logActivity } from "@/lib/notify";
import { nextOrderNumber } from "@/lib/orders";
import { contactMatch } from "@/lib/contacts";

const optional = (value: FormDataEntryValue | null) => {
  const text = typeof value === "string" ? value.trim() : "";
  return text === "" ? undefined : text;
};

const leadSchema = z.object({
  title: z.string().min(2, "Give the enquiry a short title"),
  contactName: z.string().optional(),
  email: z.string().email().optional().or(z.literal("").transform(() => undefined)),
  phone: z.string().optional(),
  company: z.string().optional(),
  requirement: z.string().optional(),
  priority: z.enum(["LOW", "MEDIUM", "HIGH", "URGENT"]),
  channel: z.enum(["ONLINE", "OFFLINE"]),
  sourceKind: z.string(),
  estimatedValue: z.coerce.number().nonnegative().optional(),
  expectedCloseDate: z.string().optional(),
  assignedToId: z.string().optional(),
  departmentId: z.string().optional(),
});

export async function createLead(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const parsed = leadSchema.parse({
    title: formData.get("title"),
    contactName: optional(formData.get("contactName")),
    email: optional(formData.get("email")),
    phone: optional(formData.get("phone")),
    company: optional(formData.get("company")),
    requirement: optional(formData.get("requirement")),
    priority: formData.get("priority") ?? "MEDIUM",
    channel: formData.get("channel") ?? "OFFLINE",
    sourceKind: formData.get("sourceKind") ?? "MANUAL",
    estimatedValue: optional(formData.get("estimatedValue")),
    expectedCloseDate: optional(formData.get("expectedCloseDate")),
    assignedToId: optional(formData.get("assignedToId")),
    departmentId: optional(formData.get("departmentId")),
  });

  // Default the creator as owner. Without this a manager or staff member logs a
  // walk-in and then cannot see it, because their scope is "assigned to me or my
  // departments" — an unassigned lead would vanish the moment it was saved.
  const assignedToId = parsed.assignedToId ?? user.id;

  const lead = await db.lead.create({
    data: {
      orgId: user.orgId,
      branchId: user.branchId,
      title: parsed.title,
      contactName: parsed.contactName,
      email: parsed.email,
      phone: parsed.phone,
      company: parsed.company,
      requirement: parsed.requirement,
      priority: parsed.priority,
      channel: parsed.channel,
      sourceKind: parsed.sourceKind as never,
      estimatedValue: parsed.estimatedValue,
      expectedCloseDate: parsed.expectedCloseDate ? new Date(parsed.expectedCloseDate) : undefined,
      assignedToId,
      departmentId: parsed.departmentId,
    },
  });

  await logActivity({
    orgId: user.orgId,
    entityType: "LEAD",
    entityId: lead.id,
    action: "lead.created",
    summary: `Lead created by ${user.name}`,
    actorId: user.id,
  });

  await notify({
    orgId: user.orgId,
    userIds: [assignedToId],
    type: "LEAD_ASSIGNED",
    title: `Lead assigned: ${lead.title}`,
    link: `/leads/${lead.id}`,
    exceptUserId: user.id,
  });

  revalidatePath("/leads");
  redirect(`/leads/${lead.id}`);
}

export async function updateLead(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const id = String(formData.get("id"));
  const before = await db.lead.findFirst({ where: { id, orgId: user.orgId } });
  if (!before) throw new Error("Lead not found");

  const status = String(formData.get("status") ?? before.status);
  const priority = String(formData.get("priority") ?? before.priority);
  const assignedToId = optional(formData.get("assignedToId")) ?? null;
  const departmentId = optional(formData.get("departmentId")) ?? null;

  await db.lead.update({
    where: { id },
    data: {
      status: status as never,
      priority: priority as never,
      assignedToId,
      departmentId,
      lostReason: optional(formData.get("lostReason")) ?? null,
      estimatedValue: optional(formData.get("estimatedValue"))
        ? Number(formData.get("estimatedValue"))
        : null,
      expectedCloseDate: optional(formData.get("expectedCloseDate"))
        ? new Date(String(formData.get("expectedCloseDate")))
        : null,
      lastActivityAt: new Date(),
    },
  });

  const changes: string[] = [];
  if (before.status !== status) changes.push(`status ${before.status} → ${status}`);
  if (before.priority !== priority) changes.push(`priority ${before.priority} → ${priority}`);
  if (before.assignedToId !== assignedToId) changes.push("reassigned");

  if (changes.length > 0) {
    await logActivity({
      orgId: user.orgId,
      entityType: "LEAD",
      entityId: id,
      action: "lead.updated",
      summary: `${user.name} changed ${changes.join(", ")}`,
      actorId: user.id,
    });
  }

  if (assignedToId && before.assignedToId !== assignedToId) {
    await notify({
      orgId: user.orgId,
      userIds: [assignedToId],
      type: "LEAD_ASSIGNED",
      title: `Lead assigned to you: ${before.title}`,
      link: `/leads/${id}`,
      exceptUserId: user.id,
    });
  }

  revalidatePath(`/leads/${id}`);
  revalidatePath("/leads");
}

/**
 * Winning a lead should not mean retyping the customer. This creates the
 * customer if they are new, opens a draft order carrying the lead's value, and
 * links the two so the pipeline reporting stays honest.
 */
export async function convertLeadToOrder(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const id = String(formData.get("id"));
  const lead = await db.lead.findFirst({ where: { id, orgId: user.orgId } });
  if (!lead) throw new Error("Lead not found");

  let customerId = lead.customerId;
  if (!customerId) {
    const identifiers = contactMatch(lead);

    const existing = identifiers
      ? await db.customer.findFirst({ where: { orgId: user.orgId, OR: identifiers } })
      : null;

    const customer =
      existing ??
      (await db.customer.create({
        data: {
          orgId: user.orgId,
          branchId: lead.branchId,
          name: lead.contactName ?? lead.title,
          company: lead.company,
          email: lead.email,
          phone: lead.phone,
          ownerId: lead.assignedToId ?? user.id,
        },
      }));
    customerId = customer.id;
  }

  const order = await db.order.create({
    data: {
      orgId: user.orgId,
      branchId: lead.branchId,
      orderNo: await nextOrderNumber(user.orgId),
      customerId,
      leadId: lead.id,
      status: "DRAFT",
      priority: lead.priority,
      channel: lead.channel,
      sourceKind: lead.sourceKind,
      sourceId: lead.sourceId,
      assignedToId: lead.assignedToId,
      departmentId: lead.departmentId,
      items: {
        create: [
          {
            name: lead.title,
            description: lead.requirement,
            quantity: 1,
            unitPrice: lead.estimatedValue ?? 0,
          },
        ],
      },
    },
  });

  await db.lead.update({ where: { id }, data: { status: "WON", customerId } });

  await logActivity({
    orgId: user.orgId,
    entityType: "LEAD",
    entityId: id,
    action: "lead.converted",
    summary: `${user.name} converted this lead into order ${order.orderNo}`,
    actorId: user.id,
  });

  revalidatePath("/leads");
  redirect(`/orders/${order.id}`);
}
