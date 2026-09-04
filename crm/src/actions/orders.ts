"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { db } from "@/lib/db";
import { requireUser } from "@/lib/auth";
import { canWrite } from "@/lib/rbac";
import { notify, logActivity } from "@/lib/notify";
import { nextOrderNumber, recalculateOrder } from "@/lib/orders";
import { humanise, formatMoney } from "@/lib/format";

const optional = (value: FormDataEntryValue | null) => {
  const text = typeof value === "string" ? value.trim() : "";
  return text === "" ? undefined : text;
};

export async function createOrder(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const customerId = String(formData.get("customerId"));
  if (!customerId) throw new Error("Choose a customer");

  // As with leads: an order nobody owns would be invisible to the person who
  // just created it, so they own it until it is reassigned.
  const assignedToId = optional(formData.get("assignedToId")) ?? user.id;

  const order = await db.order.create({
    data: {
      orgId: user.orgId,
      branchId: user.branchId,
      orderNo: await nextOrderNumber(user.orgId),
      customerId,
      status: (optional(formData.get("status")) ?? "CONFIRMED") as never,
      priority: (optional(formData.get("priority")) ?? "MEDIUM") as never,
      channel: (optional(formData.get("channel")) ?? "OFFLINE") as never,
      sourceKind: (optional(formData.get("sourceKind")) ?? "MANUAL") as never,
      assignedToId,
      departmentId: optional(formData.get("departmentId")),
      dueDate: optional(formData.get("dueDate"))
        ? new Date(String(formData.get("dueDate")))
        : undefined,
      paymentDueDate: optional(formData.get("paymentDueDate"))
        ? new Date(String(formData.get("paymentDueDate")))
        : undefined,
      taxAmount: Number(formData.get("taxAmount") ?? 0),
      shipping: Number(formData.get("shipping") ?? 0),
      discount: Number(formData.get("discount") ?? 0),
      shippingAddress: optional(formData.get("shippingAddress")),
      items: {
        create: [
          {
            name: String(formData.get("itemName") ?? "Print job"),
            description: optional(formData.get("itemDescription")),
            technology: (optional(formData.get("technology")) ?? null) as never,
            material: optional(formData.get("material")),
            colour: optional(formData.get("colour")),
            quantity: Number(formData.get("quantity") ?? 1),
            unitPrice: Number(formData.get("unitPrice") ?? 0),
            weightGrams: optional(formData.get("weightGrams"))
              ? Number(formData.get("weightGrams"))
              : null,
            printHours: optional(formData.get("printHours"))
              ? Number(formData.get("printHours"))
              : null,
            fileName: optional(formData.get("fileName")),
          },
        ],
      },
    },
  });

  await recalculateOrder(order.id);

  await logActivity({
    orgId: user.orgId,
    entityType: "ORDER",
    entityId: order.id,
    action: "order.created",
    summary: `${user.name} created order ${order.orderNo}`,
    actorId: user.id,
  });

  await notify({
    orgId: user.orgId,
    userIds: [assignedToId],
    type: "ORDER_ASSIGNED",
    title: `Order ${order.orderNo} assigned to you`,
    link: `/orders/${order.id}`,
    exceptUserId: user.id,
  });

  revalidatePath("/orders");
  redirect(`/orders/${order.id}`);
}

export async function updateOrder(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const id = String(formData.get("id"));
  const before = await db.order.findFirst({ where: { id, orgId: user.orgId } });
  if (!before) throw new Error("Order not found");

  const status = String(formData.get("status") ?? before.status);
  const assignedToId = optional(formData.get("assignedToId")) ?? null;

  await db.order.update({
    where: { id },
    data: {
      status: status as never,
      priority: String(formData.get("priority") ?? before.priority) as never,
      assignedToId,
      departmentId: optional(formData.get("departmentId")) ?? null,
      dueDate: optional(formData.get("dueDate")) ? new Date(String(formData.get("dueDate"))) : null,
      paymentDueDate: optional(formData.get("paymentDueDate"))
        ? new Date(String(formData.get("paymentDueDate")))
        : null,
      trackingNumber: optional(formData.get("trackingNumber")) ?? null,
      taxAmount: Number(formData.get("taxAmount") ?? before.taxAmount),
      shipping: Number(formData.get("shipping") ?? before.shipping),
      discount: Number(formData.get("discount") ?? before.discount),
      deliveredAt: status === "DELIVERED" ? (before.deliveredAt ?? new Date()) : null,
    },
  });

  await recalculateOrder(id);

  if (before.status !== status) {
    await logActivity({
      orgId: user.orgId,
      entityType: "ORDER",
      entityId: id,
      action: "order.status",
      summary: `${user.name} moved ${before.orderNo} to ${humanise(status)}`,
      actorId: user.id,
    });

    // Whoever owns the job and whoever owns the customer both care about a
    // status move; the person clicking the button does not need telling.
    const customer = await db.customer.findUnique({
      where: { id: before.customerId },
      select: { ownerId: true },
    });

    await notify({
      orgId: user.orgId,
      userIds: [assignedToId, customer?.ownerId],
      type: "ORDER_STATUS_CHANGED",
      title: `${before.orderNo} is now ${humanise(status)}`,
      link: `/orders/${id}`,
      exceptUserId: user.id,
    });
  }

  revalidatePath(`/orders/${id}`);
  revalidatePath("/orders");
  revalidatePath("/production");
}

export async function addOrderItem(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const orderId = String(formData.get("orderId"));
  const order = await db.order.findFirst({ where: { id: orderId, orgId: user.orgId } });
  if (!order) throw new Error("Order not found");

  await db.orderItem.create({
    data: {
      orderId,
      name: String(formData.get("name") ?? "Print job"),
      description: optional(formData.get("description")),
      technology: (optional(formData.get("technology")) ?? null) as never,
      material: optional(formData.get("material")),
      colour: optional(formData.get("colour")),
      quantity: Number(formData.get("quantity") ?? 1),
      unitPrice: Number(formData.get("unitPrice") ?? 0),
      weightGrams: optional(formData.get("weightGrams")) ? Number(formData.get("weightGrams")) : null,
      printHours: optional(formData.get("printHours")) ? Number(formData.get("printHours")) : null,
      fileName: optional(formData.get("fileName")),
    },
  });

  await recalculateOrder(orderId);
  revalidatePath(`/orders/${orderId}`);
}

export async function deleteOrderItem(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const id = String(formData.get("itemId"));
  const item = await db.orderItem.findUnique({ where: { id }, include: { order: true } });
  if (!item || item.order.orgId !== user.orgId) throw new Error("Item not found");

  await db.orderItem.delete({ where: { id } });
  await recalculateOrder(item.orderId);
  revalidatePath(`/orders/${item.orderId}`);
}

export async function recordPayment(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const orderId = String(formData.get("orderId"));
  const order = await db.order.findFirst({ where: { id: orderId, orgId: user.orgId } });
  if (!order) throw new Error("Order not found");

  const amount = Number(formData.get("amount"));
  if (!Number.isFinite(amount) || amount <= 0) throw new Error("Enter a payment amount");

  await db.payment.create({
    data: {
      orderId,
      amount,
      method: (formData.get("method") ?? "UPI") as never,
      reference: optional(formData.get("reference")),
      note: optional(formData.get("note")),
      paidAt: optional(formData.get("paidAt")) ? new Date(String(formData.get("paidAt"))) : new Date(),
      recordedById: user.id,
    },
  });

  const updated = await recalculateOrder(orderId);

  await logActivity({
    orgId: user.orgId,
    entityType: "PAYMENT",
    entityId: orderId,
    action: "payment.recorded",
    summary: `${user.name} recorded a payment of ${formatMoney(amount, order.currency)} on ${order.orderNo}`,
    actorId: user.id,
  });

  const admins = await db.user.findMany({
    where: { orgId: user.orgId, isActive: true, role: { in: ["SUPER_ADMIN", "ADMIN"] } },
    select: { id: true },
  });

  await notify({
    orgId: user.orgId,
    userIds: [...admins.map((a) => a.id), order.assignedToId],
    type: "PAYMENT_RECEIVED",
    title: `Payment on ${order.orderNo}`,
    body: `${formatMoney(amount, order.currency)} received · now ${humanise(updated?.paymentStatus ?? "")}`,
    link: `/orders/${orderId}`,
    exceptUserId: user.id,
  });

  revalidatePath(`/orders/${orderId}`);
  revalidatePath("/payments");
}
