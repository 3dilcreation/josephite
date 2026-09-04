import { db } from "@/lib/db";
import { toNumber } from "@/lib/format";
import type { PaymentStatus } from "@prisma/client";

/** Sequential, human-quotable order numbers: 3DIL-2026-0042. */
export async function nextOrderNumber(orgId: string) {
  const year = new Date().getFullYear();
  const prefix = `3DIL-${year}-`;
  const last = await db.order.findFirst({
    where: { orgId, orderNo: { startsWith: prefix } },
    orderBy: { orderNo: "desc" },
    select: { orderNo: true },
  });
  const sequence = last ? Number(last.orderNo.slice(prefix.length)) + 1 : 1;
  return `${prefix}${String(sequence).padStart(4, "0")}`;
}

export function derivePaymentStatus(total: number, paid: number, dueDate: Date | null): PaymentStatus {
  if (paid <= 0) {
    return dueDate && dueDate < new Date() ? "OVERDUE" : "UNPAID";
  }
  if (paid + 0.01 >= total) return "PAID";
  return dueDate && dueDate < new Date() ? "OVERDUE" : "PARTIAL";
}

/**
 * Recompute an order's totals from its line items and payments. Called after
 * every mutation so the money on the list page is never stale — the alternative
 * (trusting a client-supplied total) is how CRMs quietly lose revenue.
 */
export async function recalculateOrder(orderId: string) {
  const order = await db.order.findUnique({
    where: { id: orderId },
    include: { items: true, payments: true },
  });
  if (!order) return null;

  const subtotal = order.items.reduce(
    (sum, item) => sum + toNumber(item.unitPrice) * item.quantity,
    0,
  );
  const total =
    subtotal + toNumber(order.taxAmount) + toNumber(order.shipping) - toNumber(order.discount);
  const paid = order.payments.reduce((sum, payment) => sum + toNumber(payment.amount), 0);

  return db.order.update({
    where: { id: orderId },
    data: {
      subtotal,
      total,
      amountPaid: paid,
      paymentStatus: derivePaymentStatus(total, paid, order.paymentDueDate),
    },
  });
}
