"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { db } from "@/lib/db";
import { requireUser, requireRole } from "@/lib/auth";
import { canWrite } from "@/lib/rbac";
import { logActivity, notify } from "@/lib/notify";
import { nextOrderNumber, recalculateOrder } from "@/lib/orders";
import { formatMoney, toNumber } from "@/lib/format";
import { contactMatch } from "@/lib/contacts";

const optional = (value: FormDataEntryValue | null) => {
  const text = typeof value === "string" ? value.trim() : "";
  return text === "" ? undefined : text;
};

export async function setQuoteStatus(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const id = String(formData.get("id"));
  const status = String(formData.get("status"));
  const quote = await db.quote.findFirst({ where: { id, orgId: user.orgId } });
  if (!quote) throw new Error("Quote not found");

  await db.quote.update({ where: { id }, data: { status: status as never } });
  revalidatePath("/quotes");
}

/**
 * Accepting a quote should not mean retyping the job. This creates the customer
 * if needed and opens an order whose line item already carries the material,
 * technology, weight and print hours the estimate was based on — which is what
 * later lets the order report a real margin against that estimate.
 */
export async function convertQuoteToOrder(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const id = String(formData.get("id"));
  const quote = await db.quote.findFirst({
    where: { id, orgId: user.orgId },
    include: { material: true, customer: true },
  });
  if (!quote) throw new Error("Quote not found");

  let customerId = quote.customerId;
  if (!customerId) {
    const name = optional(formData.get("customerName"));
    const phone = optional(formData.get("customerPhone"));
    const email = optional(formData.get("customerEmail"));
    if (!name) throw new Error("Name the customer this order is for");

    const identifiers = contactMatch({ phone, email });
    const existing = identifiers
      ? await db.customer.findFirst({ where: { orgId: user.orgId, OR: identifiers } })
      : null;

    customerId =
      existing?.id ??
      (
        await db.customer.create({
          data: { orgId: user.orgId, branchId: user.branchId, name, phone, email, ownerId: user.id },
        })
      ).id;
  }

  const order = await db.order.create({
    data: {
      orgId: user.orgId,
      branchId: user.branchId,
      orderNo: await nextOrderNumber(user.orgId),
      customerId,
      status: "CONFIRMED",
      priority: quote.isRush ? "URGENT" : "MEDIUM",
      channel: "OFFLINE",
      sourceKind: "MANUAL",
      assignedToId: user.id,
      items: {
        create: [
          {
            name: quote.fileName.replace(/\.stl$/i, ""),
            description: `From quote ${quote.quoteNo} · ${toNumber(quote.volumeCm3).toFixed(1)} cm³ · ${
              quote.infillPercent
            }% infill · ${toNumber(quote.layerHeightMm)} mm layers`,
            technology: quote.technology,
            material: quote.material?.name,
            quantity: quote.quantity,
            unitPrice: quote.unitPrice,
            weightGrams: quote.estimatedGrams,
            printHours: quote.estimatedHours,
            layerHeightMm: quote.layerHeightMm,
            infillPercent: quote.infillPercent,
            fileName: quote.fileName,
          },
        ],
      },
    },
  });

  await recalculateOrder(order.id);
  await db.quote.update({ where: { id }, data: { status: "ACCEPTED" } });

  await logActivity({
    orgId: user.orgId,
    entityType: "ORDER",
    entityId: order.id,
    action: "order.from_quote",
    summary: `${user.name} accepted quote ${quote.quoteNo} (${formatMoney(quote.total)}) as ${order.orderNo}`,
    actorId: user.id,
  });

  const admins = await db.user.findMany({
    where: { orgId: user.orgId, isActive: true, role: { in: ["SUPER_ADMIN", "ADMIN"] } },
    select: { id: true },
  });
  await notify({
    orgId: user.orgId,
    userIds: admins.map((a) => a.id),
    type: "ORDER_ASSIGNED",
    title: `Quote ${quote.quoteNo} accepted — ${order.orderNo}`,
    body: formatMoney(quote.total),
    link: `/orders/${order.id}`,
    exceptUserId: user.id,
  });

  revalidatePath("/quotes");
  redirect(`/orders/${order.id}`);
}

// --- rate tables ------------------------------------------------------------

export async function createMaterial(formData: FormData) {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");

  await db.material.create({
    data: {
      orgId: admin.orgId,
      name: String(formData.get("name")),
      technology: (formData.get("technology") ?? "FDM") as never,
      brand: optional(formData.get("brand")),
      colour: optional(formData.get("colour")),
      densityGramsPerCm3: Number(formData.get("densityGramsPerCm3") ?? 1.24),
      costPerGram: Number(formData.get("costPerGram") ?? 1.5),
      wastePercent: Number(formData.get("wastePercent") ?? 10),
      stockGrams: Number(formData.get("stockGrams") ?? 0),
      reorderLevelGrams: Number(formData.get("reorderLevelGrams") ?? 0),
    },
  });

  revalidatePath("/admin/materials");
}

export async function updateMaterialStock(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const id = String(formData.get("id"));
  const material = await db.material.findFirst({ where: { id, orgId: user.orgId } });
  if (!material) throw new Error("Material not found");

  const delta = Number(formData.get("deltaGrams"));
  if (!Number.isFinite(delta)) throw new Error("Enter a quantity in grams");

  await db.material.update({
    where: { id },
    // Clamped at zero: stock going negative means a miscount somewhere, and a
    // negative figure would quietly corrupt every later cost calculation.
    data: { stockGrams: Math.max(0, toNumber(material.stockGrams) + delta) },
  });

  revalidatePath("/admin/materials");
}

export async function createMachine(formData: FormData) {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");

  await db.machine.create({
    data: {
      orgId: admin.orgId,
      branchId: admin.branchId,
      name: String(formData.get("name")),
      technology: (formData.get("technology") ?? "FDM") as never,
      model: optional(formData.get("model")),
      buildXmm: optional(formData.get("buildXmm")) ? Number(formData.get("buildXmm")) : null,
      buildYmm: optional(formData.get("buildYmm")) ? Number(formData.get("buildYmm")) : null,
      buildZmm: optional(formData.get("buildZmm")) ? Number(formData.get("buildZmm")) : null,
      hourlyRate: Number(formData.get("hourlyRate") ?? 60),
      cm3PerHour: Number(formData.get("cm3PerHour") ?? 15),
      referenceLayerMm: Number(formData.get("referenceLayerMm") ?? 0.2),
    },
  });

  revalidatePath("/admin/machines");
}

export async function setMachineStatus(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const id = String(formData.get("id"));
  const machine = await db.machine.findFirst({ where: { id, orgId: user.orgId } });
  if (!machine) throw new Error("Machine not found");

  await db.machine.update({
    where: { id },
    data: { status: String(formData.get("status")) as never },
  });

  revalidatePath("/admin/machines");
}

export async function updatePricing(formData: FormData) {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");

  const data = {
    setupFee: Number(formData.get("setupFee")),
    minimumCharge: Number(formData.get("minimumCharge")),
    labourRatePerHour: Number(formData.get("labourRatePerHour")),
    postProcessMinutes: Number(formData.get("postProcessMinutes")),
    marginPercent: Number(formData.get("marginPercent")),
    rushMultiplier: Number(formData.get("rushMultiplier")),
    reviewThreshold: Number(formData.get("reviewThreshold")),
    quoteValidDays: Number(formData.get("quoteValidDays")),
  };

  await db.pricingSetting.upsert({
    where: { orgId: admin.orgId },
    update: data,
    create: { orgId: admin.orgId, ...data },
  });

  revalidatePath("/admin/pricing");
}
