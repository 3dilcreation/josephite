import { db } from "@/lib/db";
import { toNumber } from "@/lib/format";
import { readStl } from "@/lib/quote/stl";
import { priceQuote, fitsInBuildVolume, type Rates } from "@/lib/quote/pricing";
import type { CurrentUser } from "@/lib/auth";
import type { PrintTechnology } from "@prisma/client";

export type EstimateParams = {
  technology: PrintTechnology;
  materialId?: string | null;
  machineId?: string | null;
  quantity: number;
  infillPercent: number;
  layerHeightMm: number;
  isRush: boolean;
  customerId?: string | null;
  quoteId?: string | null;
};

export async function nextQuoteNumber(orgId: string) {
  const prefix = `QT-${new Date().getFullYear()}-`;
  const last = await db.quote.findFirst({
    where: { orgId, quoteNo: { startsWith: prefix } },
    orderBy: { quoteNo: "desc" },
    select: { quoteNo: true },
  });
  const sequence = last ? Number(last.quoteNo.slice(prefix.length)) + 1 : 1;
  return `${prefix}${String(sequence).padStart(4, "0")}`;
}

/** Defaults used when an organisation has not set its own pricing yet. */
export async function pricingFor(orgId: string) {
  return (
    (await db.pricingSetting.findUnique({ where: { orgId } })) ??
    db.pricingSetting.create({ data: { orgId } })
  );
}

/**
 * Parse the uploaded model, price it, and persist the result as a draft quote.
 *
 * The mesh is measured here rather than in the browser, and the price is
 * recomputed from the stored measurements on every re-quote, so a tampered
 * client cannot talk the system into a number the rates do not support.
 */
export async function estimateAndSave(
  user: CurrentUser,
  file: { name: string; buffer: Buffer },
  params: EstimateParams,
) {
  const mesh = readStl(file.buffer);

  const [material, machine, pricing] = await Promise.all([
    params.materialId
      ? db.material.findFirst({ where: { id: params.materialId, orgId: user.orgId } })
      : null,
    params.machineId
      ? db.machine.findFirst({ where: { id: params.machineId, orgId: user.orgId } })
      : null,
    pricingFor(user.orgId),
  ]);

  if (!material) throw new Error("Choose a material — its density and cost drive the price");

  const rates: Rates = {
    densityGramsPerCm3: toNumber(material.densityGramsPerCm3),
    costPerGram: toNumber(material.costPerGram),
    wastePercent: toNumber(material.wastePercent),
    hourlyRate: machine ? toNumber(machine.hourlyRate) : 60,
    cm3PerHour: machine ? toNumber(machine.cm3PerHour) : 15,
    referenceLayerMm: machine ? toNumber(machine.referenceLayerMm) : 0.2,
    setupFee: toNumber(pricing.setupFee),
    minimumCharge: toNumber(pricing.minimumCharge),
    labourRatePerHour: toNumber(pricing.labourRatePerHour),
    postProcessMinutes: pricing.postProcessMinutes,
    marginPercent: toNumber(pricing.marginPercent),
    rushMultiplier: toNumber(pricing.rushMultiplier),
    reviewThreshold: toNumber(pricing.reviewThreshold),
  };

  const breakdown = priceQuote({
    mesh,
    technology: material.technology,
    quantity: Math.max(1, params.quantity),
    infillPercent: params.infillPercent,
    layerHeightMm: params.layerHeightMm,
    isRush: params.isRush,
    rates,
  });

  const warnings = [...breakdown.warnings];
  if (machine && !fitsInBuildVolume(mesh, machine)) {
    warnings.push(
      `This part does not fit ${machine.name}'s build volume — it will need splitting or a larger machine.`,
    );
  }
  if (material.stockGrams.lessThan(breakdown.gramsPerPart * params.quantity)) {
    warnings.push(
      `Only ${toNumber(material.stockGrams).toFixed(0)} g of ${material.name} in stock; this job needs ${(
        breakdown.gramsPerPart * params.quantity
      ).toFixed(0)} g.`,
    );
  }

  const validUntil = new Date();
  validUntil.setDate(validUntil.getDate() + pricing.quoteValidDays);

  const data = {
    orgId: user.orgId,
    branchId: user.branchId,
    customerId: params.customerId || null,
    fileName: file.name,
    fileSizeBytes: file.buffer.length,
    triangleCount: mesh.triangleCount,
    volumeCm3: mesh.volumeCm3,
    bboxXmm: mesh.bboxXmm,
    bboxYmm: mesh.bboxYmm,
    bboxZmm: mesh.bboxZmm,
    technology: material.technology,
    materialId: material.id,
    machineId: machine?.id ?? null,
    quantity: Math.max(1, params.quantity),
    infillPercent: params.infillPercent,
    layerHeightMm: params.layerHeightMm,
    isRush: params.isRush,
    estimatedGrams: breakdown.gramsPerPart,
    estimatedHours: breakdown.hoursPerPart,
    materialCost: breakdown.materialCost,
    machineCost: breakdown.machineCost,
    labourCost: breakdown.labourCost,
    setupCost: breakdown.setupCost,
    totalCost: breakdown.totalCost,
    unitPrice: breakdown.unitPrice,
    total: breakdown.total,
    needsReview: breakdown.needsReview,
    validUntil,
    createdById: user.id,
  };

  // Re-quoting the same upload with different settings updates the draft rather
  // than littering the list with a row per slider move.
  const existing = params.quoteId
    ? await db.quote.findFirst({
        where: { id: params.quoteId, orgId: user.orgId, status: "DRAFT" },
      })
    : null;

  const quote = existing
    ? await db.quote.update({ where: { id: existing.id }, data })
    : await db.quote.create({
        data: { ...data, quoteNo: await nextQuoteNumber(user.orgId) },
      });

  return { quote, mesh, breakdown, warnings, material, machine };
}
