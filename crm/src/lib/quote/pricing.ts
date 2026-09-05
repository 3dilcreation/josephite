import type { PrintTechnology } from "@prisma/client";
import type { MeshStats } from "@/lib/quote/stl";

/**
 * Turns a measured mesh plus a set of rates into a price.
 *
 * Be clear about what this is: a defensible estimate, not a slicer. Real print
 * time depends on geometry the mesh volume cannot express — tall thin parts are
 * slow for their volume, flat wide ones are fast. Expect roughly ±25% against
 * actual slicer output until `cm3PerHour` has been calibrated per machine from
 * real jobs, which is why quotes above the review threshold are flagged for a
 * human rather than sent automatically.
 */

export type Rates = {
  densityGramsPerCm3: number;
  costPerGram: number;
  wastePercent: number;
  hourlyRate: number;
  cm3PerHour: number;
  referenceLayerMm: number;
  setupFee: number;
  minimumCharge: number;
  labourRatePerHour: number;
  postProcessMinutes: number;
  marginPercent: number;
  rushMultiplier: number;
  reviewThreshold: number;
};

export type QuoteInput = {
  mesh: MeshStats;
  technology: PrintTechnology;
  quantity: number;
  infillPercent: number;
  layerHeightMm: number;
  isRush: boolean;
  rates: Rates;
};

export type QuoteBreakdown = {
  gramsPerPart: number;
  hoursPerPart: number;
  materialCost: number;
  machineCost: number;
  labourCost: number;
  setupCost: number;
  totalCost: number;
  unitPrice: number;
  total: number;
  needsReview: boolean;
  warnings: string[];
};

// How much extra material each process throws away on supports and rafts.
const SUPPORT_ALLOWANCE: Record<PrintTechnology, number> = {
  FDM: 0.12,
  SLA: 0.25, // resin supports are dense and always discarded
  DLP: 0.25,
  SLS: 0.0, // powder bed self-supports
  MJF: 0.0,
  DMLS: 0.3, // metal supports are substantial
  POLYJET: 0.35, // full soluble support envelope
  OTHER: 0.15,
};

/**
 * Material actually consumed, in cm³.
 *
 * Only extrusion processes benefit from infill; a resin or powder part is solid
 * whatever the slider says. For FDM the walls, top and bottom are solid
 * regardless of infill, and their share scales with surface area — which is why
 * a thin-walled part can consume more material at 20% infill than a chunky one.
 */
function consumedVolumeCm3(input: QuoteInput) {
  const { mesh, technology, infillPercent } = input;
  const support = 1 + SUPPORT_ALLOWANCE[technology];

  if (technology !== "FDM") {
    return mesh.volumeCm3 * support;
  }

  // Two solid perimeters plus solid top and bottom, approximated as a shell one
  // millimetre thick over the model's surface, capped at the part's own volume.
  const shellCm3 = Math.min(mesh.surfaceAreaCm2 * 0.1, mesh.volumeCm3);
  const infillCm3 = Math.max(mesh.volumeCm3 - shellCm3, 0) * (infillPercent / 100);
  return (shellCm3 + infillCm3) * support;
}

export function priceQuote(input: QuoteInput): QuoteBreakdown {
  const { mesh, rates, quantity, layerHeightMm } = input;
  const warnings: string[] = [];

  const consumedCm3 = consumedVolumeCm3(input);
  const gramsPerPart =
    consumedCm3 * rates.densityGramsPerCm3 * (1 + rates.wastePercent / 100);

  // Throughput is quoted at a reference layer height. Halving the layer height
  // roughly doubles the number of layers, and so the time.
  const layerFactor = layerHeightMm > 0 ? rates.referenceLayerMm / layerHeightMm : 1;
  const printHours = (consumedCm3 / Math.max(rates.cm3PerHour, 0.1)) * layerFactor;
  // Bed preparation, heat-up and removal, whatever the part's size.
  const hoursPerPart = printHours + 0.25;

  const materialCost = gramsPerPart * rates.costPerGram * quantity;
  const machineCost = hoursPerPart * rates.hourlyRate * quantity;
  const labourCost = (rates.postProcessMinutes / 60) * rates.labourRatePerHour * quantity;
  // Setup is per job, not per part — the reason 50 pieces cost less each.
  const setupCost = rates.setupFee;

  const totalCost = materialCost + machineCost + labourCost + setupCost;
  const withMargin = totalCost * (1 + rates.marginPercent / 100);
  const withRush = input.isRush ? withMargin * rates.rushMultiplier : withMargin;
  const total = Math.max(withRush, rates.minimumCharge);

  if (!mesh.looksClosed) {
    warnings.push(
      "This mesh is not watertight, so its volume — and therefore this price — may be wrong. Repair the file before quoting.",
    );
  }
  if (mesh.volumeCm3 <= 0) {
    warnings.push("Measured volume is zero. The file is probably a surface, not a solid.");
  }
  if (total >= rates.reviewThreshold) {
    warnings.push("Above the review threshold — check this by hand before sending it out.");
  }

  return {
    gramsPerPart,
    hoursPerPart,
    materialCost,
    machineCost,
    labourCost,
    setupCost,
    totalCost,
    unitPrice: total / Math.max(quantity, 1),
    total,
    needsReview: total >= rates.reviewThreshold || !mesh.looksClosed,
    warnings,
  };
}

/** Largest dimension against a machine's build envelope, allowing rotation. */
export function fitsInBuildVolume(
  mesh: MeshStats,
  machine: { buildXmm: number | null; buildYmm: number | null; buildZmm: number | null },
) {
  if (!machine.buildXmm || !machine.buildYmm || !machine.buildZmm) return true;
  const part = [mesh.bboxXmm, mesh.bboxYmm, mesh.bboxZmm].sort((a, b) => a - b);
  const bed = [machine.buildXmm, machine.buildYmm, machine.buildZmm].sort((a, b) => a - b);
  return part[0] <= bed[0] && part[1] <= bed[1] && part[2] <= bed[2];
}
