/**
 * Geometry checks against shapes with known analytic volumes. A quoting engine
 * that measures volume wrong loses money on every job silently, so these are
 * the numbers worth pinning down.
 *
 * Run with: npx tsx tests/stl.test.ts
 */
import { strict as assert } from "node:assert";
import { readStl } from "../src/lib/quote/stl";
import { priceQuote, fitsInBuildVolume, type Rates } from "../src/lib/quote/pricing";

let passed = 0;
function test(name: string, fn: () => void) {
  try {
    fn();
    passed += 1;
    console.log(`  ✓ ${name}`);
  } catch (error) {
    console.error(`  ✗ ${name}`);
    console.error(`    ${(error as Error).message}`);
    process.exitCode = 1;
  }
}

const close = (actual: number, expected: number, tolerance: number, label: string) =>
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${label}: expected ~${expected}, got ${actual}`,
  );

// --- fixtures ---------------------------------------------------------------

type Tri = [number[], number[], number[]];

function binaryStl(triangles: Tri[]): Buffer {
  const buffer = Buffer.alloc(84 + triangles.length * 50);
  buffer.writeUInt32LE(triangles.length, 80);
  let offset = 84;
  for (const [a, b, c] of triangles) {
    offset += 12; // normal left as zeroes
    for (const vertex of [a, b, c]) {
      for (const component of vertex) {
        buffer.writeFloatLE(component, offset);
        offset += 4;
      }
    }
    offset += 2; // attribute byte count
  }
  return buffer;
}

/** Axis-aligned box from (0,0,0) to (x,y,z), wound outwards. */
function boxTriangles(x: number, y: number, z: number): Tri[] {
  const v = [
    [0, 0, 0],
    [x, 0, 0],
    [x, y, 0],
    [0, y, 0],
    [0, 0, z],
    [x, 0, z],
    [x, y, z],
    [0, y, z],
  ];
  const faces: [number, number, number][] = [
    [0, 2, 1], [0, 3, 2], // bottom
    [4, 5, 6], [4, 6, 7], // top
    [0, 1, 5], [0, 5, 4], // front
    [1, 2, 6], [1, 6, 5], // right
    [2, 3, 7], [2, 7, 6], // back
    [3, 0, 4], [3, 4, 7], // left
  ];
  return faces.map(([a, b, c]) => [v[a], v[b], v[c]] as Tri);
}

function asciiStl(triangles: Tri[]): Buffer {
  const body = triangles
    .map(
      ([a, b, c]) =>
        `facet normal 0 0 0\n  outer loop\n${[a, b, c]
          .map((p) => `    vertex ${p[0]} ${p[1]} ${p[2]}`)
          .join("\n")}\n  endloop\nendfacet`,
    )
    .join("\n");
  return Buffer.from(`solid test\n${body}\nendsolid test\n`, "utf8");
}

// --- geometry ---------------------------------------------------------------

console.log("STL geometry");

test("binary cube: 20mm cube is 8 cm³", () => {
  const stats = readStl(binaryStl(boxTriangles(20, 20, 20)));
  assert.equal(stats.triangleCount, 12);
  close(stats.volumeCm3, 8, 0.001, "volume");
  close(stats.bboxXmm, 20, 0.001, "bbox x");
  close(stats.surfaceAreaCm2, 24, 0.001, "surface area"); // 6 faces × 400mm² = 2400mm²
  assert.equal(stats.looksClosed, true);
});

test("ascii cube gives the same answer as binary", () => {
  const binary = readStl(binaryStl(boxTriangles(20, 20, 20)));
  const ascii = readStl(asciiStl(boxTriangles(20, 20, 20)));
  close(ascii.volumeCm3, binary.volumeCm3, 0.001, "volume");
  assert.equal(ascii.triangleCount, binary.triangleCount);
});

test("non-cubic box: 50×30×10mm is 15 cm³", () => {
  const stats = readStl(binaryStl(boxTriangles(50, 30, 10)));
  close(stats.volumeCm3, 15, 0.001, "volume");
  close(stats.bboxXmm, 50, 0.001, "bbox x");
  close(stats.bboxZmm, 10, 0.001, "bbox z");
});

test("volume is independent of position relative to the origin", () => {
  const shifted = boxTriangles(20, 20, 20).map(
    (tri) => tri.map((p) => [p[0] + 500, p[1] - 300, p[2] + 90]) as Tri,
  );
  close(readStl(binaryStl(shifted)).volumeCm3, 8, 0.001, "shifted volume");
});

test("inside-out winding still reports a positive volume", () => {
  const flipped = boxTriangles(20, 20, 20).map((tri) => [tri[0], tri[2], tri[1]] as Tri);
  close(readStl(binaryStl(flipped)).volumeCm3, 8, 0.001, "flipped volume");
});

test("an open mesh is reported as not closed", () => {
  const openBox = boxTriangles(20, 20, 20).slice(0, 10); // drop one face
  const stats = readStl(binaryStl(openBox));
  assert.equal(stats.looksClosed, false);
});

test("an empty or junk file is rejected rather than priced", () => {
  assert.throws(() => readStl(Buffer.from("not an stl at all", "utf8")));
});

// --- pricing ----------------------------------------------------------------

console.log("Pricing");

const rates: Rates = {
  densityGramsPerCm3: 1.24, // PLA
  costPerGram: 1.6,
  wastePercent: 10,
  hourlyRate: 60,
  cm3PerHour: 15,
  referenceLayerMm: 0.2,
  setupFee: 250,
  minimumCharge: 500,
  labourRatePerHour: 250,
  postProcessMinutes: 15,
  marginPercent: 45,
  rushMultiplier: 1.5,
  reviewThreshold: 25000,
};

const cube = readStl(binaryStl(boxTriangles(20, 20, 20)));

test("solid resin part consumes its full volume plus supports", () => {
  const result = priceQuote({
    mesh: cube,
    technology: "SLA",
    quantity: 1,
    infillPercent: 20,
    layerHeightMm: 0.2,
    isRush: false,
    rates: { ...rates, densityGramsPerCm3: 1.1, wastePercent: 0 },
  });
  // 8 cm³ × 1.25 support allowance × 1.1 g/cm³ = 11 g
  close(result.gramsPerPart, 11, 0.01, "grams");
});

test("infill reduces FDM material, and does so monotonically", () => {
  const at = (infill: number) =>
    priceQuote({
      mesh: cube,
      technology: "FDM",
      quantity: 1,
      infillPercent: infill,
      layerHeightMm: 0.2,
      isRush: false,
      rates,
    }).gramsPerPart;

  assert.ok(at(15) < at(50), "15% infill should use less than 50%");
  assert.ok(at(50) < at(100), "50% infill should use less than solid");
  // Even at 0% the shell is still printed, so material never reaches zero.
  assert.ok(at(0) > 0, "a hollow part still has walls");
});

test("halving the layer height roughly doubles machine time", () => {
  const coarse = priceQuote({
    mesh: cube, technology: "FDM", quantity: 1, infillPercent: 20,
    layerHeightMm: 0.2, isRush: false, rates,
  });
  const fine = priceQuote({
    mesh: cube, technology: "FDM", quantity: 1, infillPercent: 20,
    layerHeightMm: 0.1, isRush: false, rates,
  });
  // Fixed handling time is in both, so the ratio sits just under 2.
  const ratio = (fine.hoursPerPart - 0.25) / (coarse.hoursPerPart - 0.25);
  close(ratio, 2, 0.01, "time ratio");
});

test("setup is charged once, so unit price falls with quantity", () => {
  const one = priceQuote({
    mesh: cube, technology: "FDM", quantity: 1, infillPercent: 20,
    layerHeightMm: 0.2, isRush: false, rates,
  });
  const fifty = priceQuote({
    mesh: cube, technology: "FDM", quantity: 50, infillPercent: 20,
    layerHeightMm: 0.2, isRush: false, rates,
  });
  assert.ok(fifty.unitPrice < one.unitPrice, "bulk unit price must be lower");
  close(fifty.setupCost, one.setupCost, 0.001, "setup charged once either way");
});

test("the minimum charge floors a trivially small job", () => {
  const tiny = readStl(binaryStl(boxTriangles(2, 2, 2)));
  const result = priceQuote({
    mesh: tiny, technology: "FDM", quantity: 1, infillPercent: 20,
    layerHeightMm: 0.2, isRush: false, rates,
  });
  assert.equal(result.total, rates.minimumCharge);
});

test("rush multiplier applies on top of margin", () => {
  const base = priceQuote({
    mesh: cube, technology: "FDM", quantity: 10, infillPercent: 20,
    layerHeightMm: 0.2, isRush: false, rates,
  });
  const rush = priceQuote({
    mesh: cube, technology: "FDM", quantity: 10, infillPercent: 20,
    layerHeightMm: 0.2, isRush: true, rates,
  });
  close(rush.total / base.total, rates.rushMultiplier, 0.001, "rush ratio");
});

test("price always covers cost at a positive margin", () => {
  const result = priceQuote({
    mesh: cube, technology: "FDM", quantity: 5, infillPercent: 20,
    layerHeightMm: 0.2, isRush: false, rates,
  });
  assert.ok(result.total > result.totalCost, "selling below cost");
});

test("an unwatertight mesh is flagged for review, not quietly priced", () => {
  const open = readStl(binaryStl(boxTriangles(20, 20, 20).slice(0, 10)));
  const result = priceQuote({
    mesh: open, technology: "FDM", quantity: 1, infillPercent: 20,
    layerHeightMm: 0.2, isRush: false, rates,
  });
  assert.equal(result.needsReview, true);
  assert.ok(result.warnings.some((w) => w.includes("watertight")));
});

test("build volume check allows rotation", () => {
  const long = readStl(binaryStl(boxTriangles(300, 50, 50)));
  const bed = { buildXmm: 220, buildYmm: 220, buildZmm: 250 };
  assert.equal(fitsInBuildVolume(long, bed), false);
  // The same part rotated fits a taller machine.
  assert.equal(fitsInBuildVolume(long, { buildXmm: 220, buildYmm: 220, buildZmm: 350 }), true);
});

console.log(`\n${passed} checks passed`);
