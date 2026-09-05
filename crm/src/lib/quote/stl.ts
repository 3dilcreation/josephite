/**
 * Minimal STL reader — binary and ASCII — returning the numbers a quote needs:
 * enclosed volume, bounding box and surface area.
 *
 * Volume comes from the signed-tetrahedron sum: for each triangle, the signed
 * volume of the tetrahedron it forms with the origin. Summed over a closed
 * surface those cancel out to the enclosed volume, whatever the mesh's position
 * relative to the origin. The absolute value handles meshes wound inside-out.
 *
 * The mesh is not repaired. An open or non-manifold mesh still produces a
 * number, and that number may be wrong — which is what `looksClosed` is for.
 */

export type MeshStats = {
  triangleCount: number;
  /** Enclosed volume in cm³. */
  volumeCm3: number;
  /** Bounding box in mm. STL carries no units; mm is the universal convention. */
  bboxXmm: number;
  bboxYmm: number;
  bboxZmm: number;
  /** Surface area in cm², used to estimate FDM shell material. */
  surfaceAreaCm2: number;
  /**
   * Every edge of a closed mesh is shared by exactly two triangles. Where that
   * does not hold, the volume figure should not be trusted.
   */
  looksClosed: boolean;
};

function isBinary(buffer: Buffer) {
  // An ASCII STL starts with "solid" — but so do binary files from careless
  // exporters. The reliable test is whether the declared triangle count
  // accounts for exactly the file's length.
  if (buffer.length < 84) return false;
  const declared = buffer.readUInt32LE(80);
  return buffer.length === 84 + declared * 50;
}

function parseBinary(buffer: Buffer): Float64Array[] {
  const count = buffer.readUInt32LE(80);
  const triangles: Float64Array[] = [];
  let offset = 84;

  for (let i = 0; i < count; i += 1) {
    // The 12-byte normal is skipped: exporters get it wrong often, and the
    // volume calculation does not need it.
    const vertices = new Float64Array(9);
    for (let v = 0; v < 9; v += 1) {
      vertices[v] = buffer.readFloatLE(offset + 12 + v * 4);
    }
    triangles.push(vertices);
    offset += 50;
  }
  return triangles;
}

function parseAscii(text: string): Float64Array[] {
  const numbers: number[] = [];
  const vertexPattern = /vertex\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)/g;

  let match: RegExpExecArray | null;
  while ((match = vertexPattern.exec(text)) !== null) {
    numbers.push(Number(match[1]), Number(match[2]), Number(match[3]));
  }

  const triangles: Float64Array[] = [];
  for (let i = 0; i + 8 < numbers.length; i += 9) {
    triangles.push(Float64Array.from(numbers.slice(i, i + 9)));
  }
  return triangles;
}

export function readStl(buffer: Buffer): MeshStats {
  const triangles = isBinary(buffer) ? parseBinary(buffer) : parseAscii(buffer.toString("utf8"));

  if (triangles.length === 0) {
    throw new Error("No triangles found — is this a valid STL file?");
  }

  let signedVolumeMm3 = 0;
  let areaMm2 = 0;
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  // Edge tally for the closed-mesh check, keyed on rounded coordinates so float
  // noise from the exporter does not split one shared edge into two.
  const edges = new Map<string, number>();
  const key = (x: number, y: number, z: number) =>
    `${Math.round(x * 1000)},${Math.round(y * 1000)},${Math.round(z * 1000)}`;

  for (const t of triangles) {
    const ax = t[0];
    const ay = t[1];
    const az = t[2];
    const bx = t[3];
    const by = t[4];
    const bz = t[5];
    const cx = t[6];
    const cy = t[7];
    const cz = t[8];

    // Signed volume of tetrahedron (origin, a, b, c) = a · (b × c) / 6.
    signedVolumeMm3 +=
      (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6;

    // Triangle area = |(b − a) × (c − a)| / 2.
    const ux = bx - ax;
    const uy = by - ay;
    const uz = bz - az;
    const vx = cx - ax;
    const vy = cy - ay;
    const vz = cz - az;
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    areaMm2 += Math.sqrt(nx * nx + ny * ny + nz * nz) / 2;

    minX = Math.min(minX, ax, bx, cx);
    minY = Math.min(minY, ay, by, cy);
    minZ = Math.min(minZ, az, bz, cz);
    maxX = Math.max(maxX, ax, bx, cx);
    maxY = Math.max(maxY, ay, by, cy);
    maxZ = Math.max(maxZ, az, bz, cz);

    const ka = key(ax, ay, az);
    const kb = key(bx, by, bz);
    const kc = key(cx, cy, cz);
    for (const pair of [
      [ka, kb],
      [kb, kc],
      [kc, ka],
    ]) {
      const edge = pair[0] < pair[1] ? `${pair[0]}|${pair[1]}` : `${pair[1]}|${pair[0]}`;
      edges.set(edge, (edges.get(edge) ?? 0) + 1);
    }
  }

  let closed = true;
  for (const count of edges.values()) {
    if (count !== 2) {
      closed = false;
      break;
    }
  }

  return {
    triangleCount: triangles.length,
    volumeCm3: Math.abs(signedVolumeMm3) / 1000,
    bboxXmm: maxX - minX,
    bboxYmm: maxY - minY,
    bboxZmm: maxZ - minZ,
    surfaceAreaCm2: areaMm2 / 100,
    looksClosed: closed,
  };
}
