import { NextResponse } from "next/server";
import { getCurrentUser } from "@/lib/auth";
import { canWrite } from "@/lib/rbac";
import { estimateAndSave } from "@/lib/quote/service";

export const dynamic = "force-dynamic";

// Large enough for a detailed production model, small enough that a stray
// upload cannot exhaust memory. Parsing is O(triangles) and holds the file.
const MAX_BYTES = 40 * 1024 * 1024;

export async function POST(request: Request) {
  const user = await getCurrentUser();
  if (!user) return NextResponse.json({ error: "Sign in first" }, { status: 401 });
  if (!canWrite(user.role)) {
    return NextResponse.json({ error: "Read-only access" }, { status: 403 });
  }

  const form = await request.formData();
  const file = form.get("file");

  if (!(file instanceof File)) {
    return NextResponse.json({ error: "Attach an STL file" }, { status: 400 });
  }
  if (file.size > MAX_BYTES) {
    return NextResponse.json(
      { error: `That file is ${(file.size / 1_048_576).toFixed(0)} MB; the limit is 40 MB.` },
      { status: 413 },
    );
  }
  if (!/\.stl$/i.test(file.name)) {
    return NextResponse.json(
      { error: "Only .stl files can be measured. Export your model as STL." },
      { status: 415 },
    );
  }

  // Number(null) is 0, not NaN, so an absent field must be checked for
  // explicitly — otherwise omitting "infillPercent" quietly quotes a hollow
  // part, and omitting "layerHeightMm" quotes ten times the machine time.
  const num = (key: string, fallback: number) => {
    const raw = form.get(key);
    if (raw === null || raw === "") return fallback;
    const value = Number(raw);
    return Number.isFinite(value) ? value : fallback;
  };

  try {
    const result = await estimateAndSave(
      user,
      { name: file.name, buffer: Buffer.from(await file.arrayBuffer()) },
      {
        technology: "FDM", // overridden by the material's own technology
        materialId: String(form.get("materialId") ?? "") || null,
        machineId: String(form.get("machineId") ?? "") || null,
        quantity: Math.max(1, Math.round(num("quantity", 1))),
        infillPercent: Math.min(100, Math.max(0, Math.round(num("infillPercent", 20)))),
        layerHeightMm: Math.min(1, Math.max(0.02, num("layerHeightMm", 0.2))),
        isRush: form.get("isRush") === "true",
        customerId: String(form.get("customerId") ?? "") || null,
        quoteId: String(form.get("quoteId") ?? "") || null,
      },
    );

    return NextResponse.json({
      quoteId: result.quote.id,
      quoteNo: result.quote.quoteNo,
      mesh: result.mesh,
      breakdown: result.breakdown,
      warnings: result.warnings,
      material: result.material.name,
      machine: result.machine?.name ?? null,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not read that file";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
