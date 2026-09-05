"use client";

import Link from "next/link";
import { useRef, useState } from "react";
import { setQuoteStatus } from "@/actions/quotes";

type Material = {
  id: string;
  name: string;
  technology: string;
  colour: string | null;
  costPerGram: number;
  stockGrams: number;
};
type Machine = { id: string; name: string; technology: string; status: string };
type Customer = { id: string; name: string; company: string | null };

type Result = {
  quoteId: string;
  quoteNo: string;
  mesh: {
    triangleCount: number;
    volumeCm3: number;
    bboxXmm: number;
    bboxYmm: number;
    bboxZmm: number;
    looksClosed: boolean;
  };
  breakdown: {
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
  };
  warnings: string[];
  material: string;
  machine: string | null;
};

const formatSize = (bytes: number) =>
  bytes >= 1_048_576 ? `${(bytes / 1_048_576).toFixed(2)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;

const money = (value: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(
    value,
  );

export function QuoteForm({
  materials,
  machines,
  customers,
  marginPercent,
}: {
  materials: Material[];
  machines: Machine[];
  customers: Customer[];
  marginPercent: number;
}) {
  const formRef = useRef<HTMLFormElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [materialId, setMaterialId] = useState(materials[0]?.id ?? "");

  const material = materials.find((m) => m.id === materialId);
  // Only extrusion processes have a meaningful infill setting; showing the
  // slider for resin would imply a control that changes nothing.
  const infillApplies = material?.technology === "FDM";
  const compatibleMachines = machines.filter((m) => m.technology === material?.technology);

  async function estimate(event?: React.FormEvent) {
    event?.preventDefault();
    if (!file) {
      setError("Choose an STL file first");
      return;
    }

    setPending(true);
    setError(null);

    const data = new FormData(formRef.current!);
    data.set("file", file);
    // Re-price the same draft rather than creating a new quote per adjustment.
    if (result) data.set("quoteId", result.quoteId);

    const response = await fetch("/api/quote/estimate", { method: "POST", body: data });
    const body = await response.json();

    if (!response.ok) {
      setError(body.error ?? "Could not price that file");
      setPending(false);
      return;
    }

    setResult(body);
    setPending(false);
  }

  return (
    <div className="grid gap-4 xl:grid-cols-5">
      <form ref={formRef} onSubmit={estimate} className="card space-y-4 p-5 xl:col-span-2">
        <div>
          <label className="label" htmlFor="file">
            Model file (.stl)
          </label>
          <input
            id="file"
            type="file"
            accept=".stl"
            required
            onChange={(event) => {
              setFile(event.target.files?.[0] ?? null);
              setResult(null);
            }}
            className="input"
          />
          {file ? (
            <p className="mt-1 text-xs text-ink-500">
              {file.name} · {formatSize(file.size)}
            </p>
          ) : null}
        </div>

        <div>
          <label className="label" htmlFor="materialId">
            Material
          </label>
          <select
            id="materialId"
            name="materialId"
            value={materialId}
            onChange={(event) => setMaterialId(event.target.value)}
            className="input"
          >
            {materials.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} · {m.technology} · ₹{m.costPerGram}/g
              </option>
            ))}
          </select>
          {material ? (
            <p className="mt-1 text-xs text-ink-500">
              {material.stockGrams.toFixed(0)} g in stock
            </p>
          ) : null}
        </div>

        <div>
          <label className="label" htmlFor="machineId">
            Machine
          </label>
          <select id="machineId" name="machineId" className="input">
            <option value="">— default rates —</option>
            {compatibleMachines.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.status.toLowerCase()})
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="label" htmlFor="quantity">
              Quantity
            </label>
            <input id="quantity" name="quantity" type="number" min={1} defaultValue={1} className="input" />
          </div>
          <div>
            <label className="label" htmlFor="layerHeightMm">
              Layer height (mm)
            </label>
            <input
              id="layerHeightMm"
              name="layerHeightMm"
              type="number"
              step="0.01"
              min="0.02"
              max="1"
              defaultValue={0.2}
              className="input"
            />
          </div>
        </div>

        <div className={infillApplies ? "" : "opacity-50"}>
          <label className="label" htmlFor="infillPercent">
            Infill %{infillApplies ? "" : " (not used by this process)"}
          </label>
          <input
            id="infillPercent"
            name="infillPercent"
            type="number"
            min={0}
            max={100}
            defaultValue={20}
            disabled={!infillApplies}
            className="input"
          />
        </div>

        <div>
          <label className="label" htmlFor="customerId">
            Customer (optional)
          </label>
          <select id="customerId" name="customerId" className="input">
            <option value="">— not linked yet —</option>
            {customers.map((c) => (
              <option key={c.id} value={c.id}>
                {[c.name, c.company].filter(Boolean).join(" · ")}
              </option>
            ))}
          </select>
        </div>

        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" name="isRush" value="true" />
          Rush job
        </label>

        <button type="submit" disabled={pending || !file} className="btn btn-accent w-full">
          {pending ? "Measuring…" : result ? "Re-price" : "Get price"}
        </button>

        {error ? (
          <p role="alert" className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </p>
        ) : null}
      </form>

      <div className="xl:col-span-3">
        {!result ? (
          <div className="card p-8 text-center">
            <p className="text-sm font-medium text-ink-700">No estimate yet</p>
            <p className="mx-auto mt-1 max-w-md text-sm text-ink-500">
              Upload a model to measure its volume and bounding box, then price it against your material
              and machine rates.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="card p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-ink-500">
                    Quote {result.quoteNo}
                  </p>
                  <p className="mt-1 text-3xl font-semibold tabular-nums text-ink-900">
                    {money(result.breakdown.total)}
                  </p>
                  <p className="text-sm text-ink-500">
                    {money(result.breakdown.unitPrice)} each · {result.material}
                    {result.machine ? ` · ${result.machine}` : ""}
                  </p>
                </div>
                <div className="flex gap-2">
                  <Link href="/quotes" className="btn btn-ghost">
                    All quotes
                  </Link>
                  <form action={setQuoteStatus}>
                    <input type="hidden" name="id" value={result.quoteId} />
                    <input type="hidden" name="status" value="SENT" />
                    <button type="submit" className="btn btn-primary">
                      Mark as sent
                    </button>
                  </form>
                </div>
              </div>

              {result.warnings.length > 0 ? (
                <ul className="mt-4 space-y-2">
                  {result.warnings.map((warning) => (
                    <li
                      key={warning}
                      className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800"
                    >
                      {warning}
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <section className="card p-5">
                <h2 className="mb-3 text-sm font-semibold text-ink-900">What we measured</h2>
                <dl className="space-y-2 text-sm">
                  <Row label="Volume" value={`${result.mesh.volumeCm3.toFixed(2)} cm³`} />
                  <Row
                    label="Bounding box"
                    value={`${result.mesh.bboxXmm.toFixed(1)} × ${result.mesh.bboxYmm.toFixed(
                      1,
                    )} × ${result.mesh.bboxZmm.toFixed(1)} mm`}
                  />
                  <Row label="Triangles" value={result.mesh.triangleCount.toLocaleString("en-IN")} />
                  <Row label="Watertight" value={result.mesh.looksClosed ? "Yes" : "No"} />
                  <Row label="Material per part" value={`${result.breakdown.gramsPerPart.toFixed(1)} g`} />
                  <Row label="Machine time per part" value={`${result.breakdown.hoursPerPart.toFixed(2)} h`} />
                </dl>
              </section>

              <section className="card p-5">
                <h2 className="mb-3 text-sm font-semibold text-ink-900">Where the money goes</h2>
                <dl className="space-y-2 text-sm">
                  <Row label="Material" value={money(result.breakdown.materialCost)} />
                  <Row label="Machine time" value={money(result.breakdown.machineCost)} />
                  <Row label="Labour" value={money(result.breakdown.labourCost)} />
                  <Row label="Setup" value={money(result.breakdown.setupCost)} />
                  <div className="border-t border-ink-100 pt-2">
                    <Row label="Your cost" value={money(result.breakdown.totalCost)} strong />
                  </div>
                  <Row
                    label={`Margin (${marginPercent}%)`}
                    value={money(result.breakdown.total - result.breakdown.totalCost)}
                  />
                  <div className="border-t border-ink-100 pt-2">
                    <Row label="Customer pays" value={money(result.breakdown.total)} strong />
                  </div>
                </dl>
              </section>
            </div>

            <p className="text-xs text-ink-500">
              Print time is estimated from volume and throughput, not from a slicer, so expect roughly
              ±25% until each machine&rsquo;s cm³/hour has been calibrated against real jobs. Material,
              labour and setup are exact against your rate tables.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function Row({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className={strong ? "font-semibold text-ink-900" : "text-ink-500"}>{label}</dt>
      <dd className={`tabular-nums ${strong ? "font-semibold text-ink-900" : "text-ink-800"}`}>{value}</dd>
    </div>
  );
}
