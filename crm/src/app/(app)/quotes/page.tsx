import Link from "next/link";
import { QuoteStatus } from "@prisma/client";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { canWrite } from "@/lib/rbac";
import { setQuoteStatus, convertQuoteToOrder } from "@/actions/quotes";
import { formatMoney, formatDate, toNumber, humanise } from "@/lib/format";
import { PageHeader, StatCard, EmptyState, Th, Td } from "@/components/ui";
import { Badge } from "@/components/badges";
import { Field, Select, enumOptions } from "@/components/forms";

export const dynamic = "force-dynamic";

const TONES: Record<string, "neutral" | "blue" | "amber" | "green" | "red"> = {
  DRAFT: "neutral",
  SENT: "blue",
  ACCEPTED: "green",
  REJECTED: "red",
  EXPIRED: "amber",
};

export default async function QuotesPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | undefined>>;
}) {
  const params = await searchParams;
  const user = await requireUser();
  const editable = canWrite(user.role);

  const quotes = await db.quote.findMany({
    where: {
      orgId: user.orgId,
      ...(params.status ? { status: params.status as never } : {}),
    },
    orderBy: { createdAt: "desc" },
    take: 100,
    include: {
      customer: { select: { id: true, name: true } },
      material: { select: { name: true } },
      createdBy: { select: { name: true } },
    },
  });

  const accepted = quotes.filter((q) => q.status === "ACCEPTED");
  const sent = quotes.filter((q) => q.status === "SENT" || q.status === "ACCEPTED");
  // Win rate over quotes that actually reached a customer; drafts are noise here.
  const winRate = sent.length > 0 ? Math.round((accepted.length / sent.length) * 100) : null;
  const pipeline = quotes
    .filter((q) => q.status === "SENT")
    .reduce((sum, q) => sum + toNumber(q.total), 0);

  return (
    <>
      <PageHeader
        title="Quotes"
        subtitle={`${quotes.length} recent`}
        action={
          <Link href="/quote" className="btn btn-accent">
            New quote
          </Link>
        }
      />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="Out with customers" value={formatMoney(pipeline)} tone="warn" />
        <StatCard label="Accepted" value={String(accepted.length)} tone="good" />
        <StatCard label="Win rate" value={winRate === null ? "—" : `${winRate}%`} hint="of quotes sent" />
        <StatCard
          label="Needs review"
          value={String(quotes.filter((q) => q.needsReview && q.status === "DRAFT").length)}
          tone={quotes.some((q) => q.needsReview && q.status === "DRAFT") ? "danger" : "neutral"}
        />
      </div>

      <form className="my-4 flex items-end gap-2" action="/quotes">
        <Select
          label="Status"
          name="status"
          includeBlank
          blankLabel="All"
          options={enumOptions(QuoteStatus)}
          defaultValue={params.status}
          className="w-48"
        />
        <button type="submit" className="btn btn-primary">
          Filter
        </button>
      </form>

      <div className="card">
        {quotes.length === 0 ? (
          <EmptyState title="No quotes yet" hint="Upload a model on the quote page to make one." />
        ) : (
          <div className="scroll-x">
            <table className="w-full">
              <thead className="border-b border-ink-100">
                <tr>
                  <Th>Quote</Th>
                  <Th>Part</Th>
                  <Th>Spec</Th>
                  <Th>Cost</Th>
                  <Th>Price</Th>
                  <Th>Status</Th>
                  {editable ? <Th>Actions</Th> : null}
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-100">
                {quotes.map((quote) => {
                  const margin = toNumber(quote.total) - toNumber(quote.totalCost);
                  const marginPct = toNumber(quote.total) > 0
                    ? Math.round((margin / toNumber(quote.total)) * 100)
                    : 0;
                  return (
                    <tr key={quote.id} className="hover:bg-ink-50">
                      <Td>
                        <span className="font-medium text-ink-900">{quote.quoteNo}</span>
                        <span className="block text-xs text-ink-500">
                          {formatDate(quote.createdAt)} · {quote.createdBy?.name ?? "—"}
                        </span>
                      </Td>
                      <Td>
                        <span className="text-ink-900">{quote.fileName}</span>
                        <span className="block text-xs text-ink-500">
                          {quote.customer?.name ?? "No customer linked"}
                        </span>
                      </Td>
                      <Td className="text-ink-500">
                        {quote.material?.name ?? humanise(quote.technology)} × {quote.quantity}
                        <span className="block text-xs">
                          {toNumber(quote.volumeCm3).toFixed(1)} cm³ ·{" "}
                          {toNumber(quote.estimatedGrams).toFixed(0)} g ·{" "}
                          {toNumber(quote.estimatedHours).toFixed(1)} h
                        </span>
                      </Td>
                      <Td className="whitespace-nowrap tabular-nums text-ink-500">
                        {formatMoney(quote.totalCost)}
                      </Td>
                      <Td className="whitespace-nowrap tabular-nums font-medium">
                        {formatMoney(quote.total)}
                        <span
                          className={`block text-xs ${marginPct < 20 ? "text-red-600" : "text-emerald-600"}`}
                        >
                          {marginPct}% margin
                        </span>
                      </Td>
                      <Td>
                        <Badge tone={TONES[quote.status]}>{humanise(quote.status)}</Badge>
                        {quote.needsReview ? (
                          <span className="mt-1 block">
                            <Badge tone="amber">Review</Badge>
                          </span>
                        ) : null}
                      </Td>
                      {editable ? (
                        <Td>
                          <div className="flex flex-col gap-1.5">
                            {quote.status === "DRAFT" ? (
                              <form action={setQuoteStatus}>
                                <input type="hidden" name="id" value={quote.id} />
                                <input type="hidden" name="status" value="SENT" />
                                <button type="submit" className="text-xs font-medium text-brand-600">
                                  Mark sent
                                </button>
                              </form>
                            ) : null}
                            {quote.status !== "ACCEPTED" && quote.status !== "REJECTED" ? (
                              <details>
                                <summary className="cursor-pointer text-xs font-medium text-brand-600">
                                  Accept → order
                                </summary>
                                <form
                                  action={convertQuoteToOrder}
                                  className="mt-2 w-56 space-y-2 rounded-lg border border-ink-100 p-2"
                                >
                                  <input type="hidden" name="id" value={quote.id} />
                                  {quote.customerId ? null : (
                                    <>
                                      <Field label="Customer name" name="customerName" required />
                                      <Field label="Phone" name="customerPhone" />
                                      <Field label="Email" name="customerEmail" type="email" />
                                    </>
                                  )}
                                  <button type="submit" className="btn btn-accent w-full">
                                    Create order
                                  </button>
                                </form>
                              </details>
                            ) : null}
                            {quote.status === "SENT" ? (
                              <form action={setQuoteStatus}>
                                <input type="hidden" name="id" value={quote.id} />
                                <input type="hidden" name="status" value="REJECTED" />
                                <button type="submit" className="text-xs font-medium text-ink-500">
                                  Mark lost
                                </button>
                              </form>
                            ) : null}
                          </div>
                        </Td>
                      ) : null}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
