import { requireRole } from "@/lib/auth";
import { pricingFor } from "@/lib/quote/service";
import { updatePricing } from "@/actions/quotes";
import { toNumber } from "@/lib/format";
import { PageHeader } from "@/components/ui";
import { Field } from "@/components/forms";

export const dynamic = "force-dynamic";

export default async function PricingPage() {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");
  const pricing = await pricingFor(admin.orgId);

  return (
    <>
      <PageHeader
        title="Pricing rules"
        subtitle="The knobs that turn measured cost into the number a customer sees."
      />

      <form action={updatePricing} className="card max-w-2xl space-y-4 p-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="Setup fee per job (₹)"
            name="setupFee"
            type="number"
            step="1"
            defaultValue={toNumber(pricing.setupFee)}
          />
          <Field
            label="Minimum charge (₹)"
            name="minimumCharge"
            type="number"
            step="1"
            defaultValue={toNumber(pricing.minimumCharge)}
          />
          <Field
            label="Labour rate per hour (₹)"
            name="labourRatePerHour"
            type="number"
            step="1"
            defaultValue={toNumber(pricing.labourRatePerHour)}
          />
          <Field
            label="Post-processing minutes per part"
            name="postProcessMinutes"
            type="number"
            step="1"
            defaultValue={pricing.postProcessMinutes}
          />
          <Field
            label="Margin %"
            name="marginPercent"
            type="number"
            step="0.5"
            defaultValue={toNumber(pricing.marginPercent)}
          />
          <Field
            label="Rush multiplier"
            name="rushMultiplier"
            type="number"
            step="0.05"
            defaultValue={toNumber(pricing.rushMultiplier)}
          />
          <Field
            label="Flag for review above (₹)"
            name="reviewThreshold"
            type="number"
            step="100"
            defaultValue={toNumber(pricing.reviewThreshold)}
          />
          <Field
            label="Quote valid for (days)"
            name="quoteValidDays"
            type="number"
            step="1"
            defaultValue={pricing.quoteValidDays}
          />
        </div>

        <p className="text-xs text-ink-500">
          Setup is charged once per job, which is what makes fifty pieces cheaper each than one. The
          minimum charge floors trivially small jobs that would otherwise price below the cost of
          handling them.
        </p>

        <button type="submit" className="btn btn-accent">
          Save pricing
        </button>
      </form>
    </>
  );
}
