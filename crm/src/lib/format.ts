import type { Prisma } from "@prisma/client";

type Money = Prisma.Decimal | number | string | null | undefined;

export function toNumber(value: Money): number {
  if (value === null || value === undefined) return 0;
  return typeof value === "number" ? value : Number(value.toString());
}

export function formatMoney(value: Money, currency = "INR") {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency,
    maximumFractionDigits: 0,
  }).format(toNumber(value));
}

export function formatDate(value: Date | string | null | undefined) {
  if (!value) return "—";
  const date = typeof value === "string" ? new Date(value) : value;
  return new Intl.DateTimeFormat("en-IN", { day: "2-digit", month: "short", year: "numeric" }).format(date);
}

export function formatDateTime(value: Date | string | null | undefined) {
  if (!value) return "—";
  const date = typeof value === "string" ? new Date(value) : value;
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function relativeDays(value: Date | string | null | undefined) {
  if (!value) return null;
  const date = typeof value === "string" ? new Date(value) : value;
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const target = new Date(date);
  target.setHours(0, 0, 0, 0);
  return Math.round((target.getTime() - start.getTime()) / 86_400_000);
}

// Title-casing every enum value would print "Upi" and "Indiamart", which look
// like typos to the people who use these words daily.
const LABELS: Record<string, string> = {
  UPI: "UPI",
  COD: "COD",
  GST: "GST",
  INDIAMART: "IndiaMART",
  WHATSAPP: "WhatsApp",
  FACEBOOK_LEADS: "Facebook Lead Ads",
  GOOGLE_ADS: "Google Ads",
  WOOCOMMERCE: "WooCommerce",
  FDM: "FDM",
  SLA: "SLA",
  SLS: "SLS",
  MJF: "MJF",
  DMLS: "DMLS",
  DLP: "DLP",
  POLYJET: "PolyJet",
  HQ: "Head Office",
  QUALITY_CHECK: "Quality Check",
  BANK_TRANSFER: "Bank Transfer",
};

/** Turn SCREAMING_SNAKE enum values into readable labels. */
export function humanise(value: string) {
  if (LABELS[value]) return LABELS[value];
  return value
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

/** "1 day" / "3 days" — small thing, but "1 days late" reads like a bug. */
export function pluralDays(count: number) {
  return `${count} ${count === 1 ? "day" : "days"}`;
}
