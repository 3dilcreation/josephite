import { humanise } from "@/lib/format";

const TONES = {
  neutral: "bg-ink-100 text-ink-700",
  blue: "bg-blue-50 text-blue-700",
  amber: "bg-amber-50 text-amber-700",
  green: "bg-emerald-50 text-emerald-700",
  red: "bg-red-50 text-red-700",
  purple: "bg-violet-50 text-violet-700",
} as const;

type Tone = keyof typeof TONES;

export function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: Tone }) {
  return (
    <span
      className={`inline-flex items-center whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-semibold ${TONES[tone]}`}
    >
      {children}
    </span>
  );
}

const LEAD_TONES: Record<string, Tone> = {
  NEW: "blue",
  CONTACTED: "purple",
  QUALIFIED: "purple",
  QUOTED: "amber",
  NEGOTIATION: "amber",
  WON: "green",
  LOST: "red",
};

const ORDER_TONES: Record<string, Tone> = {
  DRAFT: "neutral",
  CONFIRMED: "blue",
  DESIGN: "purple",
  PRINTING: "amber",
  POST_PROCESSING: "amber",
  QUALITY_CHECK: "amber",
  READY: "green",
  SHIPPED: "green",
  DELIVERED: "green",
  ON_HOLD: "red",
  CANCELLED: "red",
};

const PAYMENT_TONES: Record<string, Tone> = {
  UNPAID: "red",
  PARTIAL: "amber",
  PAID: "green",
  OVERDUE: "red",
  REFUNDED: "neutral",
};

const PRIORITY_TONES: Record<string, Tone> = {
  LOW: "neutral",
  MEDIUM: "blue",
  HIGH: "amber",
  URGENT: "red",
};

export const LeadStatusBadge = ({ value }: { value: string }) => (
  <Badge tone={LEAD_TONES[value] ?? "neutral"}>{humanise(value)}</Badge>
);

export const OrderStatusBadge = ({ value }: { value: string }) => (
  <Badge tone={ORDER_TONES[value] ?? "neutral"}>{humanise(value)}</Badge>
);

export const PaymentBadge = ({ value }: { value: string }) => (
  <Badge tone={PAYMENT_TONES[value] ?? "neutral"}>{humanise(value)}</Badge>
);

export const PriorityBadge = ({ value }: { value: string }) => (
  <Badge tone={PRIORITY_TONES[value] ?? "neutral"}>{humanise(value)}</Badge>
);

export const ChannelBadge = ({ value }: { value: string }) => (
  <Badge tone={value === "ONLINE" ? "blue" : "neutral"}>{humanise(value)}</Badge>
);
