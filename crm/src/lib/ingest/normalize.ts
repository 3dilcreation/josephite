import type { SourceKind, Channel, Priority } from "@prisma/client";

/**
 * Every channel speaks its own JSON. Rather than teach the CRM about each one,
 * an adapter flattens the payload into one of two shapes and the rest of the
 * system only ever sees these. Adding Etsy or a new marketplace means adding a
 * mapper below — nothing downstream changes.
 */
export type NormalizedLead = {
  type: "lead";
  externalId?: string;
  externalUrl?: string;
  title: string;
  contactName?: string;
  email?: string;
  phone?: string;
  company?: string;
  requirement?: string;
  estimatedValue?: number;
  priority?: Priority;
  channel: Channel;
};

export type NormalizedOrder = {
  type: "order";
  externalId: string;
  externalUrl?: string;
  contactName: string;
  email?: string;
  phone?: string;
  shippingAddress?: string;
  currency?: string;
  total: number;
  amountPaid?: number;
  channel: Channel;
  items: {
    name: string;
    quantity: number;
    unitPrice: number;
    material?: string;
    colour?: string;
    notes?: string;
  }[];
};

export type Normalized = NormalizedLead | NormalizedOrder;

type Payload = Record<string, any>;

const str = (...candidates: unknown[]) => {
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) return candidate.trim();
    if (typeof candidate === "number") return String(candidate);
  }
  return undefined;
};

const num = (...candidates: unknown[]) => {
  for (const candidate of candidates) {
    const parsed = Number(candidate);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
};

function websiteLead(p: Payload): NormalizedLead {
  return {
    type: "lead",
    externalId: str(p.id, p.submission_id),
    externalUrl: str(p.url, p.page_url),
    title: str(p.subject, p.service, p.title) ?? "Website enquiry",
    contactName: str(p.name, p.full_name, p.contact_name),
    email: str(p.email),
    phone: str(p.phone, p.mobile, p.contact_number),
    company: str(p.company, p.organisation, p.organization),
    requirement: str(p.message, p.requirement, p.description, p.comments),
    estimatedValue: num(p.budget, p.estimated_value),
    channel: "ONLINE",
  };
}

function indiamartLead(p: Payload): NormalizedLead {
  // IndiaMART's push API uses its own SENDER_* field names.
  return {
    type: "lead",
    externalId: str(p.UNIQUE_QUERY_ID, p.QUERY_ID),
    title: str(p.QUERY_PRODUCT_NAME, p.SUBJECT) ?? "IndiaMART enquiry",
    contactName: str(p.SENDER_NAME),
    email: str(p.SENDER_EMAIL),
    phone: str(p.SENDER_MOBILE, p.SENDER_PHONE),
    company: str(p.SENDER_COMPANY),
    requirement: str(p.QUERY_MESSAGE),
    // B2B enquiries with a stated quantity are worth calling the same day.
    priority: p.QUERY_MCAT_NAME ? "HIGH" : "MEDIUM",
    channel: "ONLINE",
  };
}

function socialLead(p: Payload, label: string): NormalizedLead {
  // WhatsApp / Instagram / Facebook lead-ad payloads all put the useful bits in
  // a flat field list once the platform envelope is unwrapped.
  const fields: Payload = p.field_data
    ? Object.fromEntries(
        (p.field_data as Payload[]).map((f) => [f.name, Array.isArray(f.values) ? f.values[0] : f.values]),
      )
    : p;

  return {
    type: "lead",
    externalId: str(p.leadgen_id, p.lead_id, p.message_id, p.id),
    title: str(fields.subject, fields.product) ?? `${label} enquiry`,
    contactName: str(fields.full_name, fields.name, p.profile_name, p.from_name),
    email: str(fields.email),
    phone: str(fields.phone_number, fields.phone, p.from, p.wa_id),
    requirement: str(fields.message, p.text, p.body?.text),
    channel: "ONLINE",
  };
}

function shopifyOrder(p: Payload): NormalizedOrder {
  const customer = p.customer ?? {};
  const shipping = p.shipping_address ?? {};
  return {
    type: "order",
    externalId: String(p.id ?? p.order_number ?? p.name),
    externalUrl: str(p.order_status_url),
    contactName:
      str(`${customer.first_name ?? ""} ${customer.last_name ?? ""}`.trim(), shipping.name, p.email) ??
      "Online customer",
    email: str(p.email, customer.email),
    phone: str(p.phone, customer.phone, shipping.phone),
    shippingAddress: [shipping.address1, shipping.address2, shipping.city, shipping.province, shipping.zip]
      .filter(Boolean)
      .join(", "),
    currency: str(p.currency) ?? "INR",
    total: num(p.total_price, p.current_total_price) ?? 0,
    amountPaid: p.financial_status === "paid" ? num(p.total_price) ?? 0 : num(p.total_price_paid) ?? 0,
    channel: "ONLINE",
    items: (p.line_items ?? []).map((li: Payload) => ({
      name: str(li.title, li.name) ?? "Item",
      quantity: num(li.quantity) ?? 1,
      unitPrice: num(li.price) ?? 0,
      notes: str(li.variant_title),
    })),
  };
}

function marketplaceOrder(p: Payload, label: string): NormalizedOrder {
  // Amazon / Flipkart / Etsy each expose a different report shape; this reads
  // the common denominator and leaves the rest in the stored raw payload.
  const items: Payload[] = p.items ?? p.OrderItems ?? p.line_items ?? [];
  return {
    type: "order",
    externalId: String(p.order_id ?? p.AmazonOrderId ?? p.orderId ?? p.receipt_id ?? p.id),
    externalUrl: str(p.order_url),
    contactName: str(p.buyer_name, p.BuyerName, p.customer_name, p.name) ?? `${label} buyer`,
    email: str(p.buyer_email, p.BuyerEmail, p.email),
    phone: str(p.buyer_phone, p.phone),
    shippingAddress: str(p.shipping_address, p.ShippingAddress?.AddressLine1),
    currency: str(p.currency, p.CurrencyCode) ?? "INR",
    total: num(p.total, p.OrderTotal?.Amount, p.grand_total) ?? 0,
    amountPaid: num(p.amount_paid) ?? 0,
    channel: "ONLINE",
    items: items.map((li) => ({
      name: str(li.title, li.Title, li.name, li.product_name) ?? "Item",
      quantity: num(li.quantity, li.QuantityOrdered) ?? 1,
      unitPrice: num(li.price, li.unit_price, li.ItemPrice?.Amount) ?? 0,
    })),
  };
}

export function normalizePayload(kind: SourceKind, payload: Payload): Normalized {
  switch (kind) {
    case "SHOPIFY":
    case "WOOCOMMERCE":
      return shopifyOrder(payload);
    case "AMAZON":
      return marketplaceOrder(payload, "Amazon");
    case "FLIPKART":
      return marketplaceOrder(payload, "Flipkart");
    case "ETSY":
      return marketplaceOrder(payload, "Etsy");
    case "INDIAMART":
      return indiamartLead(payload);
    case "WHATSAPP":
      return socialLead(payload, "WhatsApp");
    case "INSTAGRAM":
      return socialLead(payload, "Instagram");
    case "FACEBOOK_LEADS":
      return socialLead(payload, "Facebook");
    default:
      return websiteLead(payload);
  }
}
