const calendarService = require('./calendarService');
const emailService = require('./emailService');
const jobSheetGenerator = require('./jobSheetGenerator');
const whatsappService = require('./whatsappService');
const parser = require('../utils/messageParser');

// ─── Helpers ────────────────────────────────────────────────────────────────

function generateJobNumber() {
  const now = new Date();
  const yy = now.getFullYear().toString().slice(-2);
  const mm = String(now.getMonth() + 1).padStart(2, '0');
  const rand = Math.random().toString(36).substr(2, 4).toUpperCase();
  return `JOB-${yy}${mm}-${rand}`;
}

// Working-day deadline: skip weekends
function calcDeadline(serviceType, complexity = 'standard') {
  const dayMap = {
    '360-tour':        { standard: 3, complex: 5,  rush: 1 },
    '3d-modeling':     { standard: 5, complex: 10, rush: 2 },
    '3d-printing':     { standard: 7, complex: 14, rush: 3 },
    'rendering':       { standard: 4, complex: 7,  rush: 2 },
    'virtual-staging': { standard: 3, complex: 5,  rush: 1 },
    default:           { standard: 5, complex: 7,  rush: 2 },
  };
  const days = (dayMap[serviceType] ?? dayMap.default)[complexity] ?? 5;
  const date = new Date();
  let added = 0;
  while (added < days) {
    date.setDate(date.getDate() + 1);
    const d = date.getDay();
    if (d !== 0 && d !== 6) added++; // skip Sun/Sat
  }
  return date;
}

// Core dispatcher — runs for every source
async function processOrder(order) {
  order.deadline = calcDeadline(order.serviceType, order.complexity || 'standard');
  console.log(`[${order.jobNumber}] Processing ${order.source} order for "${order.customerName}" — deadline ${order.deadline.toDateString()}`);

  const [calEvent] = await Promise.all([
    calendarService.createJobEvent(order),
  ]);
  order.calendarLink = calEvent?.htmlLink;

  const jobSheetHtml = jobSheetGenerator.generate(order);

  // Send to customer if we have their email
  if (order.customerEmail) {
    await emailService.sendJobSheet(order, jobSheetHtml);
  }

  // Always notify the business owner
  await emailService.sendOwnerNotification(order, jobSheetHtml);

  return order;
}

// ─── Source handlers ────────────────────────────────────────────────────────

async function processShopifyOrder(shopifyOrder) {
  const items = shopifyOrder.line_items ?? [];
  const firstItem = items[0] ?? {};
  const customProp = firstItem.properties?.find(p => p.name === 'service_type');

  const order = {
    jobNumber: generateJobNumber(),
    source: 'website',
    customerName: [
      shopifyOrder.customer?.first_name,
      shopifyOrder.customer?.last_name,
    ].filter(Boolean).join(' ') || shopifyOrder.billing_address?.name || 'Customer',
    customerEmail: shopifyOrder.customer?.email ?? shopifyOrder.contact_email,
    customerPhone:
      shopifyOrder.customer?.phone ??
      shopifyOrder.billing_address?.phone ??
      shopifyOrder.shipping_address?.phone,
    serviceType: customProp?.value ?? parser.parseServiceType(firstItem.title ?? ''),
    complexity: 'standard',
    items: items.map(i => i.title).join(', '),
    notes: shopifyOrder.note ?? '',
    address: shopifyOrder.shipping_address
      ? [shopifyOrder.shipping_address.address1, shopifyOrder.shipping_address.city]
          .filter(Boolean)
          .join(', ')
      : '',
    amount: `${shopifyOrder.currency ?? ''} ${shopifyOrder.total_price ?? ''}`.trim(),
    orderId: String(shopifyOrder.order_number ?? shopifyOrder.id ?? ''),
  };

  return processOrder(order);
}

async function processWhatsAppMessage(message, contact, _metadata) {
  const text = message.text?.body ?? message.interactive?.body?.text ?? '';
  const parsed = parser.parse(text);
  const fromPhone = message.from;
  const customerName = parsed.name ?? contact?.profile?.name ?? `+${fromPhone}`;

  // If the message is too short / vague, ask for more info instead of creating a bare order
  const isOrderIntent = parsed.serviceType !== 'default' || parsed.email || text.length > 30;
  if (!isOrderIntent) {
    await whatsappService.sendInfoRequest(fromPhone, customerName);
    return null;
  }

  const order = {
    jobNumber: generateJobNumber(),
    source: 'whatsapp',
    customerName,
    customerEmail: parsed.email,
    customerPhone: parsed.phone ?? fromPhone,
    serviceType: parsed.serviceType,
    complexity: parsed.complexity,
    notes: text,
    address: parsed.address ?? '',
    amount: parsed.amount ?? 'TBD',
  };

  const result = await processOrder(order);

  // Send WhatsApp acknowledgement reply
  await whatsappService.sendReply(fromPhone, result);

  return result;
}

async function processInstagramMessage(messaging) {
  const text = messaging.message?.text ?? '';
  const parsed = parser.parse(text);

  const order = {
    jobNumber: generateJobNumber(),
    source: 'instagram',
    customerName: parsed.name ?? `Instagram ${messaging.sender?.id}`,
    customerEmail: parsed.email,
    customerPhone: parsed.phone,
    serviceType: parsed.serviceType,
    complexity: parsed.complexity,
    notes: text,
    address: parsed.address ?? '',
    amount: parsed.amount ?? 'TBD',
    instagramSenderId: messaging.sender?.id,
  };

  return processOrder(order);
}

async function processManualOrder(data) {
  const order = {
    jobNumber: generateJobNumber(),
    source: 'manual',
    customerName: data.customerName ?? 'Unknown',
    customerEmail: data.customerEmail ?? null,
    customerPhone: data.customerPhone ?? null,
    serviceType: data.serviceType ?? parser.parseServiceType(data.notes ?? ''),
    complexity: data.complexity ?? 'standard',
    notes: data.notes ?? '',
    address: data.address ?? '',
    amount: data.amount ?? 'TBD',
  };

  return processOrder(order);
}

module.exports = {
  processShopifyOrder,
  processWhatsAppMessage,
  processInstagramMessage,
  processManualOrder,
};
