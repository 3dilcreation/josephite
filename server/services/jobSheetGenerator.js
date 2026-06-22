const SERVICE_LABELS = {
  '360-tour':        '360° Virtual Tour',
  '3d-modeling':     '3D Modeling',
  '3d-printing':     '3D Printing',
  'rendering':       '3D Rendering',
  'virtual-staging': 'Virtual Staging',
  default:           'Custom Order',
};

const SOURCE_LABELS = {
  website:   'Website Order',
  whatsapp:  'WhatsApp Order',
  instagram: 'Instagram Order',
  manual:    'Direct / Phone Order',
};

const DEADLINE_DAYS = {
  '360-tour':        { standard: 3, complex: 5,  rush: 1 },
  '3d-modeling':     { standard: 5, complex: 10, rush: 2 },
  '3d-printing':     { standard: 7, complex: 14, rush: 3 },
  'rendering':       { standard: 4, complex: 7,  rush: 2 },
  'virtual-staging': { standard: 3, complex: 5,  rush: 1 },
  default:           { standard: 5, complex: 7,  rush: 2 },
};

function formatDate(date) {
  return date.toLocaleDateString('en-IN', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: process.env.CALENDAR_TIMEZONE ?? 'Asia/Kolkata',
  });
}

function now() {
  return new Date().toLocaleString('en-IN', {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
    timeZone: process.env.CALENDAR_TIMEZONE ?? 'Asia/Kolkata',
  });
}

function row(label, value) {
  if (!value || value === '—' || value === 'null' || value === 'undefined') return '';
  return `
    <tr>
      <td style="padding:8px 12px;color:#888;font-size:12px;text-transform:uppercase;letter-spacing:1px;white-space:nowrap;width:140px;">${label}</td>
      <td style="padding:8px 12px;color:#1a1a2e;font-size:14px;font-weight:500;">${value}</td>
    </tr>`;
}

function generate(order) {
  const serviceLabel = SERVICE_LABELS[order.serviceType] ?? SERVICE_LABELS.default;
  const sourceLabel  = SOURCE_LABELS[order.source]      ?? 'Order';
  const deadline     = formatDate(order.deadline);
  const created      = now();
  const days         = (DEADLINE_DAYS[order.serviceType] ?? DEADLINE_DAYS.default)[order.complexity ?? 'standard'] ?? 5;

  const calendarBtn = order.calendarLink
    ? `<a href="${order.calendarLink}" style="display:inline-block;margin-top:14px;background:#c9a227;color:#1a1a2e;padding:10px 22px;border-radius:4px;font-weight:700;font-size:13px;text-decoration:none;">View in Google Calendar</a>`
    : '';

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Job Sheet — ${order.jobNumber}</title>
</head>
<body style="margin:0;padding:0;background:#f0f0f0;font-family:'Segoe UI',Arial,sans-serif;">
<div style="max-width:680px;margin:30px auto 50px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,.12);">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 55%,#0f3460 100%);padding:36px 40px 28px;">
    <div style="font-size:26px;font-weight:800;letter-spacing:3px;color:#c9a227;">3DIL CREATION</div>
    <div style="font-size:11px;color:#888;letter-spacing:2px;margin-top:4px;">3D DESIGN &amp; VISUALISATION STUDIO</div>
    <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-top:24px;">
      <div>
        <div style="font-size:18px;font-weight:600;color:#fff;">Job Sheet</div>
        <div style="display:inline-block;margin-top:6px;background:rgba(201,162,39,.18);border:1px solid #c9a227;color:#c9a227;padding:3px 12px;border-radius:20px;font-size:11px;letter-spacing:1px;">${sourceLabel}</div>
      </div>
      <div style="background:#c9a227;color:#1a1a2e;padding:10px 18px;border-radius:4px;font-weight:800;font-size:15px;letter-spacing:1px;">${order.jobNumber}</div>
    </div>
  </div>

  <!-- Status bar -->
  <div style="background:#e8f5e9;border-top:3px solid #43a047;padding:10px 40px;display:flex;align-items:center;gap:8px;">
    <span style="display:inline-block;width:8px;height:8px;background:#43a047;border-radius:50%;"></span>
    <span style="font-size:13px;color:#2e7d32;"><strong>Order Received</strong> &nbsp;·&nbsp; ${created}</span>
  </div>

  <!-- Customer info -->
  <div style="padding:24px 40px 0;">
    <div style="font-size:11px;font-weight:700;color:#c9a227;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;">Customer Information</div>
    <table style="width:100%;border-collapse:collapse;background:#fafafa;border-radius:6px;overflow:hidden;">
      ${row('Full Name',    order.customerName)}
      ${row('Email',        order.customerEmail)}
      ${row('Phone',        order.customerPhone)}
      ${row('Order Source', sourceLabel)}
      ${order.orderId ? row('Order Ref', `#${order.orderId}`) : ''}
    </table>
  </div>

  <!-- Job details -->
  <div style="padding:24px 40px 0;">
    <div style="font-size:11px;font-weight:700;color:#c9a227;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;">Job Details</div>
    <table style="width:100%;border-collapse:collapse;background:#fafafa;border-radius:6px;overflow:hidden;">
      ${row('Service Type', serviceLabel)}
      ${row('Amount',       order.amount)}
      ${row('Priority',     order.complexity ? order.complexity.charAt(0).toUpperCase() + order.complexity.slice(1) : null)}
      ${order.items   ? row('Items',    order.items)   : ''}
      ${order.address ? row('Location', order.address) : ''}
    </table>
  </div>

  <!-- Deadline -->
  <div style="padding:24px 40px 0;">
    <div style="font-size:11px;font-weight:700;color:#c9a227;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;">Project Deadline</div>
    <div style="background:linear-gradient(135deg,#1a1a2e,#0f3460);border-radius:8px;padding:22px;text-align:center;">
      <div style="font-size:11px;color:#c9a227;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">Scheduled Completion</div>
      <div style="font-size:22px;font-weight:700;color:#fff;">${deadline}</div>
      <div style="font-size:12px;color:#888;margin-top:6px;">${days} working day${days !== 1 ? 's' : ''} from order date</div>
      ${calendarBtn}
    </div>
  </div>

  <!-- Notes -->
  ${order.notes ? `
  <div style="padding:24px 40px 0;">
    <div style="font-size:11px;font-weight:700;color:#c9a227;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;">Requirements &amp; Notes</div>
    <div style="background:#fafafa;border-left:4px solid #c9a227;padding:14px 16px;border-radius:0 4px 4px 0;font-size:14px;line-height:1.7;color:#444;white-space:pre-wrap;">${order.notes.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>
  </div>` : ''}

  <!-- Terms -->
  <div style="padding:24px 40px;">
    <div style="font-size:11px;font-weight:700;color:#c9a227;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;">Next Steps</div>
    <ul style="font-size:13px;line-height:2.2;color:#555;padding-left:20px;margin:0;">
      <li>Our team will review your requirements and contact you within <strong>24 hours</strong></li>
      <li>Final quote confirmed before any work begins</li>
      <li>50% advance payment required to start the project</li>
      <li>Revisions included as per the agreed package</li>
      <li>All files delivered in agreed formats by the deadline above</li>
    </ul>
  </div>

  <!-- Drive link injected here by emailService after Drive upload -->
  <!-- DRIVE_LINK_PLACEHOLDER -->

  <!-- Footer -->
  <div style="background:#1a1a2e;padding:20px 40px;text-align:center;">
    <div style="font-size:14px;color:#fff;">Thank you for choosing <strong style="color:#c9a227;">3Dil Creation</strong></div>
    <div style="margin-top:8px;font-size:12px;color:#888;">
      <a href="mailto:3dilcreation@gmail.com" style="color:#c9a227;text-decoration:none;">3dilcreation@gmail.com</a>
    </div>
    <div style="margin-top:8px;font-size:11px;color:#555;">Auto-generated job sheet · ${order.jobNumber}</div>
  </div>

</div>
</body>
</html>`;
}

module.exports = { generate };
