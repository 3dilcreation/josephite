const nodemailer = require('nodemailer');

function createTransporter() {
  return nodemailer.createTransport({
    service: 'gmail',
    auth: {
      type: 'OAuth2',
      user: process.env.EMAIL_USER,
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      refreshToken: process.env.GOOGLE_REFRESH_TOKEN,
    },
  });
}

function formatDeadline(date) {
  return date.toLocaleDateString('en-IN', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: process.env.CALENDAR_TIMEZONE ?? 'Asia/Kolkata',
  });
}

function buildAttachments(order) {
  const attachments = [];

  // Attach PDF if available from Drive export
  if (order.pdfBuffer) {
    attachments.push({
      filename: `JobSheet-${order.jobNumber}.pdf`,
      content: order.pdfBuffer,
      contentType: 'application/pdf',
    });
  }

  return attachments;
}

function buildDriveSection(order) {
  if (!order.driveLink) return '';
  return `
    <div style="margin:0 0 16px;padding:14px 16px;background:#e8f0fe;border-left:4px solid #4285f4;border-radius:0 6px 6px 0;">
      <span style="font-size:13px;color:#1a73e8;">
        📄 <a href="${order.driveLink}" style="color:#1a73e8;font-weight:700;text-decoration:none;">View &amp; Download Job Sheet on Google Drive</a>
      </span>
    </div>`;
}

// Send job sheet + PDF to the customer
async function sendJobSheet(order, html) {
  if (!order.customerEmail) return;
  const transporter = createTransporter();

  // Inject Drive link section before the closing footer
  const enrichedHtml = html.replace(
    '<!-- DRIVE_LINK_PLACEHOLDER -->',
    buildDriveSection(order)
  );

  await transporter.sendMail({
    from: `"3Dil Creation" <${process.env.EMAIL_USER}>`,
    to: order.customerEmail,
    subject: `Order Confirmed — ${order.jobNumber} | 3Dil Creation`,
    html: enrichedHtml,
    text: buildPlainText(order),
    attachments: buildAttachments(order),
  });

  console.log(`Job sheet + PDF emailed to ${order.customerEmail}`);
}

// Notify the business owner with full details + PDF
async function sendOwnerNotification(order, html) {
  const transporter = createTransporter();
  const ownerEmail = process.env.OWNER_EMAIL ?? process.env.EMAIL_USER;
  const sourceIcon = { website: '🛍️', whatsapp: '💬', instagram: '📸', manual: '📝' }[order.source] ?? '📋';

  const enrichedHtml = html.replace(
    '<!-- DRIVE_LINK_PLACEHOLDER -->',
    buildDriveSection(order)
  );

  await transporter.sendMail({
    from: `"Order Bot" <${process.env.EMAIL_USER}>`,
    to: ownerEmail,
    subject: `${sourceIcon} New ${order.source} order: ${order.jobNumber} — ${order.customerName}`,
    html: enrichedHtml,
    text: buildPlainText(order),
    attachments: buildAttachments(order),
  });

  console.log(`Owner notified: ${order.jobNumber}`);
}

function buildPlainText(order) {
  return `
3DIL CREATION — JOB SHEET
==========================
Job Number : ${order.jobNumber}
Source     : ${order.source.toUpperCase()}
Customer   : ${order.customerName}
Email      : ${order.customerEmail ?? '—'}
Phone      : ${order.customerPhone ?? '—'}
Service    : ${order.serviceType}
Amount     : ${order.amount}
Deadline   : ${formatDeadline(order.deadline)}
Address    : ${order.address || '—'}

Notes:
${order.notes || 'None'}

Calendar  : ${order.calendarLink ?? 'See Google Calendar'}
Drive Doc : ${order.driveLink ?? 'N/A'}
`.trim();
}

module.exports = { sendJobSheet, sendOwnerNotification };
