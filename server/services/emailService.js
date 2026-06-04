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

// Send job sheet to the customer
async function sendJobSheet(order, html) {
  if (!order.customerEmail) return;
  const transporter = createTransporter();

  await transporter.sendMail({
    from: `"3Dil Creation" <${process.env.EMAIL_USER}>`,
    to: order.customerEmail,
    subject: `Order Confirmed — ${order.jobNumber} | 3Dil Creation`,
    html,
    text: buildPlainText(order),
  });

  console.log(`Job sheet emailed to ${order.customerEmail}`);
}

// Notify the business owner (always runs)
async function sendOwnerNotification(order, html) {
  const transporter = createTransporter();
  const ownerEmail = process.env.OWNER_EMAIL ?? process.env.EMAIL_USER;

  const sourceIcon = { website: '🛍️', whatsapp: '💬', instagram: '📸', manual: '📝' }[order.source] ?? '📋';

  await transporter.sendMail({
    from: `"Order Bot" <${process.env.EMAIL_USER}>`,
    to: ownerEmail,
    subject: `${sourceIcon} New ${order.source} order: ${order.jobNumber} — ${order.customerName}`,
    html,
    text: buildPlainText(order),
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

Calendar link: ${order.calendarLink ?? 'See Google Calendar'}
`.trim();
}

module.exports = { sendJobSheet, sendOwnerNotification };
