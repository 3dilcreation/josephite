/**
 * WhatsApp Auto-Reply Service
 * Sends an instant acknowledgement message back to the customer via WhatsApp
 * when their message is received and processed as an order.
 */

const axios = require('axios');

const BASE_URL = 'https://graph.facebook.com/v19.0';

const SERVICE_LABELS = {
  '360-tour':        '360° Virtual Tour',
  '3d-modeling':     '3D Modeling',
  '3d-printing':     '3D Printing',
  'rendering':       '3D Rendering',
  'virtual-staging': 'Virtual Staging',
  default:           'Custom Order',
};

async function sendReply(toPhone, order) {
  const token = process.env.WHATSAPP_ACCESS_TOKEN;
  const phoneNumberId = process.env.WHATSAPP_PHONE_NUMBER_ID;

  if (!token || !phoneNumberId) {
    console.warn('WhatsApp reply skipped — WHATSAPP_ACCESS_TOKEN or WHATSAPP_PHONE_NUMBER_ID not set');
    return;
  }

  const serviceLabel = SERVICE_LABELS[order.serviceType] ?? SERVICE_LABELS.default;
  const deadline = order.deadline.toLocaleDateString('en-IN', {
    weekday: 'long', day: 'numeric', month: 'long', year: 'numeric',
    timeZone: process.env.CALENDAR_TIMEZONE ?? 'Asia/Kolkata',
  });

  const message = `✅ *Order Received — ${order.jobNumber}*\n\nHi ${order.customerName.split(' ')[0]}! 👋\n\nThank you for reaching out to *3Dil Creation*.\n\nHere's a summary of your request:\n\n📋 *Service:* ${serviceLabel}\n⏰ *Estimated Deadline:* ${deadline}\n💰 *Quote:* ${order.amount !== 'TBD' ? order.amount : 'Will be confirmed shortly'}\n\nA detailed job sheet has been sent to your email${order.customerEmail ? ` (${order.customerEmail})` : ''}.\n\nOur team will review your requirements and get back to you within *24 hours*.\n\n_3Dil Creation — 3D Design & Visualisation Studio_`;

  try {
    await axios.post(
      `${BASE_URL}/${phoneNumberId}/messages`,
      {
        messaging_product: 'whatsapp',
        to: toPhone,
        type: 'text',
        text: { body: message, preview_url: false },
      },
      {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      }
    );
    console.log(`WhatsApp reply sent to ${toPhone}`);
  } catch (err) {
    console.error('WhatsApp reply failed:', err.response?.data ?? err.message);
  }
}

// Send a "we need more info" message for messages that don't contain order intent
async function sendInfoRequest(toPhone, customerName) {
  const token = process.env.WHATSAPP_ACCESS_TOKEN;
  const phoneNumberId = process.env.WHATSAPP_PHONE_NUMBER_ID;
  if (!token || !phoneNumberId) return;

  const name = customerName?.split(' ')[0] ?? 'there';
  const message = `Hi ${name}! 👋 Welcome to *3Dil Creation*.\n\nTo process your order, please share:\n1️⃣ *Service needed* (360° Tour / 3D Model / Print / Rendering)\n2️⃣ *Your name & email*\n3️⃣ *Project description*\n4️⃣ *Any deadline in mind?*\n\nWe'll get back to you within 24 hours! 🙌`;

  try {
    await axios.post(
      `${BASE_URL}/${phoneNumberId}/messages`,
      {
        messaging_product: 'whatsapp',
        to: toPhone,
        type: 'text',
        text: { body: message, preview_url: false },
      },
      { headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } }
    );
  } catch (err) {
    console.error('WhatsApp info-request failed:', err.response?.data ?? err.message);
  }
}

module.exports = { sendReply, sendInfoRequest };
