/**
 * Instagram DM Auto-Reply Service
 * Sends an instant acknowledgement via Instagram Graph API when an order DM arrives.
 * Requires: Instagram Business account linked to a Meta App with messaging permissions.
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

async function sendReply(recipientId, order) {
  const token = process.env.INSTAGRAM_ACCESS_TOKEN;
  const igAccountId = process.env.INSTAGRAM_ACCOUNT_ID;

  if (!token || !igAccountId) {
    console.warn('Instagram reply skipped — INSTAGRAM_ACCESS_TOKEN or INSTAGRAM_ACCOUNT_ID not set');
    return;
  }

  const serviceLabel = SERVICE_LABELS[order.serviceType] ?? SERVICE_LABELS.default;
  const deadline = order.deadline.toLocaleDateString('en-IN', {
    weekday: 'long', day: 'numeric', month: 'long', year: 'numeric',
    timeZone: process.env.CALENDAR_TIMEZONE ?? 'Asia/Kolkata',
  });

  const message = `✅ Order received — ${order.jobNumber}\n\nHi! Thanks for reaching out to 3Dil Creation 🙌\n\nService: ${serviceLabel}\nDeadline: ${deadline}\nQuote: ${order.amount !== 'TBD' ? order.amount : 'Will be confirmed shortly'}\n\nWe'll review your request and DM you within 24 hours with a full quote and job sheet.\n\n— 3Dil Creation Studio`;

  try {
    await axios.post(
      `${BASE_URL}/${igAccountId}/messages`,
      {
        recipient: { id: recipientId },
        message: { text: message },
      },
      {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      }
    );
    console.log(`Instagram reply sent to ${recipientId}`);
  } catch (err) {
    console.error('Instagram reply failed:', err.response?.data ?? err.message);
  }
}

// Sent when DM is too vague to parse as an order
async function sendInfoRequest(recipientId) {
  const token = process.env.INSTAGRAM_ACCESS_TOKEN;
  const igAccountId = process.env.INSTAGRAM_ACCOUNT_ID;
  if (!token || !igAccountId) return;

  const message = `Hi! 👋 Welcome to 3Dil Creation!\n\nTo process your order, please share:\n1. Service needed (360° Tour / 3D Model / Print / Rendering)\n2. Your name & email\n3. Project description\n4. Any deadline?\n\nWe'll get back to you within 24 hours! ✨`;

  try {
    await axios.post(
      `${BASE_URL}/${igAccountId}/messages`,
      {
        recipient: { id: recipientId },
        message: { text: message },
      },
      { headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } }
    );
  } catch (err) {
    console.error('Instagram info-request failed:', err.response?.data ?? err.message);
  }
}

module.exports = { sendReply, sendInfoRequest };
