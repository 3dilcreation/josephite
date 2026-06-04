const { google } = require('googleapis');

const SERVICE_LABELS = {
  '360-tour':        '360° Virtual Tour',
  '3d-modeling':     '3D Modeling',
  '3d-printing':     '3D Printing',
  'rendering':       '3D Rendering',
  'virtual-staging': 'Virtual Staging',
  default:           'Custom Order',
};

// Colour coding by order source (Google Calendar colour IDs)
const SOURCE_COLORS = {
  website:   '2',  // Sage green
  whatsapp:  '10', // Basil dark green
  instagram: '6',  // Flamingo pink
  manual:    '5',  // Banana yellow
};

function getAuth() {
  const client = new google.auth.OAuth2(
    process.env.GOOGLE_CLIENT_ID,
    process.env.GOOGLE_CLIENT_SECRET,
    process.env.GOOGLE_REDIRECT_URI
  );
  client.setCredentials({ refresh_token: process.env.GOOGLE_REFRESH_TOKEN });
  return client;
}

async function createJobEvent(order) {
  const auth = getAuth();
  const calendar = google.calendar({ version: 'v3', auth });

  const serviceLabel = SERVICE_LABELS[order.serviceType] ?? SERVICE_LABELS.default;
  const tz = process.env.CALENDAR_TIMEZONE ?? 'Asia/Kolkata';

  // Deadline day: 9 AM – 5 PM
  const start = new Date(order.deadline);
  start.setHours(9, 0, 0, 0);
  const end = new Date(order.deadline);
  end.setHours(17, 0, 0, 0);

  const description = [
    `JOB: ${order.jobNumber}`,
    `SOURCE: ${order.source.toUpperCase()}`,
    `CUSTOMER: ${order.customerName}`,
    order.customerEmail  ? `EMAIL: ${order.customerEmail}`  : null,
    order.customerPhone  ? `PHONE: ${order.customerPhone}`  : null,
    `SERVICE: ${serviceLabel}`,
    order.amount         ? `AMOUNT: ${order.amount}`         : null,
    order.orderId        ? `ORDER ID: #${order.orderId}`     : null,
    order.address        ? `LOCATION: ${order.address}`      : null,
    '',
    order.notes ? `NOTES:\n${order.notes}` : null,
  ].filter(Boolean).join('\n');

  const event = {
    summary: `[${order.jobNumber}] ${order.customerName} — ${serviceLabel}`,
    description,
    location: order.address || undefined,
    colorId: SOURCE_COLORS[order.source] ?? '1',
    start: { dateTime: start.toISOString(), timeZone: tz },
    end:   { dateTime: end.toISOString(),   timeZone: tz },
    reminders: {
      useDefault: false,
      overrides: [
        { method: 'email',  minutes: 24 * 60 }, // 1 day before
        { method: 'popup',  minutes: 60 },       // 1 hour before
      ],
    },
  };

  const response = await calendar.events.insert({
    calendarId: process.env.GOOGLE_CALENDAR_ID ?? 'primary',
    resource: event,
    sendUpdates: 'none',
  });

  console.log(`Calendar event created: ${response.data.htmlLink}`);
  return response.data;
}

module.exports = { createJobEvent };
