/**
 * Google Drive Service
 * Saves each job sheet as a Google Doc, exports it as PDF, and returns
 * both the Drive view link and a PDF buffer for email attachment.
 *
 * Reuses the same OAuth2 credentials as calendarService.js.
 */

const { google } = require('googleapis');

function getAuth() {
  const client = new google.auth.OAuth2(
    process.env.GOOGLE_CLIENT_ID,
    process.env.GOOGLE_CLIENT_SECRET,
    process.env.GOOGLE_REDIRECT_URI
  );
  client.setCredentials({ refresh_token: process.env.GOOGLE_REFRESH_TOKEN });
  return client;
}

async function saveJobSheet(order, html) {
  const auth = getAuth();
  const drive = google.drive({ version: 'v3', auth });

  const folderId = process.env.GOOGLE_DRIVE_FOLDER_ID || 'root';

  // Upload HTML → Drive auto-converts to Google Doc
  const created = await drive.files.create({
    requestBody: {
      name: `[${order.jobNumber}] ${order.customerName} — Job Sheet`,
      mimeType: 'application/vnd.google-apps.document',
      parents: [folderId],
    },
    media: {
      mimeType: 'text/html',
      body: html,
    },
    fields: 'id,webViewLink',
  });

  const fileId = created.data.id;
  const viewLink = created.data.webViewLink;

  // Export the Google Doc as PDF binary
  const exported = await drive.files.export(
    { fileId, mimeType: 'application/pdf' },
    { responseType: 'arraybuffer' }
  );

  const pdfBuffer = Buffer.from(exported.data);

  console.log(`Job sheet saved to Drive: ${viewLink}`);
  return { fileId, viewLink, pdfBuffer };
}

module.exports = { saveJobSheet };
