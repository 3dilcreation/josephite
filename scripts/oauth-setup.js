/**
 * Run once to get your Google refresh token.
 *   npm run oauth
 *
 * Steps:
 *  1. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env
 *  2. Run this script — it prints an auth URL
 *  3. Open the URL, sign in as 3dilcreation@gmail.com, copy the code
 *  4. Paste the code when prompted — the script prints your GOOGLE_REFRESH_TOKEN
 *  5. Add GOOGLE_REFRESH_TOKEN to your .env
 */

require('dotenv').config();
const { google } = require('googleapis');
const readline = require('readline');

const SCOPES = [
  'https://www.googleapis.com/auth/calendar',
  'https://www.googleapis.com/auth/gmail.send',
];

const oauth2Client = new google.auth.OAuth2(
  process.env.GOOGLE_CLIENT_ID,
  process.env.GOOGLE_CLIENT_SECRET,
  'urn:ietf:wg:oauth:2.0:oob' // desktop / CLI flow
);

const url = oauth2Client.generateAuthUrl({
  access_type: 'offline',
  prompt: 'consent',
  scope: SCOPES,
});

console.log('\n=== Google OAuth2 Setup ===');
console.log('\n1. Open this URL in your browser (sign in as 3dilcreation@gmail.com):\n');
console.log(url);
console.log('\n2. After authorising, paste the code below:\n');

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

rl.question('Code: ', async (code) => {
  rl.close();
  try {
    const { tokens } = await oauth2Client.getToken(code.trim());
    console.log('\n✅ Success! Add this to your .env:\n');
    console.log(`GOOGLE_REFRESH_TOKEN=${tokens.refresh_token}`);
    console.log('\nDone. You can now start the server with: npm start\n');
  } catch (err) {
    console.error('Failed to exchange code:', err.message);
    process.exit(1);
  }
});
