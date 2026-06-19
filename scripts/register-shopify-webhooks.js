/**
 * Register Shopify webhooks pointing to your server.
 *   node scripts/register-shopify-webhooks.js
 *
 * Requires in .env:
 *   SHOPIFY_STORE_URL    e.g. https://your-store.myshopify.com
 *   SHOPIFY_ACCESS_TOKEN (Admin API access token — Shopify Admin → Apps → private apps)
 *   SERVER_URL           e.g. https://yourdomain.com
 */

require('dotenv').config();
const https = require('https');

const { SHOPIFY_STORE_URL, SHOPIFY_ACCESS_TOKEN, SERVER_URL } = process.env;

if (!SHOPIFY_STORE_URL || !SHOPIFY_ACCESS_TOKEN || !SERVER_URL) {
  console.error('Missing env vars: SHOPIFY_STORE_URL, SHOPIFY_ACCESS_TOKEN, SERVER_URL');
  process.exit(1);
}

const WEBHOOKS = [
  {
    topic: 'orders/create',
    address: `${SERVER_URL}/webhooks/shopify`,
    format: 'json',
  },
];

async function request(method, path, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, SHOPIFY_STORE_URL);
    const data = body ? JSON.stringify(body) : null;

    const options = {
      hostname: url.hostname,
      path: url.pathname,
      method,
      headers: {
        'X-Shopify-Access-Token': SHOPIFY_ACCESS_TOKEN,
        'Content-Type': 'application/json',
        ...(data ? { 'Content-Length': Buffer.byteLength(data) } : {}),
      },
    };

    const req = https.request(options, (res) => {
      let raw = '';
      res.on('data', (chunk) => (raw += chunk));
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(raw) }); }
        catch { resolve({ status: res.statusCode, body: raw }); }
      });
    });

    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

async function main() {
  console.log(`\nRegistering webhooks with Shopify store: ${SHOPIFY_STORE_URL}\n`);

  // List existing webhooks
  const existing = await request('GET', '/admin/api/2024-01/webhooks.json');
  const existingTopics = (existing.body.webhooks ?? []).map((w) => w.topic);
  console.log('Existing webhooks:', existingTopics.length ? existingTopics : 'none');

  for (const hook of WEBHOOKS) {
    if (existingTopics.includes(hook.topic)) {
      console.log(`⚠️  Webhook already registered: ${hook.topic} — skipping`);
      continue;
    }

    const res = await request('POST', '/admin/api/2024-01/webhooks.json', { webhook: hook });

    if (res.status === 201) {
      console.log(`✅ Registered: ${hook.topic} → ${hook.address}`);
    } else {
      console.error(`❌ Failed (${res.status}):`, JSON.stringify(res.body));
    }
  }

  console.log('\nDone. Test with: curl -X POST https://yourdomain.com/webhooks/shopify -H "Content-Type: application/json" -d \'{"test":true}\'\n');
}

main().catch(console.error);
