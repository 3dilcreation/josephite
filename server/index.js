require('dotenv').config();
const express = require('express');
const path = require('path');
const webhooksRouter = require('./routes/webhooks');

const app = express();

// Raw body needed for Shopify HMAC verification — must come before json parser
app.use('/webhooks/shopify', express.raw({ type: 'application/json' }));

// JSON parser for all other routes
app.use(express.json());

// Serve existing static frontend
app.use(express.static(path.join(__dirname, '..')));

// Order automation webhooks
app.use('/webhooks', webhooksRouter);

// Admin panel
app.use('/admin', express.static(path.join(__dirname, '..', 'admin')));

// Health check
app.get('/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`3Dil Creation order server running on port ${PORT}`);
  console.log(`Webhooks ready at /webhooks/{shopify,whatsapp,instagram}`);
  console.log(`Admin panel at /admin`);
});
