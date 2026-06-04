const express = require('express');
const crypto = require('crypto');
const router = express.Router();
const orderProcessor = require('../services/orderProcessor');

// ─── Shopify ────────────────────────────────────────────────────────────────

router.post('/shopify', async (req, res) => {
  // Verify Shopify HMAC signature (req.body is raw Buffer here)
  const secret = process.env.SHOPIFY_WEBHOOK_SECRET;
  const hmacHeader = req.headers['x-shopify-hmac-sha256'];

  if (secret && hmacHeader) {
    const digest = crypto
      .createHmac('sha256', secret)
      .update(req.body)
      .digest('base64');

    if (digest !== hmacHeader) {
      console.warn('Shopify webhook: invalid HMAC');
      return res.status(401).json({ error: 'Invalid signature' });
    }
  }

  const topic = req.headers['x-shopify-topic'];
  let body;
  try {
    body = JSON.parse(req.body.toString());
  } catch {
    return res.status(400).json({ error: 'Invalid JSON' });
  }

  if (topic === 'orders/create') {
    try {
      await orderProcessor.processShopifyOrder(body);
      return res.status(200).json({ success: true });
    } catch (err) {
      console.error('Shopify order processing failed:', err.message);
      return res.status(500).json({ error: 'Processing failed' });
    }
  }

  res.status(200).json({ received: true, topic });
});

// ─── WhatsApp (Meta Business API) ──────────────────────────────────────────

// Webhook verification challenge
router.get('/whatsapp', (req, res) => {
  const mode = req.query['hub.mode'];
  const token = req.query['hub.verify_token'];
  const challenge = req.query['hub.challenge'];

  if (mode === 'subscribe' && token === process.env.WHATSAPP_VERIFY_TOKEN) {
    return res.status(200).send(challenge);
  }
  res.status(403).json({ error: 'Verification failed' });
});

router.post('/whatsapp', async (req, res) => {
  const body = req.body;

  if (body.object !== 'whatsapp_business_account') {
    return res.status(200).json({ received: true });
  }

  try {
    const entry = body.entry?.[0];
    const change = entry?.changes?.[0];
    const value = change?.value;
    const messages = value?.messages;
    const contacts = value?.contacts;

    if (messages?.length) {
      const message = messages[0];
      const contact = contacts?.[0];

      // Only process text and interactive (button/list) messages
      if (['text', 'interactive', 'image', 'document'].includes(message.type)) {
        await orderProcessor.processWhatsAppMessage(message, contact, value);
      }
    }
  } catch (err) {
    console.error('WhatsApp processing failed:', err.message);
  }

  // Always return 200 to Meta or they'll retry
  res.status(200).json({ success: true });
});

// ─── Instagram (Meta Graph API) ────────────────────────────────────────────

router.get('/instagram', (req, res) => {
  const mode = req.query['hub.mode'];
  const token = req.query['hub.verify_token'];
  const challenge = req.query['hub.challenge'];

  if (mode === 'subscribe' && token === process.env.INSTAGRAM_VERIFY_TOKEN) {
    return res.status(200).send(challenge);
  }
  res.status(403).json({ error: 'Verification failed' });
});

router.post('/instagram', async (req, res) => {
  const body = req.body;

  if (body.object !== 'instagram') {
    return res.status(200).json({ received: true });
  }

  try {
    const entry = body.entry?.[0];
    const messaging = entry?.messaging?.[0];

    if (messaging?.message) {
      await orderProcessor.processInstagramMessage(messaging);
    }
  } catch (err) {
    console.error('Instagram processing failed:', err.message);
  }

  res.status(200).json({ success: true });
});

// ─── Manual order entry (phone / walk-in / WhatsApp noted manually) ─────────

router.post('/manual', async (req, res) => {
  try {
    await orderProcessor.processManualOrder(req.body);
    res.status(200).json({ success: true });
  } catch (err) {
    console.error('Manual order failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
