// Extract structured order info from free-text messages (WhatsApp / Instagram DMs)

const SERVICE_PATTERNS = [
  { pattern: /360\s*(tour|photo|view|panorama|virtual)/i, type: '360-tour' },
  { pattern: /(360|panorama)/i, type: '360-tour' },
  { pattern: /(3d\s*print|printing|printed)/i, type: '3d-printing' },
  { pattern: /(render|rendering|visualis|visualiz)/i, type: 'rendering' },
  { pattern: /(virtual\s*stag|staging)/i, type: 'virtual-staging' },
  { pattern: /(3d\s*model|modell?ing)/i, type: '3d-modeling' },
  { pattern: /(viewing|book\s*a\s*view)/i, type: '360-tour' },
];

const EMAIL_RE = /[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/;
const PHONE_RE = /(\+?\d[\d\s\-().]{8,14}\d)/;
// Only match amounts with an explicit currency symbol or keyword like "price/cost/quote"
const PRICE_RE = /(?:(?:price|cost|quote|amount|charge)[:\s]+)?(?:R|INR|₹|ZAR|Rs\.?)\s*[\d,]+(?:\.\d{1,2})?/i;

const RUSH_KEYWORDS = /\b(urgent|rush|asap|immediately|today|tonight|tomorrow)\b/i;
const COMPLEX_KEYWORDS = /\b(complex|large|big|multiple|detailed|full\s*house|entire)\b/i;

function parseServiceType(text) {
  for (const { pattern, type } of SERVICE_PATTERNS) {
    if (pattern.test(text)) return type;
  }
  return 'default';
}

function parseComplexity(text) {
  if (RUSH_KEYWORDS.test(text)) return 'rush';
  if (COMPLEX_KEYWORDS.test(text)) return 'complex';
  return 'standard';
}

function parseEmail(text) {
  const match = text.match(EMAIL_RE);
  return match ? match[0] : null;
}

function parsePhone(text) {
  const match = text.match(PHONE_RE);
  return match ? match[1].trim() : null;
}

function parseAmount(text) {
  const match = text.match(PRICE_RE);
  // Reject if the match is just a lone number without a currency indicator
  if (!match) return null;
  const m = match[0].trim();
  return /[R₹]|INR|ZAR|Rs/i.test(m) ? m : null;
}

// Try to extract a name — looks for "my name is X" / "I am X" / "this is X"
function parseName(text) {
  const patterns = [
    /(?:my name is|i am|i'm|this is)\s+([A-Za-z][\w\s]{1,30}?)(?:\s*[,.\n]|$)/i,
    /(?:name[:\-]\s*)([A-Za-z][\w\s]{1,30}?)(?:\s*[,.\n]|$)/i,
  ];
  for (const p of patterns) {
    const m = text.match(p);
    if (m) return m[1].trim();
  }
  return null;
}

// Extract address / location hints
function parseAddress(text) {
  const patterns = [
    /(?:at|located at|address[:\-]?)\s+([^.\n]{5,80})/i,
    /(?:property|location|place)[:\-]?\s+([^.\n]{5,80})/i,
  ];
  for (const p of patterns) {
    const m = text.match(p);
    if (m) return m[1].trim();
  }
  return null;
}

function parse(text) {
  return {
    serviceType: parseServiceType(text),
    complexity: parseComplexity(text),
    email: parseEmail(text),
    phone: parsePhone(text),
    amount: parseAmount(text),
    name: parseName(text),
    address: parseAddress(text),
  };
}

module.exports = { parse, parseServiceType, parseComplexity };
