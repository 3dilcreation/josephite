// ============================================================
// TechSei LMS — Stripe Payments & Subscriptions
// ============================================================

import { initStripe } from '@stripe/stripe-react-native';
import type { Subscription } from '@/types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Stripe publishable key — replace with your actual key before shipping. */
export const STRIPE_PUBLISHABLE_KEY =
  process.env.EXPO_PUBLIC_STRIPE_PUBLISHABLE_KEY ??
  'pk_test_your_stripe_publishable_key_here';

/**
 * Backend API base URL.
 * Your server handles Stripe secret-key operations (create PaymentIntent, etc.)
 * so the mobile app never touches the Stripe secret key directly.
 */
const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_BASE_URL ?? 'https://api.techsei.app';

// ---------------------------------------------------------------------------
// Plan definitions
// ---------------------------------------------------------------------------

export interface Plan {
  id: string;
  name: string;
  /** Price in the smallest currency unit (e.g. cents for USD). */
  priceInCents: number;
  /** Human-readable price string. */
  displayPrice: string;
  currency: string;
  interval: 'month' | 'year';
  /** Percentage saved compared to monthly * 12 (only set on yearly plan). */
  savingsPercent?: number;
  features: string[];
  /** Stripe Price ID — set in your Stripe dashboard. */
  stripePriceId: string;
}

export const PLANS: Record<'monthly' | 'yearly', Plan> = {
  monthly: {
    id: 'pro_monthly',
    name: 'TechSei Pro — Monthly',
    priceInCents: 999, // $9.99 / month
    displayPrice: '$9.99',
    currency: 'usd',
    interval: 'month',
    stripePriceId: process.env.EXPO_PUBLIC_STRIPE_PRICE_MONTHLY ?? 'price_monthly_placeholder',
    features: [
      '50+ Premium Courses',
      'Advanced AI Tutor (unlimited)',
      'Offline Downloads',
      'Certificates of Completion',
      'Priority Support',
    ],
  },
  yearly: {
    id: 'pro_yearly',
    name: 'TechSei Pro — Yearly',
    priceInCents: 7999, // $79.99 / year (≈ $6.67/mo vs $9.99/mo)
    displayPrice: '$79.99',
    currency: 'usd',
    interval: 'year',
    savingsPercent: 33,
    stripePriceId: process.env.EXPO_PUBLIC_STRIPE_PRICE_YEARLY ?? 'price_yearly_placeholder',
    features: [
      'Everything in Monthly',
      'All Courses (including upcoming)',
      'Admin / Instructor Tools',
      'Downloadable Certificates',
      'Best Value — Save 33%',
    ],
  },
};

// ---------------------------------------------------------------------------
// Stripe initialisation
// ---------------------------------------------------------------------------

/**
 * Call once at app startup (e.g. in _layout.tsx) to set up the Stripe SDK.
 * Wraps `initStripe` from @stripe/stripe-react-native.
 */
export async function initializeStripe(): Promise<void> {
  await initStripe({
    publishableKey: STRIPE_PUBLISHABLE_KEY,
    merchantIdentifier: 'merchant.com.techsei.lms', // matches app.json
    urlScheme: 'techsei',
  });
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Thin wrapper around fetch for authenticated calls to the TechSei backend.
 * The backend holds the Stripe secret key and performs server-side validation.
 */
async function apiRequest<T>(
  endpoint: string,
  options: {
    method?: 'GET' | 'POST' | 'DELETE' | 'PATCH';
    body?: Record<string, unknown>;
    authToken?: string;
  } = {}
): Promise<T> {
  const { method = 'POST', body, authToken } = options;

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const errorData = (await response.json().catch(() => null)) as
      | { message?: string }
      | null;
    throw new Error(
      errorData?.message ?? `Request failed: HTTP ${response.status}`
    );
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Subscription management
// ---------------------------------------------------------------------------

/**
 * Create a new subscription for the given customer.
 * The backend creates a Stripe Subscription and returns the PaymentIntent
 * client secret, which you pass to `confirmPayment` in your UI.
 *
 * @param planId - One of the Stripe Price IDs (e.g. PLANS.monthly.stripePriceId)
 * @param customerId - Stripe Customer ID stored in your database
 * @param authToken - The user's Supabase JWT for server-side authorisation
 * @returns clientSecret — pass to @stripe/stripe-react-native's confirmPayment
 */
export async function createSubscription(
  planId: string,
  customerId: string,
  authToken: string
): Promise<{ clientSecret: string; subscriptionId: string }> {
  const result = await apiRequest<{
    clientSecret: string;
    subscriptionId: string;
  }>('/payments/create-subscription', {
    method: 'POST',
    body: { planId, customerId },
    authToken,
  });

  return result;
}

/**
 * Cancel a Stripe subscription at period end (the user retains access until
 * the current billing period expires).
 *
 * @param subscriptionId - The Stripe subscription ID (stripe_subscription_id in DB)
 * @param authToken - The user's Supabase JWT
 */
export async function cancelSubscription(
  subscriptionId: string,
  authToken: string
): Promise<void> {
  await apiRequest<void>('/payments/cancel-subscription', {
    method: 'POST',
    body: { subscriptionId },
    authToken,
  });
}

/**
 * Fetch the current status of a Stripe subscription from the backend.
 * Returns a Subscription object matching the TechSei DB schema.
 *
 * @param subscriptionId - The Stripe subscription ID
 * @param authToken - The user's Supabase JWT
 */
export async function getSubscriptionStatus(
  subscriptionId: string,
  authToken: string
): Promise<Subscription> {
  const result = await apiRequest<Subscription>(
    `/payments/subscription-status/${subscriptionId}`,
    { method: 'GET', authToken }
  );

  return result;
}

/**
 * Reactivate a previously cancelled subscription (if still within billing period).
 *
 * @param subscriptionId - The Stripe subscription ID
 * @param authToken - The user's Supabase JWT
 */
export async function reactivateSubscription(
  subscriptionId: string,
  authToken: string
): Promise<void> {
  await apiRequest<void>('/payments/reactivate-subscription', {
    method: 'POST',
    body: { subscriptionId },
    authToken,
  });
}

/**
 * Update a subscription to a different plan (e.g. monthly → yearly).
 * Uses Stripe's proration by default.
 *
 * @param subscriptionId - The Stripe subscription ID
 * @param newPlanId - The new Stripe Price ID
 * @param authToken - The user's Supabase JWT
 */
export async function updateSubscriptionPlan(
  subscriptionId: string,
  newPlanId: string,
  authToken: string
): Promise<{ clientSecret?: string }> {
  const result = await apiRequest<{ clientSecret?: string }>(
    '/payments/update-subscription',
    {
      method: 'POST',
      body: { subscriptionId, newPlanId },
      authToken,
    }
  );

  return result;
}

/**
 * Create or retrieve a Stripe Customer for the given user.
 * Called during signup or when the user first navigates to the subscription screen.
 *
 * @param userId - The TechSei / Supabase user ID
 * @param email - The user's email address
 * @param name - The user's display name
 * @param authToken - The user's Supabase JWT
 */
export async function getOrCreateStripeCustomer(
  userId: string,
  email: string,
  name: string,
  authToken: string
): Promise<{ customerId: string }> {
  const result = await apiRequest<{ customerId: string }>(
    '/payments/get-or-create-customer',
    {
      method: 'POST',
      body: { userId, email, name },
      authToken,
    }
  );

  return result;
}

// ---------------------------------------------------------------------------
// Payment sheet helper
// ---------------------------------------------------------------------------

/**
 * Prepare a Payment Sheet for one-time or subscription payments.
 * Returns the parameters needed to initialise @stripe/stripe-react-native's
 * `initPaymentSheet` function.
 *
 * @param customerId - Stripe Customer ID
 * @param planId - Stripe Price ID for the chosen plan
 * @param authToken - The user's Supabase JWT
 */
export async function preparePaymentSheet(
  customerId: string,
  planId: string,
  authToken: string
): Promise<{
  paymentIntentClientSecret: string;
  ephemeralKeySecret: string;
  customerId: string;
}> {
  const result = await apiRequest<{
    paymentIntentClientSecret: string;
    ephemeralKeySecret: string;
    customerId: string;
  }>('/payments/prepare-payment-sheet', {
    method: 'POST',
    body: { customerId, planId },
    authToken,
  });

  return result;
}

// ---------------------------------------------------------------------------
// Price formatting utility
// ---------------------------------------------------------------------------

/**
 * Format a price in the smallest currency unit to a display string.
 * e.g. formatPrice(999, 'usd') => '$9.99'
 *
 * @param amountInCents - Amount in the smallest unit (cents for USD)
 * @param currency - ISO 4217 currency code
 * @param locale - BCP 47 locale string for number formatting
 */
export function formatPrice(
  amountInCents: number,
  currency = 'usd',
  locale = 'en-US'
): string {
  const amount = amountInCents / 100;
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency: currency.toUpperCase(),
    minimumFractionDigits: 2,
  }).format(amount);
}

/**
 * Calculate the effective monthly cost of an annual plan.
 * e.g. effectiveMonthlyPrice(7999) => '$6.67'
 */
export function effectiveMonthlyPrice(
  yearlyAmountInCents: number,
  currency = 'usd',
  locale = 'en-US'
): string {
  return formatPrice(Math.round(yearlyAmountInCents / 12), currency, locale);
}

/**
 * Check if the current environment is using a real (production) Stripe key.
 */
export function isProductionStripe(): boolean {
  return STRIPE_PUBLISHABLE_KEY.startsWith('pk_live_');
}
