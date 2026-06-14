# 3DIL Creation Mobile App

A full-featured React Native (Expo) mobile app for **3DIL Creation** — a premium 3D printing services company in India.

## Features

### Services
- 3D Printing (FDM & Resin)
- Custom Medals & Trophies
- 3D Modeling & Design
- Architectural Models
- Statues & Miniatures

### Revenue Features
- **E-commerce** — Direct product sales with UPI/Card/COD via Razorpay
- **Custom Quote System** — WhatsApp & email quote flow
- **Subscription Plans** — Starter (₹499), Pro (₹1299), Enterprise (₹3499)/month
- **Loyalty Program** — Bronze → Silver → Gold → Platinum tiers
- **Referral Program** — Earn 200 points per successful referral

### Innovative Features
- **AR Preview** — See 3D models in your real space via camera
- **AI Quote Calculator** — Auto-estimate price by size, material & complexity
- **Real-time Order Tracking** — 6-stage live timeline (Placed → Designed → Printing → QC → Shipped → Delivered)
- **3D Model Viewer** — Interactive product preview

### Screens (21 total)
- Splash + Onboarding (3 slides)
- Home — hero, stats, services, portfolio, testimonials
- Products — catalog with search & filters
- Product Detail — material/color/size selector + AR view
- Custom Order — 4-step wizard with live price calculator
- AI Quote Calculator — instant estimate + WhatsApp share
- Portfolio — gallery with category filters
- Cart + Checkout — full order flow with payment selection
- Order Tracking — live status timeline
- Loyalty Rewards — points, tiers, redemption
- Subscription Plans — monthly/annual toggle
- AR Viewer — camera-based product placement
- Blog — 3D printing articles
- Contact — WhatsApp, form, map
- Login / Register — with referral code
- Profile — orders, settings, tier info

## Quick Start

### Prerequisites
- Node.js 18+
- Expo CLI: `npm install -g expo-cli`
- Expo Go app on your phone (iOS/Android)

### Setup
```bash
cd 3dil-creation-app
npm install
expo start
```

Scan the QR code with Expo Go to run on your phone.

### Build for Production
```bash
npm install -g eas-cli
eas login
eas build --platform android  # For Play Store
eas build --platform ios      # For App Store
```

## Configuration

### Before Launch
1. Replace `assets/icon.png` (1024x1024) and `assets/splash.png` with real images
2. Update `assets/adaptive-icon.png` (1024x1024) for Android
3. Update WhatsApp number in `QuoteCalculatorScreen.tsx` and `ContactScreen.tsx`
4. Update contact email in `ContactScreen.tsx`
5. Add Razorpay API key in `CheckoutScreen.tsx`
6. Replace mock data in `src/data/` with real API calls

### Coupon Codes (Demo)
- `FIRST10` — 10% off
- `DIWALI20` — 20% off
- `3DIL15` — 15% off
- `WELCOME25` — 25% off

## Tech Stack
- React Native 0.74 + Expo SDK 51
- TypeScript
- React Navigation 6 (Stack + Bottom Tabs)
- Expo Linear Gradient
- @expo/vector-icons (Ionicons)
- AsyncStorage (local storage)
- React Context API (state management)

## Monetization Strategy
1. **Product Sales** — direct e-commerce revenue
2. **Custom Orders** — high-margin custom jobs
3. **Subscriptions** — recurring monthly revenue
4. **Loyalty** — increases repeat purchases
5. **Referrals** — organic growth + acquisition

---
Made with ❤️ for 3DIL Creation | www.3dilcreation.in
