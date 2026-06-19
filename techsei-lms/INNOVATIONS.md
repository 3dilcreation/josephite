# TechSei LMS — Innovation Document

**Product:** TechSei Mobile Learning Management System
**Version:** 1.0.0
**Platform:** iOS & Android (React Native + Expo)
**Date:** June 2026

---

## Executive Summary

TechSei is a next-generation mobile LMS that combines AI-powered personalised tutoring, deep gamification mechanics, and enterprise-grade administration into a single cross-platform app. Unlike conventional e-learning platforms, TechSei puts engagement at the centre — every screen is designed to make learners want to come back — while giving administrators real-time intelligence to manage courses and students at scale. With support for 10 languages (including right-to-left Arabic), built-in Stripe subscriptions, and a Claude-powered multilingual AI tutor, TechSei is built to serve a global, mobile-first audience.

---

## 1. AI-Powered Multilingual Tutoring

> *"Like having a personal tutor who speaks your language and knows exactly where you are in the course."*

### What we built
- **Claude API integration** (claude-sonnet-4-6) embedded directly in the student app as an always-available AI tutor
- **Student-context injection** — the AI automatically knows the student's name, current level, active course, streak, and subscription tier before every conversation
- **Multilingual responses** — students pick their language (10 options) and the AI responds fluently in that language, while always keeping code examples in English for clarity
- **Conversation memory** — stores the last 10 exchanges per session, giving the AI context for follow-up questions
- **Guided discovery** — suggested question chips ("Explain recursion", "Help with Python lists") reduce blank-screen anxiety for new learners
- **Code block rendering** — AI responses containing code are displayed in styled monospace blocks, not raw text
- **Resilience** — exponential backoff retry (1s, 2s) with graceful fallback messages if the API is unavailable

### Why it's innovative
Most LMS platforms bolt on a third-party chatbot with no course context. TechSei's AI tutor is woven into the learning experience — it knows what lesson the student just completed, adjusts its language automatically, and escalates complexity based on the student's level.

---

## 2. Gamification & Progression Engine

> *"Turning learning into a game students don't want to stop playing."*

### XP & Level System
- **10-level progression** with meaningful titles:
  `Newcomer → Explorer → Apprentice → Scholar → Rising Star → Adept → Expert → Master → Grandmaster → Legend`
- **XP thresholds** designed for sustained engagement (not too fast, not too slow):

| Level | XP Required |
|---|---|
| 1 — Newcomer | 0 |
| 2 — Explorer | 100 |
| 3 — Apprentice | 250 |
| 4 — Scholar | 500 |
| 5 — Rising Star | 1,000 |
| 6 — Adept | 2,000 |
| 7 — Expert | 3,500 |
| 8 — Master | 5,500 |
| 9 — Grandmaster | 8,000 |
| 10 — Legend | 12,000 |

- **XP reward events** tied to meaningful actions:

| Action | XP Reward |
|---|---|
| Complete a lesson | +50 XP |
| Perfect quiz score (100%) | +100 XP |
| Daily streak maintained | +25 XP |
| First course enrolled | +200 XP |
| Full course completed | +500 XP |

### Streak System
- Tracks consecutive days of activity using ISO date comparison
- Automatic streak reset if a day is missed (with a warning state on the home screen)
- Animated 🔥 fire pulse effect on the home screen when streak is active
- Streak bonus XP (+25) automatically awarded each active day

### Badges & Achievements
- **10 pre-seeded achievement badges** covering key milestones:
  - 🎯 First Step, ⚡ Quick Learner, 🔥 Week Warrior, 🌟 Month Master
  - ⭐ Rising Star, 💻 Code Master, 🏆 Perfect Score, 🤝 Social Learner
  - 💎 XP Hunter, 👑 Elite Learner
- **Rarity tiers**: Common → Rare → Epic → Legendary (visual differentiation with glow effects)
- Earned badges shown with animated glow; locked badges shown greyed with XP requirement

### Leaderboard
- Four views: **This Week / This Month / All Time / Friends**
- **Podium display** for top 3 (gold crown, silver, bronze) with avatars
- **Rank change indicators** (↑3, ↓2, NEW) in green/red for live competition feel
- Current user's row highlighted in purple so they always know their position

### Visual Gamification
- **Animated SVG circular XP ring** on the progress screen (1200ms smooth animation)
- **GitHub-style 30-day activity heatmap** — colour-coded squares showing daily engagement intensity
- **Weekly XP bar chart** — 7-day view of learning consistency
- **Level-up push notification** via Expo Notifications

---

## 3. Intelligent Adaptive Onboarding

> *"The app learns about the student before the student starts learning."*

### 3-Step Animated Wizard
1. **Learning Goal Selection** — student picks from 6 tech tracks (Web Development, Data Science, Mobile Apps, AI/ML, Cybersecurity, Cloud Computing). This seeds course recommendations.
2. **Language Preference** — choose from 10 languages with flag + native script display. Immediately applied to the full app UI.
3. **Daily Commitment** — pledge 10, 20, 30, or 60 minutes per day. Each option shows a personalised motivational message. Sets the daily XP goal bar on the home screen.

### Technical Highlights
- Slide-transition animations between steps
- Progress dots (active dot expands to a pill shape)
- Skip + back navigation so no student feels trapped
- All preferences saved to Supabase `profiles` in a single `upsert` on completion

---

## 4. Multi-Modal Lesson Player

> *"One app, four completely different ways to learn."*

The lesson player supports four distinct content types, each with a purpose-built UI:

| Content Type | Player | Use Case |
|---|---|---|
| **Video** | expo-av with play/pause, seek, speed control | Lectures, tutorials |
| **Text** | Rich renderer with code block styling | Reading material, guides |
| **Quiz** | Full MCQ flow (see below) | Knowledge checks |
| **Interactive** | WebView | Code sandboxes, simulations |

### Quiz Engine (standalone component)
- One question at a time with smooth slide-transition animations
- **Immediate visual feedback**: correct answer highlighted green, wrong answer red, correct answer revealed
- **Explanation text** shown after every answer to reinforce learning (not just right/wrong)
- **Progress indicator**: "Question 3 of 10" with fill bar
- **Score screen**: percentage, pass/fail status, XP earned, retry option
- Pass threshold configurable per quiz (default 70%)

### Lesson Completion Flow
- "Mark as Complete" → triggers XP award → shows animated `+50 XP` notification → updates progress bar
- Next lesson button navigates within module without leaving the player
- Breadcrumb trail: Course > Module > Lesson

---

## 5. Real-Time Analytics & Admin Intelligence

> *"Admins know what's working before students do."*

### Admin Dashboard
- Live stats: Total Students, Active Courses, Monthly Revenue, Completion Rate
- Activity feed showing the last 10 platform events (payments, signups, completions, enrollments) with timestamps and colour-coded icons
- Pending actions queue: courses awaiting review, support tickets, flagged content — all with badge counts

### Analytics Screen
- **Period selector**: Last 7 days / 30 days / 90 days / All time
- **Daily enrollment bar chart** — built with pure React Native SVG (no external charting library)
- **Daily revenue bar chart** — same approach, gold bars
- **Top 5 courses** by enrollment with horizontal completion-rate bars
- **Engagement metrics**: Avg. session time, Daily Active Users, XP earned per week, Streak leaders
- **Export Report** button (CSV)

### Course Builder (3-Step Wizard)
- **Step 1** — Basic Info: title, description, category selector, thumbnail picker (Expo ImagePicker), instructor name, free/premium toggle
- **Step 2** — Curriculum: add modules, add lessons within modules, set content type per lesson (video/text/quiz), reorder with up/down arrows
- **Step 3** — Settings & Publish: difficulty level, preview summary, save-as-draft or publish to live

### Student Management
- Search by name or email with live filtering
- Filter by tier (Free/Pro), sort by name/XP/last active/join date
- Student detail modal: stats, course enrollment history, actions (Grant Pro, Suspend, Delete)
- Bulk operations: select multiple students → notify or export
- Add student form with name, email, temporary password

---

## 6. Secure Multi-Tenant Architecture

> *"Enterprise-grade security with zero extra configuration."*

### Supabase Row-Level Security (18 Policies)
Every table has RLS enabled. Users can only read and write their own data. Admins get elevated access through role-checked policies — no separate admin database needed.

| Policy | Behaviour |
|---|---|
| Profiles | Users see only their own row; admins see all |
| Courses | Anyone reads published courses; only admins write |
| Enrollments | Students manage only their own enrollments |
| Lesson Progress | Students manage only their own progress |
| Chat Messages | Students see only their own chat history |
| Subscriptions | Students see only their own subscription |

### Database Triggers (Automatic, Zero-Code Maintenance)
- **`handle_new_user`** — fires on every `auth.users` INSERT, automatically creates a corresponding `profiles` row. New users are never orphaned.
- **`update_course_student_count`** — fires on every new `enrollments` INSERT, keeps `courses.total_students` accurate in real time without manual queries.
- **`set_updated_at`** — fires on course updates, auto-stamps the `updated_at` timestamp.

### Database Schema (12 Tables)
`profiles` · `courses` · `modules` · `lessons` · `quizzes` · `enrollments` · `lesson_progress` · `badges` · `student_badges` · `chat_messages` · `subscriptions` · `certificates`

Performance indexes on all hot query paths (category, published, student_id, course_id, created_at desc).

---

## 7. Internationalisation — 10 Languages + RTL

> *"Learning in your mother tongue is always faster."*

### Languages Supported

| Code | Language | Native Script | RTL |
|---|---|---|---|
| en | English | English | No |
| hi | Hindi | हिन्दी | No |
| ta | Tamil | தமிழ் | No |
| te | Telugu | తెలుగు | No |
| kn | Kannada | ಕನ್ನಡ | No |
| bn | Bengali | বাংলা | No |
| mr | Marathi | मराठी | No |
| ar | Arabic | العربية | **Yes** |
| fr | French | Français | No |
| es | Spanish | Español | No |

### How It Works
- `useTranslation()` React hook reads the current language from the auth store and returns a `t(key)` function
- **Dot-notation keys**: `t('auth.login')` → "Sign In" / "साइन इन करें"
- **Variable interpolation**: `t('home.greeting', { name: 'Priya' })` → "Good morning, Priya!"
- **Automatic English fallback** — if a key is missing in the selected language, silently falls back to English with a console warning (no crashes, no blank text)
- Language change is **instant** — updates Zustand store, re-renders all components, and persists to Supabase in one action
- RTL flag per language for future layout mirroring

---

## 8. Subscription & Monetisation

> *"Simple pricing, zero friction, full Stripe integration."*

### Plans

| Plan | Price | Included |
|---|---|---|
| **Free** | $0/month | Free courses only, basic AI tutor (limited) |
| **Pro Monthly** | $9.99/month | All 50+ courses, unlimited AI tutor, offline downloads, certificates, priority support |
| **Pro Yearly** | $79.99/year | Everything in Pro + all future courses + admin/instructor tools + **33% savings** |

### Full Stripe Lifecycle (Backend-Delegated)
The Stripe secret key never touches the app bundle. All sensitive operations go through a backend API:

| Function | What it does |
|---|---|
| `createSubscription` | Creates subscription + returns PaymentIntent client secret |
| `cancelSubscription` | Cancels at period end (student keeps access until expiry) |
| `reactivateSubscription` | Reactivates a cancelled subscription before expiry |
| `updateSubscriptionPlan` | Upgrades/downgrades with automatic proration |
| `getOrCreateStripeCustomer` | Idempotent customer creation |
| `preparePaymentSheet` | Initialises Stripe's native payment sheet |

---

## Innovation Quick-Reference

| # | Innovation | What It Does | Inspired By |
|---|---|---|---|
| 1 | **Claude AI Tutor** | Context-aware multilingual tutoring powered by Anthropic | Khanmigo (Khan Academy) |
| 2 | **10-Level XP System** | Newcomer → Legend progression with meaningful milestones | Duolingo, Stack Overflow |
| 3 | **Daily Streak Engine** | Date-comparison streak tracking with push notifications | Duolingo, GitHub |
| 4 | **Adaptive Onboarding** | 3-step wizard captures goals, language, and daily commitment | Duolingo, Brilliant.org |
| 5 | **4 Lesson Types** | Video, Text, Quiz, Interactive in one unified player | Coursera, Codecademy |
| 6 | **MCQ Quiz Engine** | Per-answer explanations, score screen, XP rewards, retry | Brilliant.org |
| 7 | **Activity Heatmap** | 30-day GitHub-style calendar showing learning consistency | GitHub |
| 8 | **Podium Leaderboard** | Gold/silver/bronze podium + rank-change arrows | Duolingo |
| 9 | **10-Language i18n** | Full UI translation + RTL support + AI responds in-language | WhatsApp, Duolingo |
| 10 | **Admin Course Builder** | 3-step wizard to build and publish courses without code | Teachable, Thinkific |
| 11 | **RLS Security** | 18 Supabase policies ensuring zero cross-user data leakage | Enterprise SaaS |
| 12 | **Database Triggers** | Auto-create profile on signup, auto-count enrollments | Firebase, Supabase |
| 13 | **Full Stripe Lifecycle** | Subscribe, cancel, reactivate, upgrade with proration | Coursera, Udemy |
| 14 | **Animated SVG XP Ring** | Circular progress ring renders XP-to-next-level smoothly | Apple Fitness, Strava |
| 15 | **Skeleton Loaders** | Shimmer placeholders prevent layout shift during data fetch | Facebook, LinkedIn |

---

## Tech Stack Summary

| Layer | Technology | Why |
|---|---|---|
| Mobile Framework | React Native + Expo SDK 51 | Cross-platform iOS + Android from one codebase |
| Routing | Expo Router (file-based) | Convention over configuration, deep linking built-in |
| State Management | Zustand + AsyncStorage | Lightweight, offline-first, no boilerplate |
| Backend / Database | Supabase (PostgreSQL) | Real-time, RLS, triggers, free tier, no server to manage |
| Authentication | Supabase Auth | Email/password + OAuth ready, session management built-in |
| AI Tutor | Anthropic Claude API | Best-in-class reasoning, multilingual, context window |
| Payments | Stripe React Native | Most trusted payment processor, full subscription lifecycle |
| UI | Expo LinearGradient + Ionicons | Consistent dark-theme design across all screens |
| Animations | React Native Animated API + SVG | Smooth 60fps animations without Reanimated complexity |
| Notifications | Expo Notifications | Cross-platform push, local notifications, no FCM setup |

---

*TechSei LMS — Built to make every student want to come back tomorrow.*

*© 2026 TechSei. All rights reserved.*
