# TechSei LMS — Mobile Learning Platform

A feature-rich mobile LMS app for TechSei built with React Native + Expo, Supabase, Claude AI, and Stripe.

## Features

### Student App
- **Auth**: Email/password + Google SSO with role-based routing
- **Onboarding**: Learning goals, language preference, daily goal setup
- **Course Player**: Video, text, quiz, and interactive lessons
- **Progress Tracking**: XP system, level progression, skill map, streak calendar
- **Gamification**: XP points, levels, badges, daily streaks, leaderboard
- **AI Tutor**: Claude-powered chatbot in 10+ languages
- **Multilingual**: Switch UI language anytime (EN, HI, TA, TE, BN, AR, FR, ES)
- **Certificates**: Auto-generated PDF on course completion
- **Offline Mode**: Download lessons for offline study

### Admin App
- **Dashboard**: Real-time stats (students, revenue, completions)
- **Student Management**: Add, edit, suspend, export students
- **Course Builder**: WYSIWYG course editor with modules + lessons
- **Quiz Builder**: Multiple-choice questions with explanations
- **Analytics**: Revenue charts, completion rates, engagement metrics
- **Push Notifications**: Send announcements to all/filtered students

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React Native + Expo (SDK 51) |
| Routing | Expo Router (file-based) |
| State | Zustand |
| Backend | Supabase (PostgreSQL + Auth + Storage + Realtime) |
| AI Chatbot | Claude API (claude-sonnet-4-6) |
| Payments | Stripe (React Native SDK) |
| Languages | 10 languages with RTL support |

## Getting Started

### 1. Install dependencies
```bash
cd techsei-lms
npm install
```

### 2. Set up environment
```bash
cp .env.example .env
# Fill in your Supabase, Anthropic, and Stripe keys
```

### 3. Set up Supabase
1. Create a project at [supabase.com](https://supabase.com)
2. Run the migration: `supabase/migrations/001_initial_schema.sql`
3. Enable Email Auth in Supabase Auth settings
4. (Optional) Enable Google OAuth provider

### 4. Set up Stripe
1. Create account at [stripe.com](https://stripe.com)
2. Create Monthly and Yearly subscription products
3. Copy Price IDs to `.env`

### 5. Start the app
```bash
npx expo start
```
Scan the QR code with Expo Go (iOS/Android) or press `a`/`i` for emulator.

## Project Structure

```
techsei-lms/
├── app/                    # Expo Router file-based routes
│   ├── _layout.tsx         # Root layout + auth guard
│   ├── index.tsx           # Redirect based on auth
│   ├── auth/               # Login, Signup, Onboarding
│   ├── student/            # Student tabs + screens
│   │   ├── home.tsx
│   │   ├── courses/
│   │   ├── progress.tsx
│   │   ├── chatbot.tsx
│   │   ├── leaderboard.tsx
│   │   └── profile.tsx
│   └── admin/              # Admin tabs + screens
│       ├── dashboard.tsx
│       ├── students/
│       ├── courses/
│       └── analytics.tsx
├── components/             # Reusable components
│   ├── common/             # Shared (CourseCard, ProgressBar)
│   ├── student/            # Student-specific (Quiz, Badge, XP)
│   └── admin/              # Admin-specific (StatCard, StudentRow)
├── lib/                    # API integrations
│   ├── supabase.ts         # Supabase client + helpers
│   ├── claude.ts           # Claude AI chatbot
│   └── stripe.ts           # Stripe payments
├── stores/                 # Zustand state management
│   ├── authStore.ts
│   ├── courseStore.ts
│   ├── gamificationStore.ts
│   └── chatStore.ts
├── constants/              # App-wide constants
│   ├── colors.ts           # Design tokens
│   ├── fonts.ts
│   └── i18n/               # Translations (EN, HI, TA, TE, AR, FR, ES)
├── types/                  # TypeScript interfaces
└── supabase/
    └── migrations/         # SQL schema
```

## Database Schema (Supabase)

| Table | Description |
|---|---|
| `profiles` | Extends auth.users with XP, level, streak, subscription |
| `courses` | Course catalog with metadata |
| `modules` | Course sections/chapters |
| `lessons` | Individual lessons (video/text/quiz) |
| `quizzes` | Quiz questions in JSONB |
| `enrollments` | Student ↔ course relationships |
| `lesson_progress` | Per-lesson completion + scores |
| `badges` | Badge definitions |
| `student_badges` | Earned badges per student |
| `chat_messages` | AI chatbot history |
| `subscriptions` | Stripe subscription records |
| `certificates` | Earned certificates |

## Supported Languages

| Code | Language | Native | RTL |
|---|---|---|---|
| en | English | English | No |
| hi | Hindi | हिन्दी | No |
| ta | Tamil | தமிழ் | No |
| te | Telugu | తెలుగు | No |
| bn | Bengali | বাংলা | No |
| ar | Arabic | عربية | Yes |
| fr | French | Français | No |
| es | Spanish | Español | No |

## Gamification System

| Event | XP Reward |
|---|---|
| Complete a lesson | +50 XP |
| Perfect quiz score | +100 XP |
| Daily streak | +25 XP |
| First course enroll | +200 XP |
| Course completion | +500 XP |

Levels: 1 (0 XP) → 2 (100) → 3 (250) → 4 (500) → 5 (1000) → ... → 10 (12000)

## Subscription Plans

| Plan | Price | Access |
|---|---|---|
| Free | $0 | Free courses only |
| Pro Monthly | $9.99/mo | All courses |
| Pro Yearly | $79.99/yr | All courses + 33% savings |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Proprietary — TechSei © 2026
