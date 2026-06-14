// ============================================================
// TechSei LMS — Core TypeScript Types
// ============================================================

// -----------------------------------------------------------
// User & Authentication
// -----------------------------------------------------------

export type UserRole = 'student' | 'admin';

export interface User {
  id: string;
  email: string;
  role: UserRole;
  name: string;
  avatar_url: string | null;
  language_pref: LanguageCode;
  created_at: string;
}

// -----------------------------------------------------------
// Profile & Gamification
// -----------------------------------------------------------

export type SubscriptionTier = 'free' | 'pro' | 'premium';

export interface Profile {
  user_id: string;
  xp: number;
  streak: number;
  level: number;
  subscription_tier: SubscriptionTier;
  last_activity: string | null;
  total_courses_completed: number;
  total_lessons_completed: number;
  total_time_spent_seconds: number;
}

// -----------------------------------------------------------
// Courses
// -----------------------------------------------------------

export type CourseCategory =
  | 'programming'
  | 'data-science'
  | 'web-development'
  | 'mobile-development'
  | 'cloud'
  | 'cybersecurity'
  | 'ai-ml'
  | 'devops'
  | 'design'
  | 'business';

export interface Course {
  id: string;
  title: string;
  description: string;
  category: CourseCategory;
  thumbnail_url: string | null;
  is_premium: boolean;
  instructor_name: string;
  rating: number;
  total_students: number;
  duration_hours: number;
  created_by: string;
  created_at: string;
  modules?: Module[];
}

// -----------------------------------------------------------
// Modules & Lessons
// -----------------------------------------------------------

export type ContentType = 'video' | 'text' | 'quiz' | 'interactive';

export interface Module {
  id: string;
  course_id: string;
  title: string;
  order_index: number;
  lessons: Lesson[];
}

export interface Lesson {
  id: string;
  module_id: string;
  title: string;
  content_type: ContentType;
  content_url: string | null;
  duration_minutes: number;
  order_index: number;
  is_preview?: boolean;
}

// -----------------------------------------------------------
// Quizzes
// -----------------------------------------------------------

export interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correct_index: number;
  explanation: string;
}

export interface Quiz {
  id: string;
  lesson_id: string;
  title: string;
  passing_score: number;
  questions: QuizQuestion[];
}

export interface QuizAttempt {
  id: string;
  student_id: string;
  quiz_id: string;
  answers: number[];
  score: number;
  passed: boolean;
  attempted_at: string;
}

// -----------------------------------------------------------
// Enrollment & Progress
// -----------------------------------------------------------

export type EnrollmentStatus = 'active' | 'completed' | 'paused';

export interface Enrollment {
  id: string;
  student_id: string;
  course_id: string;
  enrolled_at: string;
  status: EnrollmentStatus;
  completion_percentage: number;
  course?: Course;
}

export interface LessonProgress {
  id: string;
  student_id: string;
  lesson_id: string;
  completed: boolean;
  score: number | null;
  time_spent_seconds: number;
  completed_at: string | null;
  last_position_seconds?: number;
}

// -----------------------------------------------------------
// Badges & Achievements
// -----------------------------------------------------------

export type BadgeType = 'achievement' | 'course' | 'streak';

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  xp_required: number;
  badge_type: BadgeType;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
}

export interface StudentBadge {
  student_id: string;
  badge_id: string;
  earned_at: string;
  badge?: Badge;
}

// -----------------------------------------------------------
// AI Chat
// -----------------------------------------------------------

export interface ChatMessage {
  id: string;
  student_id: string;
  message: string;
  response: string;
  language: LanguageCode;
  created_at: string;
  is_loading?: boolean;
}

export interface ChatSession {
  id: string;
  student_id: string;
  title: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

// -----------------------------------------------------------
// Subscriptions & Payments
// -----------------------------------------------------------

export type SubscriptionPlan = 'monthly' | 'yearly';
export type SubscriptionStatus = 'active' | 'cancelled' | 'expired';

export interface Subscription {
  id: string;
  student_id: string;
  plan: SubscriptionPlan;
  status: SubscriptionStatus;
  start_date: string;
  end_date: string;
  stripe_subscription_id: string;
  stripe_customer_id: string;
  auto_renew: boolean;
}

// -----------------------------------------------------------
// Leaderboard
// -----------------------------------------------------------

export interface LeaderboardEntry {
  rank: number;
  user_id: string;
  name: string;
  avatar_url: string | null;
  xp: number;
  level: number;
  streak: number;
}

export type LeaderboardPeriod = 'weekly' | 'monthly' | 'all_time';

// -----------------------------------------------------------
// Languages / i18n
// -----------------------------------------------------------

export type LanguageCode = 'en' | 'hi' | 'ta' | 'te' | 'kn' | 'bn' | 'mr' | 'ar' | 'fr' | 'es';

export interface Language {
  code: LanguageCode;
  name: string;
  nativeName: string;
  flag: string;
  rtl: boolean;
}

// -----------------------------------------------------------
// Notifications
// -----------------------------------------------------------

export type NotificationType =
  | 'streak_reminder'
  | 'new_course'
  | 'badge_earned'
  | 'level_up'
  | 'enrollment_complete'
  | 'quiz_result'
  | 'subscription_expiry';

export interface AppNotification {
  id: string;
  student_id: string;
  type: NotificationType;
  title: string;
  body: string;
  data?: Record<string, unknown>;
  read: boolean;
  created_at: string;
}

// -----------------------------------------------------------
// Admin / Analytics
// -----------------------------------------------------------

export interface CourseAnalytics {
  course_id: string;
  total_enrollments: number;
  active_students: number;
  completion_rate: number;
  average_rating: number;
  average_completion_time_hours: number;
  revenue: number;
}

export interface PlatformStats {
  total_users: number;
  active_users_today: number;
  total_courses: number;
  total_enrollments: number;
  total_revenue: number;
  new_signups_this_week: number;
}

// -----------------------------------------------------------
// API Response Wrappers
// -----------------------------------------------------------

export interface ApiResponse<T> {
  data: T | null;
  error: string | null;
  status: 'success' | 'error';
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  per_page: number;
  has_more: boolean;
}

// -----------------------------------------------------------
// Navigation Param Lists (Expo Router compatible)
// -----------------------------------------------------------

export type RootStackParamList = {
  '(auth)/login': undefined;
  '(auth)/signup': undefined;
  '(tabs)': undefined;
  'course/[id]': { id: string };
  'lesson/[id]': { id: string; courseId: string };
  'quiz/[id]': { id: string };
  'profile/edit': undefined;
  'admin/dashboard': undefined;
  'admin/course-builder': { courseId?: string };
};
