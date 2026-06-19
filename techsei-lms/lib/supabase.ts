// ============================================================
// TechSei LMS — Supabase Client & Helpers
// ============================================================

import { createClient, SupabaseClient } from '@supabase/supabase-js';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type {
  User,
  Profile,
  Course,
  Module,
  Lesson,
  Quiz,
  QuizQuestion,
  QuizAttempt,
  Enrollment,
  LessonProgress,
  Badge,
  StudentBadge,
  ChatMessage,
  Subscription,
  LeaderboardEntry,
  AppNotification,
  CourseAnalytics,
  PlatformStats,
  LanguageCode,
  SubscriptionTier,
} from '@/types';

// ---------------------------------------------------------------------------
// Environment constants
// Replace these with your actual Supabase project values.
// In production, load from expo-constants or a .env loader.
// ---------------------------------------------------------------------------

export const SUPABASE_URL =
  process.env.EXPO_PUBLIC_SUPABASE_URL ?? 'https://your-project.supabase.co';

export const SUPABASE_ANON_KEY =
  process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY ?? 'your-anon-key-here';

// ---------------------------------------------------------------------------
// Database schema type definitions
// ---------------------------------------------------------------------------

export interface Database {
  public: {
    Tables: {
      users: {
        Row: User;
        Insert: Omit<User, 'id' | 'created_at'>;
        Update: Partial<Omit<User, 'id' | 'created_at'>>;
      };
      profiles: {
        Row: Profile;
        Insert: Omit<Profile, 'total_courses_completed' | 'total_lessons_completed' | 'total_time_spent_seconds'>;
        Update: Partial<Profile>;
      };
      courses: {
        Row: Course;
        Insert: Omit<Course, 'id' | 'created_at' | 'rating' | 'total_students'>;
        Update: Partial<Omit<Course, 'id' | 'created_at'>>;
      };
      modules: {
        Row: Omit<Module, 'lessons'>;
        Insert: Omit<Module, 'id' | 'lessons'>;
        Update: Partial<Omit<Module, 'id' | 'lessons'>>;
      };
      lessons: {
        Row: Lesson;
        Insert: Omit<Lesson, 'id'>;
        Update: Partial<Omit<Lesson, 'id'>>;
      };
      quizzes: {
        Row: Omit<Quiz, 'questions'>;
        Insert: Omit<Quiz, 'id' | 'questions'>;
        Update: Partial<Omit<Quiz, 'id' | 'questions'>>;
      };
      quiz_questions: {
        Row: QuizQuestion;
        Insert: Omit<QuizQuestion, 'id'>;
        Update: Partial<Omit<QuizQuestion, 'id'>>;
      };
      quiz_attempts: {
        Row: QuizAttempt;
        Insert: Omit<QuizAttempt, 'id' | 'attempted_at'>;
        Update: Partial<Omit<QuizAttempt, 'id' | 'attempted_at'>>;
      };
      enrollments: {
        Row: Omit<Enrollment, 'course'>;
        Insert: Omit<Enrollment, 'id' | 'enrolled_at' | 'completion_percentage' | 'course'>;
        Update: Partial<Omit<Enrollment, 'id' | 'enrolled_at' | 'course'>>;
      };
      lesson_progress: {
        Row: LessonProgress;
        Insert: Omit<LessonProgress, 'id'>;
        Update: Partial<Omit<LessonProgress, 'id'>>;
      };
      badges: {
        Row: Badge;
        Insert: Omit<Badge, 'id'>;
        Update: Partial<Omit<Badge, 'id'>>;
      };
      student_badges: {
        Row: StudentBadge;
        Insert: Omit<StudentBadge, 'badge'>;
        Update: never;
      };
      chat_messages: {
        Row: ChatMessage;
        Insert: Omit<ChatMessage, 'id' | 'created_at' | 'is_loading'>;
        Update: never;
      };
      subscriptions: {
        Row: Subscription;
        Insert: Omit<Subscription, 'id'>;
        Update: Partial<Omit<Subscription, 'id' | 'student_id'>>;
      };
      notifications: {
        Row: AppNotification;
        Insert: Omit<AppNotification, 'id' | 'created_at'>;
        Update: Partial<Pick<AppNotification, 'read'>>;
      };
    };
    Views: {
      leaderboard_weekly: {
        Row: LeaderboardEntry;
      };
      leaderboard_monthly: {
        Row: LeaderboardEntry;
      };
      leaderboard_all_time: {
        Row: LeaderboardEntry;
      };
      course_analytics: {
        Row: CourseAnalytics;
      };
      platform_stats: {
        Row: PlatformStats;
      };
    };
    Functions: {
      increment_xp: {
        Args: { p_user_id: string; p_xp: number };
        Returns: { new_xp: number; new_level: number; leveled_up: boolean };
      };
      update_streak: {
        Args: { p_user_id: string };
        Returns: { new_streak: number; streak_broken: boolean };
      };
      get_course_progress: {
        Args: { p_student_id: string; p_course_id: string };
        Returns: { completion_percentage: number; completed_lessons: number; total_lessons: number };
      };
    };
    Enums: {
      user_role: 'student' | 'admin';
      subscription_tier: 'free' | 'pro' | 'premium';
      content_type: 'video' | 'text' | 'quiz' | 'interactive';
      enrollment_status: 'active' | 'completed' | 'paused';
      badge_type: 'achievement' | 'course' | 'streak';
      subscription_plan: 'monthly' | 'yearly';
      subscription_status: 'active' | 'cancelled' | 'expired';
      notification_type:
        | 'streak_reminder'
        | 'new_course'
        | 'badge_earned'
        | 'level_up'
        | 'enrollment_complete'
        | 'quiz_result'
        | 'subscription_expiry';
    };
  };
}

// ---------------------------------------------------------------------------
// Supabase client singleton
// ---------------------------------------------------------------------------

export const supabase: SupabaseClient<Database> = createClient<Database>(
  SUPABASE_URL,
  SUPABASE_ANON_KEY,
  {
    auth: {
      storage: AsyncStorage,
      autoRefreshToken: true,
      persistSession: true,
      detectSessionInUrl: false,
    },
    realtime: {
      params: {
        eventsPerSecond: 10,
      },
    },
  }
);

// ---------------------------------------------------------------------------
// Auth helpers
// ---------------------------------------------------------------------------

/**
 * Sign in with email and password.
 * Returns the authenticated User row from the public.users table.
 */
export async function signIn(
  email: string,
  password: string
): Promise<{ user: User; error: null } | { user: null; error: string }> {
  const { data, error } = await supabase.auth.signInWithPassword({ email, password });

  if (error || !data.user) {
    return { user: null, error: error?.message ?? 'Sign-in failed.' };
  }

  const { data: userRow, error: userError } = await supabase
    .from('users')
    .select('*')
    .eq('id', data.user.id)
    .single();

  if (userError || !userRow) {
    return { user: null, error: userError?.message ?? 'User record not found.' };
  }

  return { user: userRow, error: null };
}

/**
 * Sign up a new user with email, password, display name, and role.
 * Creates both the auth record and the public.users + public.profiles rows.
 */
export async function signUp(
  email: string,
  password: string,
  name: string,
  role: 'student' | 'admin' = 'student'
): Promise<{ user: User; error: null } | { user: null; error: string }> {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: { name, role },
    },
  });

  if (error || !data.user) {
    return { user: null, error: error?.message ?? 'Sign-up failed.' };
  }

  // Insert into public.users (trigger may also do this; upsert is safe)
  const { data: userRow, error: userError } = await supabase
    .from('users')
    .upsert({
      id: data.user.id,
      email,
      name,
      role,
      avatar_url: null,
      language_pref: 'en' as LanguageCode,
    })
    .select()
    .single();

  if (userError || !userRow) {
    return { user: null, error: userError?.message ?? 'Failed to create user record.' };
  }

  // Create default profile
  await supabase.from('profiles').upsert({
    user_id: data.user.id,
    xp: 0,
    streak: 0,
    level: 1,
    subscription_tier: 'free' as SubscriptionTier,
    last_activity: null,
  });

  return { user: userRow, error: null };
}

/**
 * Sign the current user out and clear the local session.
 */
export async function signOut(): Promise<{ error: string | null }> {
  const { error } = await supabase.auth.signOut();
  return { error: error?.message ?? null };
}

// ---------------------------------------------------------------------------
// User & Profile helpers
// ---------------------------------------------------------------------------

/**
 * Returns the currently authenticated User row, or null if not signed in.
 */
export async function getCurrentUser(): Promise<User | null> {
  const { data: { user: authUser } } = await supabase.auth.getUser();
  if (!authUser) return null;

  const { data } = await supabase
    .from('users')
    .select('*')
    .eq('id', authUser.id)
    .single();

  return data ?? null;
}

/**
 * Returns the Profile row for the given user_id.
 */
export async function getUserProfile(userId: string): Promise<Profile | null> {
  const { data } = await supabase
    .from('profiles')
    .select('*')
    .eq('user_id', userId)
    .single();

  return data ?? null;
}

/**
 * Partially update a profile.
 */
export async function updateProfile(
  userId: string,
  updates: Partial<Profile>
): Promise<{ error: string | null }> {
  const { error } = await supabase
    .from('profiles')
    .update(updates)
    .eq('user_id', userId);

  return { error: error?.message ?? null };
}

/**
 * Update editable user fields (name, language_pref).
 */
export async function updateUser(
  userId: string,
  updates: Partial<Pick<User, 'name' | 'language_pref' | 'avatar_url'>>
): Promise<{ error: string | null }> {
  const { error } = await supabase
    .from('users')
    .update(updates)
    .eq('id', userId);

  return { error: error?.message ?? null };
}

/**
 * Upload an avatar image for the given user.
 * Accepts a local file URI (from expo-image-picker) and uploads to Supabase Storage.
 * Returns the public URL of the uploaded avatar.
 */
export async function uploadAvatar(
  userId: string,
  fileUri: string
): Promise<{ url: string; error: null } | { url: null; error: string }> {
  const ext = fileUri.split('.').pop()?.toLowerCase() ?? 'jpg';
  const mimeType = ext === 'png' ? 'image/png' : 'image/jpeg';
  const fileName = `${userId}/avatar.${ext}`;

  // Read file as base64 via fetch (works with Expo file URIs)
  const response = await fetch(fileUri);
  const blob = await response.blob();

  const { error: uploadError } = await supabase.storage
    .from('avatars')
    .upload(fileName, blob, { contentType: mimeType, upsert: true });

  if (uploadError) {
    return { url: null, error: uploadError.message };
  }

  const { data: publicUrlData } = supabase.storage
    .from('avatars')
    .getPublicUrl(fileName);

  const publicUrl = publicUrlData.publicUrl;

  // Persist the URL on the users row
  await supabase.from('users').update({ avatar_url: publicUrl }).eq('id', userId);

  return { url: publicUrl, error: null };
}

// ---------------------------------------------------------------------------
// Course helpers
// ---------------------------------------------------------------------------

/**
 * Fetch all published courses, optionally filtered by category.
 */
export async function getCourses(options?: {
  category?: string;
  premiumOnly?: boolean;
  freeOnly?: boolean;
  limit?: number;
  offset?: number;
}): Promise<Course[]> {
  let query = supabase.from('courses').select('*');

  if (options?.category) query = query.eq('category', options.category);
  if (options?.premiumOnly) query = query.eq('is_premium', true);
  if (options?.freeOnly) query = query.eq('is_premium', false);
  if (options?.limit) query = query.limit(options.limit);
  if (options?.offset) query = query.range(options.offset, options.offset + (options.limit ?? 10) - 1);

  query = query.order('created_at', { ascending: false });

  const { data } = await query;
  return (data as Course[]) ?? [];
}

/**
 * Fetch a single course with its modules and lessons.
 */
export async function getCourseWithContent(courseId: string): Promise<Course | null> {
  const { data: course } = await supabase
    .from('courses')
    .select('*')
    .eq('id', courseId)
    .single();

  if (!course) return null;

  const { data: modules } = await supabase
    .from('modules')
    .select('*')
    .eq('course_id', courseId)
    .order('order_index');

  const modulesWithLessons: Module[] = await Promise.all(
    (modules ?? []).map(async (mod) => {
      const { data: lessons } = await supabase
        .from('lessons')
        .select('*')
        .eq('module_id', mod.id)
        .order('order_index');
      return { ...mod, lessons: (lessons as Lesson[]) ?? [] };
    })
  );

  return { ...(course as Course), modules: modulesWithLessons };
}

// ---------------------------------------------------------------------------
// Enrollment helpers
// ---------------------------------------------------------------------------

/**
 * Enroll a student in a course.
 */
export async function enrollInCourse(
  studentId: string,
  courseId: string
): Promise<{ enrollment: Enrollment; error: null } | { enrollment: null; error: string }> {
  const { data, error } = await supabase
    .from('enrollments')
    .upsert({ student_id: studentId, course_id: courseId, status: 'active' })
    .select()
    .single();

  if (error || !data) {
    return { enrollment: null, error: error?.message ?? 'Enrollment failed.' };
  }

  return { enrollment: data as Enrollment, error: null };
}

/**
 * Get all enrollments for a student with course details.
 */
export async function getStudentEnrollments(studentId: string): Promise<Enrollment[]> {
  const { data } = await supabase
    .from('enrollments')
    .select('*, course:courses(*)')
    .eq('student_id', studentId)
    .order('enrolled_at', { ascending: false });

  return (data as Enrollment[]) ?? [];
}

// ---------------------------------------------------------------------------
// Progress helpers
// ---------------------------------------------------------------------------

/**
 * Record or update lesson progress for a student.
 */
export async function upsertLessonProgress(
  progress: Omit<LessonProgress, 'id'>
): Promise<{ error: string | null }> {
  const { error } = await supabase.from('lesson_progress').upsert(
    {
      student_id: progress.student_id,
      lesson_id: progress.lesson_id,
      completed: progress.completed,
      score: progress.score,
      time_spent_seconds: progress.time_spent_seconds,
      completed_at: progress.completed_at,
      last_position_seconds: progress.last_position_seconds,
    },
    { onConflict: 'student_id,lesson_id' }
  );

  return { error: error?.message ?? null };
}

/**
 * Get all lesson progress records for a student in a given course.
 */
export async function getLessonProgressForCourse(
  studentId: string,
  courseId: string
): Promise<LessonProgress[]> {
  const { data } = await supabase
    .from('lesson_progress')
    .select('*, lesson:lessons!inner(module_id, modules!inner(course_id))')
    .eq('student_id', studentId)
    .eq('lesson.modules.course_id', courseId);

  return (data as LessonProgress[]) ?? [];
}

// ---------------------------------------------------------------------------
// XP / Gamification helpers
// ---------------------------------------------------------------------------

/**
 * Award XP to a student. Uses a Supabase RPC to atomically update XP and level.
 */
export async function awardXP(
  userId: string,
  xpAmount: number
): Promise<{ newXP: number; newLevel: number; leveledUp: boolean; error: null } | { error: string }> {
  const { data, error } = await supabase.rpc('increment_xp', {
    p_user_id: userId,
    p_xp: xpAmount,
  });

  if (error || !data) {
    return { error: error?.message ?? 'Failed to award XP.' };
  }

  const result = data as { new_xp: number; new_level: number; leveled_up: boolean };
  return {
    newXP: result.new_xp,
    newLevel: result.new_level,
    leveledUp: result.leveled_up,
    error: null,
  };
}

/**
 * Update the student's daily streak. Call once per day when the user opens the app.
 */
export async function updateStreak(
  userId: string
): Promise<{ newStreak: number; streakBroken: boolean; error: null } | { error: string }> {
  const { data, error } = await supabase.rpc('update_streak', { p_user_id: userId });

  if (error || !data) {
    return { error: error?.message ?? 'Failed to update streak.' };
  }

  const result = data as { new_streak: number; streak_broken: boolean };
  return { newStreak: result.new_streak, streakBroken: result.streak_broken, error: null };
}

// ---------------------------------------------------------------------------
// Badge helpers
// ---------------------------------------------------------------------------

/**
 * Award a badge to a student (idempotent — ignores duplicate).
 */
export async function awardBadge(
  studentId: string,
  badgeId: string
): Promise<{ error: string | null }> {
  const { error } = await supabase
    .from('student_badges')
    .upsert({ student_id: studentId, badge_id: badgeId, earned_at: new Date().toISOString() }, {
      onConflict: 'student_id,badge_id',
      ignoreDuplicates: true,
    });

  return { error: error?.message ?? null };
}

/**
 * Get all badges earned by a student.
 */
export async function getStudentBadges(studentId: string): Promise<StudentBadge[]> {
  const { data } = await supabase
    .from('student_badges')
    .select('*, badge:badges(*)')
    .eq('student_id', studentId)
    .order('earned_at', { ascending: false });

  return (data as StudentBadge[]) ?? [];
}

// ---------------------------------------------------------------------------
// Leaderboard helpers
// ---------------------------------------------------------------------------

export async function getLeaderboard(
  period: 'weekly' | 'monthly' | 'all_time',
  limit = 50
): Promise<LeaderboardEntry[]> {
  const viewName =
    period === 'weekly'
      ? 'leaderboard_weekly'
      : period === 'monthly'
      ? 'leaderboard_monthly'
      : 'leaderboard_all_time';

  const { data } = await supabase.from(viewName).select('*').limit(limit);
  return (data as LeaderboardEntry[]) ?? [];
}

// ---------------------------------------------------------------------------
// Chat history helpers
// ---------------------------------------------------------------------------

/**
 * Save a completed chat exchange to the database.
 */
export async function saveChatMessage(
  msg: Omit<ChatMessage, 'id' | 'created_at' | 'is_loading'>
): Promise<{ id: string; error: null } | { id: null; error: string }> {
  const { data, error } = await supabase
    .from('chat_messages')
    .insert(msg)
    .select('id')
    .single();

  if (error || !data) {
    return { id: null, error: error?.message ?? 'Failed to save message.' };
  }

  return { id: (data as { id: string }).id, error: null };
}

/**
 * Retrieve the most recent N chat messages for a student.
 */
export async function getChatHistory(
  studentId: string,
  limit = 50
): Promise<ChatMessage[]> {
  const { data } = await supabase
    .from('chat_messages')
    .select('*')
    .eq('student_id', studentId)
    .order('created_at', { ascending: false })
    .limit(limit);

  return ((data as ChatMessage[]) ?? []).reverse();
}

// ---------------------------------------------------------------------------
// Subscription helpers
// ---------------------------------------------------------------------------

/**
 * Get the active subscription for a student, or null if none.
 */
export async function getActiveSubscription(studentId: string): Promise<Subscription | null> {
  const { data } = await supabase
    .from('subscriptions')
    .select('*')
    .eq('student_id', studentId)
    .eq('status', 'active')
    .order('start_date', { ascending: false })
    .limit(1)
    .single();

  return (data as Subscription) ?? null;
}

// ---------------------------------------------------------------------------
// Notification helpers
// ---------------------------------------------------------------------------

export async function getUnreadNotifications(studentId: string): Promise<AppNotification[]> {
  const { data } = await supabase
    .from('notifications')
    .select('*')
    .eq('student_id', studentId)
    .eq('read', false)
    .order('created_at', { ascending: false });

  return (data as AppNotification[]) ?? [];
}

export async function markNotificationsRead(notificationIds: string[]): Promise<void> {
  await supabase
    .from('notifications')
    .update({ read: true })
    .in('id', notificationIds);
}
