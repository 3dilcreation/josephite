-- ============================================================
-- TechSei LMS — Initial Schema Migration
-- Run against a fresh Supabase project (PostgreSQL 15+)
-- ============================================================

-- ── Extensions ────────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Tables ────────────────────────────────────────────────────────────────────

-- Users profile table (extends Supabase auth.users)
-- A single flat table stores both identity fields and gamification state so
-- that the client can read everything it needs in one query.
CREATE TABLE public.profiles (
  id                       UUID        REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  email                    TEXT        NOT NULL,
  name                     TEXT        NOT NULL,
  role                     TEXT        NOT NULL DEFAULT 'student'
                                       CHECK (role IN ('student', 'admin', 'instructor')),
  avatar_url               TEXT,
  language_pref            TEXT        NOT NULL DEFAULT 'en',
  xp                       INTEGER     NOT NULL DEFAULT 0,
  level                    INTEGER     NOT NULL DEFAULT 1,
  streak                   INTEGER     NOT NULL DEFAULT 0,
  last_activity            DATE,
  subscription_tier        TEXT        NOT NULL DEFAULT 'free'
                                       CHECK (subscription_tier IN ('free', 'pro', 'premium')),
  total_courses_completed  INTEGER     NOT NULL DEFAULT 0,
  total_lessons_completed  INTEGER     NOT NULL DEFAULT 0,
  total_time_spent_seconds INTEGER     NOT NULL DEFAULT 0,
  created_at               TIMESTAMPTZ DEFAULT NOW()
);

-- Courses table
CREATE TABLE public.courses (
  id              UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
  title           TEXT        NOT NULL,
  description     TEXT,
  category        TEXT        NOT NULL,
  thumbnail_url   TEXT,
  is_premium      BOOLEAN     DEFAULT FALSE,
  instructor_name TEXT,
  rating          DECIMAL(3,2) DEFAULT 0.00,
  total_students  INTEGER     DEFAULT 0,
  duration_hours  DECIMAL(5,2) DEFAULT 0.00,
  created_by      UUID        REFERENCES public.profiles(id) ON DELETE SET NULL,
  published       BOOLEAN     DEFAULT FALSE,
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Modules table (ordered sections within a course)
CREATE TABLE public.modules (
  id          UUID    DEFAULT uuid_generate_v4() PRIMARY KEY,
  course_id   UUID    NOT NULL REFERENCES public.courses(id) ON DELETE CASCADE,
  title       TEXT    NOT NULL,
  order_index INTEGER NOT NULL DEFAULT 0,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Lessons table (ordered items within a module)
CREATE TABLE public.lessons (
  id               UUID    DEFAULT uuid_generate_v4() PRIMARY KEY,
  module_id        UUID    NOT NULL REFERENCES public.modules(id) ON DELETE CASCADE,
  title            TEXT    NOT NULL,
  content_type     TEXT    NOT NULL
                           CHECK (content_type IN ('video', 'text', 'quiz', 'interactive')),
  content_url      TEXT,
  content_text     TEXT,
  duration_minutes INTEGER DEFAULT 0,
  order_index      INTEGER NOT NULL DEFAULT 0,
  is_preview       BOOLEAN DEFAULT FALSE,
  created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Quizzes table (one quiz per lesson, questions stored as JSONB)
CREATE TABLE public.quizzes (
  id          UUID    DEFAULT uuid_generate_v4() PRIMARY KEY,
  lesson_id   UUID    NOT NULL REFERENCES public.lessons(id) ON DELETE CASCADE,
  title       TEXT    NOT NULL,
  questions   JSONB   NOT NULL DEFAULT '[]',
  pass_score  INTEGER DEFAULT 70,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Enrollments table (student → course relationship)
CREATE TABLE public.enrollments (
  id           UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  student_id   UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  course_id    UUID NOT NULL REFERENCES public.courses(id) ON DELETE CASCADE,
  status       TEXT DEFAULT 'active'
               CHECK (status IN ('active', 'completed', 'paused')),
  enrolled_at  TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  UNIQUE (student_id, course_id)
);

-- Lesson progress table (tracks per-lesson completion + quiz scores)
CREATE TABLE public.lesson_progress (
  id                  UUID    DEFAULT uuid_generate_v4() PRIMARY KEY,
  student_id          UUID    NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  lesson_id           UUID    NOT NULL REFERENCES public.lessons(id) ON DELETE CASCADE,
  completed           BOOLEAN DEFAULT FALSE,
  score               INTEGER,
  time_spent_seconds  INTEGER DEFAULT 0,
  last_position_seconds INTEGER DEFAULT 0,
  completed_at        TIMESTAMPTZ,
  updated_at          TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (student_id, lesson_id)
);

-- Badges catalogue
CREATE TABLE public.badges (
  id           UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  name         TEXT NOT NULL,
  description  TEXT,
  icon         TEXT NOT NULL,
  xp_required  INTEGER DEFAULT 0,
  badge_type   TEXT NOT NULL
               CHECK (badge_type IN ('achievement', 'course', 'streak', 'special')),
  rarity       TEXT NOT NULL DEFAULT 'common'
               CHECK (rarity IN ('common', 'rare', 'epic', 'legendary')),
  created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Student ↔ badge junction table
CREATE TABLE public.student_badges (
  student_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  badge_id   UUID NOT NULL REFERENCES public.badges(id) ON DELETE CASCADE,
  earned_at  TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (student_id, badge_id)
);

-- AI chat history (one row per exchange for audit / analytics)
CREATE TABLE public.chat_messages (
  id         UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  student_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  message    TEXT NOT NULL,
  response   TEXT NOT NULL,
  language   TEXT DEFAULT 'en',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Subscription records (Stripe integration)
CREATE TABLE public.subscriptions (
  id                     UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  student_id             UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  plan                   TEXT NOT NULL CHECK (plan IN ('monthly', 'yearly')),
  status                 TEXT NOT NULL
                         CHECK (status IN ('active', 'cancelled', 'expired', 'trialing')),
  start_date             TIMESTAMPTZ NOT NULL,
  end_date               TIMESTAMPTZ NOT NULL,
  stripe_subscription_id TEXT,
  stripe_customer_id     TEXT,
  created_at             TIMESTAMPTZ DEFAULT NOW()
);

-- Course completion certificates
CREATE TABLE public.certificates (
  id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  student_id      UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  course_id       UUID NOT NULL REFERENCES public.courses(id) ON DELETE CASCADE,
  issued_at       TIMESTAMPTZ DEFAULT NOW(),
  certificate_url TEXT,
  UNIQUE (student_id, course_id)
);

-- ── Indexes ───────────────────────────────────────────────────────────────────

-- Frequently filtered / joined columns.
CREATE INDEX idx_courses_category        ON public.courses (category);
CREATE INDEX idx_courses_published       ON public.courses (published);
CREATE INDEX idx_modules_course_id       ON public.modules (course_id, order_index);
CREATE INDEX idx_lessons_module_id       ON public.lessons (module_id, order_index);
CREATE INDEX idx_enrollments_student_id  ON public.enrollments (student_id);
CREATE INDEX idx_enrollments_course_id   ON public.enrollments (course_id);
CREATE INDEX idx_lesson_progress_student ON public.lesson_progress (student_id);
CREATE INDEX idx_lesson_progress_lesson  ON public.lesson_progress (lesson_id);
CREATE INDEX idx_student_badges_student  ON public.student_badges (student_id);
CREATE INDEX idx_chat_messages_student   ON public.chat_messages (student_id, created_at DESC);

-- ── Row Level Security ────────────────────────────────────────────────────────

ALTER TABLE public.profiles       ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.courses        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.modules        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lessons        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.enrollments    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lesson_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.badges         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.student_badges ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.subscriptions  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.certificates   ENABLE ROW LEVEL SECURITY;

-- ── RLS Policies — profiles ───────────────────────────────────────────────────

CREATE POLICY "Users can view their own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

-- Admins can view all profiles (e.g. for the admin dashboard).
CREATE POLICY "Admins can view all profiles"
  ON public.profiles FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE id = auth.uid() AND role = 'admin'
    )
  );

-- ── RLS Policies — courses ────────────────────────────────────────────────────

CREATE POLICY "Anyone can view published courses"
  ON public.courses FOR SELECT
  USING (published = TRUE);

CREATE POLICY "Admins and instructors can manage courses"
  ON public.courses FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE id = auth.uid() AND role IN ('admin', 'instructor')
    )
  );

-- ── RLS Policies — modules ────────────────────────────────────────────────────

CREATE POLICY "Anyone can view modules of published courses"
  ON public.modules FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.courses
      WHERE id = course_id AND published = TRUE
    )
  );

CREATE POLICY "Admins and instructors can manage modules"
  ON public.modules FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE id = auth.uid() AND role IN ('admin', 'instructor')
    )
  );

-- ── RLS Policies — lessons ────────────────────────────────────────────────────

CREATE POLICY "Anyone can view lessons of published courses"
  ON public.lessons FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.modules m
      JOIN public.courses c ON c.id = m.course_id
      WHERE m.id = module_id AND c.published = TRUE
    )
  );

CREATE POLICY "Admins and instructors can manage lessons"
  ON public.lessons FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE id = auth.uid() AND role IN ('admin', 'instructor')
    )
  );

-- ── RLS Policies — enrollments ────────────────────────────────────────────────

CREATE POLICY "Students manage their own enrollments"
  ON public.enrollments FOR ALL
  USING (auth.uid() = student_id);

CREATE POLICY "Admins can view all enrollments"
  ON public.enrollments FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE id = auth.uid() AND role = 'admin'
    )
  );

-- ── RLS Policies — lesson_progress ───────────────────────────────────────────

CREATE POLICY "Students manage their own progress"
  ON public.lesson_progress FOR ALL
  USING (auth.uid() = student_id);

-- ── RLS Policies — badges ─────────────────────────────────────────────────────

CREATE POLICY "Everyone can view badges"
  ON public.badges FOR SELECT
  USING (TRUE);

-- ── RLS Policies — student_badges ────────────────────────────────────────────

CREATE POLICY "Students view their own badges"
  ON public.student_badges FOR SELECT
  USING (auth.uid() = student_id);

CREATE POLICY "System can insert student badges"
  ON public.student_badges FOR INSERT
  WITH CHECK (auth.uid() = student_id);

-- ── RLS Policies — chat_messages ─────────────────────────────────────────────

CREATE POLICY "Students view their own chat history"
  ON public.chat_messages FOR ALL
  USING (auth.uid() = student_id);

-- ── RLS Policies — subscriptions ─────────────────────────────────────────────

CREATE POLICY "Students view their own subscription"
  ON public.subscriptions FOR SELECT
  USING (auth.uid() = student_id);

CREATE POLICY "Admins can manage all subscriptions"
  ON public.subscriptions FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE id = auth.uid() AND role = 'admin'
    )
  );

-- ── RLS Policies — certificates ──────────────────────────────────────────────

CREATE POLICY "Students view their own certificates"
  ON public.certificates FOR SELECT
  USING (auth.uid() = student_id);

-- ── Seed Data — badges ────────────────────────────────────────────────────────

INSERT INTO public.badges (name, description, icon, xp_required, badge_type, rarity) VALUES
  ('First Step',      'Complete your first lesson',             '🎯',  0,      'achievement', 'common'),
  ('Quick Learner',   'Complete 5 lessons in one day',          '⚡',  0,      'achievement', 'rare'),
  ('Week Warrior',    'Maintain a 7-day learning streak',       '🔥',  0,      'streak',      'rare'),
  ('Month Master',    'Maintain a 30-day learning streak',      '🌟',  0,      'streak',      'epic'),
  ('Rising Star',     'Reach Level 5',                          '⭐',  1000,   'achievement', 'rare'),
  ('Code Master',     'Complete a programming course',          '💻',  0,      'course',      'rare'),
  ('Perfect Score',   'Get 100% on any quiz',                   '🏆',  0,      'achievement', 'epic'),
  ('Social Learner',  'Join 3 different courses',               '🤝',  0,      'achievement', 'common'),
  ('XP Hunter',       'Earn 5 000 XP total',                    '💎',  5000,   'achievement', 'epic'),
  ('Elite Learner',   'Reach Level 10',                         '👑',  12000,  'achievement', 'legendary');

-- ── Functions & Triggers ──────────────────────────────────────────────────────

-- Auto-create a profiles row when a new auth.users record is inserted.
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  INSERT INTO public.profiles (id, email, name, role)
  VALUES (
    NEW.id,
    NEW.email,
    COALESCE(NEW.raw_user_meta_data->>'name', split_part(NEW.email, '@', 1)),
    COALESCE(NEW.raw_user_meta_data->>'role', 'student')
  )
  ON CONFLICT (id) DO NOTHING; -- safe to re-run
  RETURN NEW;
END;
$$;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_new_user();

-- Keep courses.total_students in sync whenever an enrollment is created.
CREATE OR REPLACE FUNCTION public.update_course_student_count()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  UPDATE public.courses
  SET total_students = (
    SELECT COUNT(*) FROM public.enrollments WHERE course_id = NEW.course_id
  )
  WHERE id = NEW.course_id;
  RETURN NEW;
END;
$$;

CREATE TRIGGER on_enrollment_created
  AFTER INSERT ON public.enrollments
  FOR EACH ROW
  EXECUTE FUNCTION public.update_course_student_count();

-- Automatically refresh courses.updated_at on any UPDATE.
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

CREATE TRIGGER courses_set_updated_at
  BEFORE UPDATE ON public.courses
  FOR EACH ROW
  EXECUTE FUNCTION public.set_updated_at();
