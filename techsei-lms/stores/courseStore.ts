import { create } from 'zustand';
import { supabase } from '../lib/supabase';
import type { Course, Lesson, LessonProgress } from '../types';

// ─── State & Actions ─────────────────────────────────────────────────────────

interface CourseState {
  courses: Course[];
  enrolledCourses: Course[];
  currentCourse: Course | null;
  currentLesson: Lesson | null;
  /** Keyed by lesson_id. Populated by fetchCourseProgress. */
  lessonProgress: Record<string, LessonProgress>;
  isLoading: boolean;
  error: string | null;
}

interface CourseActions {
  fetchCourses: (category?: string) => Promise<void>;
  fetchEnrolledCourses: (studentId: string) => Promise<void>;
  enrollInCourse: (courseId: string, studentId: string) => Promise<void>;
  markLessonComplete: (lessonId: string, studentId: string, score?: number) => Promise<void>;
  fetchCourseProgress: (courseId: string, studentId: string) => Promise<void>;
  setCurrentCourse: (course: Course | null) => void;
  setCurrentLesson: (lesson: Lesson | null) => void;
  /** Returns completion percentage (0–100) for the given course. */
  getCourseProgress: (courseId: string) => number;
  /** Looks up a lesson by id from the current course's modules or currentLesson. */
  getLessonById: (lessonId: string) => Lesson | undefined;
}

type CourseStore = CourseState & CourseActions;

// ─── Store ───────────────────────────────────────────────────────────────────

export const useCourseStore = create<CourseStore>()((set, get) => ({
  // ── Initial state ──────────────────────────────────────────────────────────
  courses: [],
  enrolledCourses: [],
  currentCourse: null,
  currentLesson: null,
  lessonProgress: {},
  isLoading: false,
  error: null,

  // ── Actions ────────────────────────────────────────────────────────────────

  /**
   * Fetch all published courses, optionally filtered by category.
   * Pass "all" or omit category to load every published course.
   */
  fetchCourses: async (category?: string) => {
    set({ isLoading: true, error: null });
    try {
      let query = supabase
        .from('courses')
        .select('*')
        .eq('published', true)
        .order('created_at', { ascending: false });

      if (category && category !== 'all') {
        query = query.eq('category', category);
      }

      const { data, error } = await query;
      if (error) throw error;

      set({ courses: (data ?? []) as Course[], isLoading: false });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to fetch courses';
      set({ error: message, isLoading: false });
      throw err;
    }
  },

  /**
   * Fetch all courses the given student is actively enrolled in.
   * Uses a join on the enrollments → courses relationship.
   */
  fetchEnrolledCourses: async (studentId: string) => {
    set({ isLoading: true, error: null });
    try {
      const { data, error } = await supabase
        .from('enrollments')
        .select(`
          course_id,
          status,
          enrolled_at,
          courses (
            id, title, description, category, thumbnail_url,
            is_premium, instructor_name, rating, total_students,
            duration_hours, published, created_by, created_at
          )
        `)
        .eq('student_id', studentId)
        .eq('status', 'active');

      if (error) throw error;

      const enrolled: Course[] = (data ?? [])
        .map((row: Record<string, unknown>) => row.courses as Course)
        .filter(Boolean);

      set({ enrolledCourses: enrolled, isLoading: false });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to fetch enrolled courses';
      set({ error: message, isLoading: false });
      throw err;
    }
  },

  /**
   * Enroll a student in a course. Idempotent — re-enrolling an existing row
   * simply sets the status back to "active". Optimistically updates local state.
   */
  enrollInCourse: async (courseId: string, studentId: string) => {
    set({ isLoading: true, error: null });
    try {
      const { error } = await supabase
        .from('enrollments')
        .upsert(
          { student_id: studentId, course_id: courseId, status: 'active' },
          { onConflict: 'student_id,course_id' },
        );

      if (error) throw error;

      // Optimistic local update: add the course if we already have it in the
      // catalogue and it's not already in enrolledCourses.
      const course = get().courses.find((c) => c.id === courseId);
      if (course) {
        set((state) => ({
          enrolledCourses: state.enrolledCourses.some((c) => c.id === courseId)
            ? state.enrolledCourses
            : [...state.enrolledCourses, course],
          isLoading: false,
        }));
      } else {
        set({ isLoading: false });
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to enroll in course';
      set({ error: message, isLoading: false });
      throw err;
    }
  },

  /**
   * Mark a lesson as complete for the given student. An optional quiz score can
   * be supplied (0–100). Updates local lessonProgress map optimistically after
   * the upsert succeeds.
   */
  markLessonComplete: async (lessonId: string, studentId: string, score?: number) => {
    try {
      const now = new Date().toISOString();
      const upsertData: Record<string, unknown> = {
        student_id: studentId,
        lesson_id: lessonId,
        completed: true,
        completed_at: now,
        updated_at: now,
      };
      if (score !== undefined) {
        upsertData.score = score;
      }

      const { data, error } = await supabase
        .from('lesson_progress')
        .upsert(upsertData, { onConflict: 'student_id,lesson_id' })
        .select('*')
        .single();

      if (error) throw error;

      set((state) => ({
        lessonProgress: {
          ...state.lessonProgress,
          [lessonId]: data as LessonProgress,
        },
      }));
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to mark lesson complete';
      set({ error: message });
      throw err;
    }
  },

  /**
   * Load all lesson progress rows for every lesson in the specified course.
   * Merges results into the lessonProgress map (preserves existing entries for
   * other courses).
   */
  fetchCourseProgress: async (courseId: string, studentId: string) => {
    try {
      // Resolve all lesson IDs that belong to this course via its modules.
      const { data: modules, error: modulesError } = await supabase
        .from('modules')
        .select('id, lessons(id)')
        .eq('course_id', courseId);

      if (modulesError) throw modulesError;

      const lessonIds: string[] = (modules ?? []).flatMap(
        (m: Record<string, unknown>) =>
          ((m.lessons as Array<{ id: string }>) ?? []).map((l) => l.id),
      );

      if (lessonIds.length === 0) return;

      // Fetch progress rows for all lessons in this course.
      const { data: progressRows, error: progressError } = await supabase
        .from('lesson_progress')
        .select('*')
        .eq('student_id', studentId)
        .in('lesson_id', lessonIds);

      if (progressError) throw progressError;

      const progressMap: Record<string, LessonProgress> = {};
      for (const row of progressRows ?? []) {
        const progress = row as LessonProgress;
        progressMap[progress.lesson_id] = progress;
      }

      set((state) => ({
        lessonProgress: { ...state.lessonProgress, ...progressMap },
      }));
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to fetch course progress';
      set({ error: message });
      throw err;
    }
  },

  setCurrentCourse: (course) => set({ currentCourse: course }),

  setCurrentLesson: (lesson) => set({ currentLesson: lesson }),

  // ── Selector methods ────────────────────────────────────────────────────────

  /**
   * Returns the completion percentage (0–100) for a given course based on the
   * lessonProgress entries loaded by fetchCourseProgress.
   *
   * NOTE: accuracy depends on fetchCourseProgress having been called first so
   * that lessonProgress contains all lessons for this course (not just touched
   * ones). In practice callers should always load progress before rendering a
   * progress bar.
   */
  getCourseProgress: (courseId: string): number => {
    const { lessonProgress, enrolledCourses } = get();

    const isEnrolled = enrolledCourses.some((c) => c.id === courseId);
    if (!isEnrolled) return 0;

    const allProgress = Object.values(lessonProgress);
    if (allProgress.length === 0) return 0;

    const completedCount = allProgress.filter((p) => p.completed).length;
    return Math.round((completedCount / allProgress.length) * 100);
  },

  /**
   * Finds a lesson by id. First checks currentLesson, then walks the
   * currentCourse module tree if available.
   */
  getLessonById: (lessonId: string): Lesson | undefined => {
    const { currentLesson, currentCourse } = get();

    if (currentLesson?.id === lessonId) return currentLesson;

    if (currentCourse?.modules) {
      for (const mod of currentCourse.modules) {
        const found = mod.lessons?.find((l) => l.id === lessonId);
        if (found) return found;
      }
    }

    return undefined;
  },
}));

// ─── Typed Selectors ──────────────────────────────────────────────────────────

export const selectCourses = (state: CourseStore) => state.courses;
export const selectEnrolledCourses = (state: CourseStore) => state.enrolledCourses;
export const selectCurrentCourse = (state: CourseStore) => state.currentCourse;
export const selectCurrentLesson = (state: CourseStore) => state.currentLesson;
export const selectLessonProgress = (state: CourseStore) => state.lessonProgress;
export const selectIsLoading = (state: CourseStore) => state.isLoading;
export const selectError = (state: CourseStore) => state.error;
