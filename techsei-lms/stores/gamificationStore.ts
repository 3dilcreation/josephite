import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Notifications from 'expo-notifications';
import { supabase } from '../lib/supabase';
import type { StudentBadge } from '../types';

// ─── Constants ────────────────────────────────────────────────────────────────

export type XPReason =
  | 'complete_lesson'
  | 'perfect_quiz'
  | 'daily_streak'
  | 'first_course'
  | 'course_complete';

/**
 * Minimum cumulative XP required to reach each level.
 * Index 0 → Level 1 (0 XP), index 9 → Level 10 (12 000 XP).
 */
export const LEVEL_THRESHOLDS: readonly number[] = [
  0,      // level 1
  100,    // level 2
  250,    // level 3
  500,    // level 4
  1_000,  // level 5
  2_000,  // level 6
  3_500,  // level 7
  5_500,  // level 8
  8_000,  // level 9
  12_000, // level 10
] as const;

/** Flat XP awarded for each event type. */
export const XP_REWARDS: Record<XPReason, number> = {
  complete_lesson: 50,
  perfect_quiz: 100,
  daily_streak: 25,
  first_course: 200,
  course_complete: 500,
} as const;

// ─── State & Actions ─────────────────────────────────────────────────────────

interface GamificationState {
  xp: number;
  level: number;
  streak: number;
  badges: StudentBadge[];
  dailyGoalCompleted: boolean;
  lastStreakDate: string | null;
}

interface GamificationActions {
  addXP: (amount: number, reason: XPReason) => Promise<void>;
  checkAndUpdateStreak: () => Promise<void>;
  earnBadge: (badgeId: string) => Promise<void>;
  loadGamification: (userId: string) => Promise<void>;
  calculateLevel: (xp: number) => number;
}

type GamificationStore = GamificationState & GamificationActions;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function todayISO(): string {
  return new Date().toISOString().slice(0, 10); // "YYYY-MM-DD"
}

function yesterdayISO(): string {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return d.toISOString().slice(0, 10);
}

async function scheduleNotification(title: string, body: string): Promise<void> {
  try {
    await Notifications.scheduleNotificationAsync({
      content: { title, body, sound: true },
      trigger: null, // fire immediately
    });
  } catch {
    // Notifications may not be permitted — fail silently.
  }
}

// ─── Store ───────────────────────────────────────────────────────────────────

export const useGamificationStore = create<GamificationStore>()(
  persist(
    (set, get) => ({
      // ── Initial state ──────────────────────────────────────────────────────
      xp: 0,
      level: 1,
      streak: 0,
      badges: [],
      dailyGoalCompleted: false,
      lastStreakDate: null,

      // ── Actions ────────────────────────────────────────────────────────────

      /**
       * Pure, synchronous level calculation. Returns the highest level whose
       * XP threshold the learner has met or exceeded.
       */
      calculateLevel: (xp: number): number => {
        let level = 1;
        for (let i = LEVEL_THRESHOLDS.length - 1; i >= 0; i--) {
          if (xp >= LEVEL_THRESHOLDS[i]) {
            level = i + 1;
            break;
          }
        }
        return level;
      },

      /**
       * Award XP and recalculate the learner's level. If the learner crosses a
       * level threshold a push notification is dispatched. Changes are persisted
       * to Supabase asynchronously (fire-and-forget with error logging).
       */
      addXP: async (amount: number, reason: XPReason) => {
        const { xp, level, calculateLevel } = get();

        // `amount` is the explicit XP to award; fall back to the lookup table.
        const xpReward = amount > 0 ? amount : XP_REWARDS[reason] ?? 0;
        const newXP = xp + xpReward;
        const newLevel = calculateLevel(newXP);
        const leveledUp = newLevel > level;

        set({ xp: newXP, level: newLevel });

        // Persist to Supabase (fire-and-forget).
        const { data: { session } } = await supabase.auth.getSession();
        if (session?.user) {
          supabase
            .from('profiles')
            .update({ xp: newXP, level: newLevel })
            .eq('id', session.user.id)
            .then(({ error }) => {
              if (error) console.warn('[gamificationStore] XP sync error:', error.message);
            });
        }

        if (leveledUp) {
          await scheduleNotification(
            '🎉 Level Up!',
            `Congratulations! You've reached Level ${newLevel}!`,
          );
        }
      },

      /**
       * Must be called whenever the learner completes an activity. Handles:
       * - First-ever activity (streak = 1)
       * - Consecutive days (streak++)
       * - Broken streak (reset to 1)
       * - Already recorded today (no-op)
       */
      checkAndUpdateStreak: async () => {
        const { streak, lastStreakDate } = get();
        const today = todayISO();
        const yesterday = yesterdayISO();

        if (lastStreakDate === today) {
          // Already counted today — nothing to do.
          return;
        }

        const newStreak =
          lastStreakDate === yesterday
            ? streak + 1  // Consecutive day — extend streak
            : 1;           // First activity or broken streak

        set({ streak: newStreak, lastStreakDate: today });

        // Award daily streak XP (uses internal addXP so level logic runs too).
        await get().addXP(XP_REWARDS.daily_streak, 'daily_streak');

        // Persist streak + last_activity to Supabase.
        const { data: { session } } = await supabase.auth.getSession();
        if (session?.user) {
          supabase
            .from('profiles')
            .update({ streak: newStreak, last_activity: today })
            .eq('id', session.user.id)
            .then(({ error }) => {
              if (error) console.warn('[gamificationStore] Streak sync error:', error.message);
            });
        }
      },

      /**
       * Award a badge to the current learner. Idempotent — duplicate badge_ids
       * are silently ignored. Fetches badge metadata from Supabase and dispatches
       * a push notification.
       */
      earnBadge: async (badgeId: string) => {
        const { badges } = get();

        // Prevent duplicates.
        if (badges.some((b) => b.badge_id === badgeId)) return;

        const { data: { session } } = await supabase.auth.getSession();
        if (!session?.user) return;

        // Fetch badge metadata.
        const { data: badgeData, error: badgeError } = await supabase
          .from('badges')
          .select('id, name, icon, description, badge_type, xp_required')
          .eq('id', badgeId)
          .single();

        if (badgeError) {
          console.warn('[gamificationStore] Badge fetch error:', badgeError.message);
          return;
        }

        // Upsert into student_badges junction table.
        const { error: upsertError } = await supabase
          .from('student_badges')
          .upsert(
            { student_id: session.user.id, badge_id: badgeId },
            { onConflict: 'student_id,badge_id' },
          );

        if (upsertError) {
          console.warn('[gamificationStore] Badge upsert error:', upsertError.message);
          return;
        }

        const newStudentBadge: StudentBadge = {
          student_id: session.user.id,
          badge_id: badgeId,
          earned_at: new Date().toISOString(),
          badge: badgeData as StudentBadge['badge'],
        };

        set((state) => ({ badges: [...state.badges, newStudentBadge] }));

        // Dispatch badge-earned notification.
        const badge = badgeData as { icon: string; name: string };
        await scheduleNotification(
          `${badge.icon} Badge Earned!`,
          `You earned the "${badge.name}" badge!`,
        );
      },

      /**
       * Full remote load of the learner's gamification state. Call after sign-in
       * or when mounting a screen that requires fresh data.
       */
      loadGamification: async (userId: string) => {
        try {
          // Load XP / level / streak from the profiles table.
          const { data: profile, error: profileError } = await supabase
            .from('profiles')
            .select('xp, level, streak, last_activity')
            .eq('id', userId)
            .single();

          if (profileError) throw profileError;

          // Load badges with full badge metadata via a Supabase join.
          const { data: studentBadges, error: badgesError } = await supabase
            .from('student_badges')
            .select(
              'student_id, badge_id, earned_at, badge:badges(id, name, icon, description, badge_type, xp_required)',
            )
            .eq('student_id', userId);

          if (badgesError) throw badgesError;

          const p = profile as Record<string, unknown>;
          const today = todayISO();
          const lastActivity = (p.last_activity as string) ?? null;

          set({
            xp: (p.xp as number) ?? 0,
            level: (p.level as number) ?? 1,
            streak: (p.streak as number) ?? 0,
            lastStreakDate: lastActivity,
            dailyGoalCompleted: lastActivity === today,
            badges: (studentBadges ?? []) as StudentBadge[],
          });
        } catch (err: unknown) {
          console.warn(
            '[gamificationStore] Load error:',
            err instanceof Error ? err.message : err,
          );
        }
      },
    }),

    {
      name: 'techsei-gamification-store',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        xp: state.xp,
        level: state.level,
        streak: state.streak,
        badges: state.badges,
        dailyGoalCompleted: state.dailyGoalCompleted,
        lastStreakDate: state.lastStreakDate,
      }),
    },
  ),
);

// ─── Typed Selectors ──────────────────────────────────────────────────────────

/**
 * Returns the learner's XP progress toward the next level as a value in [0, 1].
 * Returns 1 at max level.
 */
export function selectLevelProgress(state: GamificationState): number {
  const { xp, level } = state;
  const currentThreshold = LEVEL_THRESHOLDS[level - 1] ?? 0;
  const nextThreshold = LEVEL_THRESHOLDS[level] ?? LEVEL_THRESHOLDS[LEVEL_THRESHOLDS.length - 1];
  if (nextThreshold === currentThreshold) return 1; // max level
  return Math.min((xp - currentThreshold) / (nextThreshold - currentThreshold), 1);
}

/**
 * Returns the XP still needed to reach the next level. Returns 0 at max level.
 */
export function selectXPToNextLevel(state: GamificationState): number {
  const { xp, level } = state;
  const nextThreshold = LEVEL_THRESHOLDS[level] ?? LEVEL_THRESHOLDS[LEVEL_THRESHOLDS.length - 1];
  return Math.max(nextThreshold - xp, 0);
}

/**
 * Returns true if the learner is at the maximum level.
 */
export function selectIsMaxLevel(state: GamificationState): boolean {
  return state.level >= LEVEL_THRESHOLDS.length;
}
