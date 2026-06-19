import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../lib/supabase';
import type { User, Profile, LanguageCode } from '../types';

// ─── State & Actions ─────────────────────────────────────────────────────────

interface AuthState {
  user: User | null;
  profile: Profile | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  error: string | null;
}

interface AuthActions {
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, name: string, role: 'student' | 'admin') => Promise<void>;
  signOut: () => Promise<void>;
  loadUser: () => Promise<void>;
  updateProfile: (updates: Partial<User & Profile>) => Promise<void>;
  setLanguage: (code: LanguageCode) => Promise<void>;
  clearError: () => void;
}

type AuthStore = AuthState & AuthActions;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Maps a Supabase profiles row into our typed User and Profile shapes.
 * The profiles table is a single flat table that stores both user identity
 * fields and gamification/subscription data.
 */
function mapRowToUserAndProfile(row: Record<string, unknown>): { user: User; profile: Profile } {
  const user: User = {
    id: row.id as string,
    email: row.email as string,
    role: ((row.role as string) === 'admin' ? 'admin' : 'student') as 'student' | 'admin',
    name: (row.name as string) ?? '',
    avatar_url: (row.avatar_url as string) ?? null,
    language_pref: ((row.language_pref as LanguageCode) ?? 'en') as LanguageCode,
    created_at: (row.created_at as string) ?? new Date().toISOString(),
  };

  const profile: Profile = {
    user_id: row.id as string,
    xp: (row.xp as number) ?? 0,
    level: (row.level as number) ?? 1,
    streak: (row.streak as number) ?? 0,
    subscription_tier: ((row.subscription_tier as 'free' | 'pro' | 'premium') ?? 'free'),
    last_activity: (row.last_activity as string) ?? null,
    total_courses_completed: (row.total_courses_completed as number) ?? 0,
    total_lessons_completed: (row.total_lessons_completed as number) ?? 0,
    total_time_spent_seconds: (row.total_time_spent_seconds as number) ?? 0,
  };

  return { user, profile };
}

// ─── Store ───────────────────────────────────────────────────────────────────

export const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      // ── Initial state ──────────────────────────────────────────────────────
      user: null,
      profile: null,
      isLoading: false,
      isAuthenticated: false,
      error: null,

      // ── Actions ────────────────────────────────────────────────────────────

      /**
       * Sign in with email and password. Fetches the profile row from Supabase
       * immediately after a successful auth so the store is fully populated.
       */
      signIn: async (email, password) => {
        set({ isLoading: true, error: null });
        try {
          const { data, error } = await supabase.auth.signInWithPassword({ email, password });
          if (error) throw error;
          if (!data.user) throw new Error('No user returned from sign-in');

          const { data: profileRow, error: profileError } = await supabase
            .from('profiles')
            .select('*')
            .eq('id', data.user.id)
            .single();

          if (profileError) throw profileError;

          const { user, profile } = mapRowToUserAndProfile(profileRow as Record<string, unknown>);
          set({ user, profile, isAuthenticated: true, isLoading: false });
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : 'Sign-in failed';
          set({ error: message, isLoading: false, isAuthenticated: false });
          throw err;
        }
      },

      /**
       * Create a new account. A Supabase trigger (handle_new_user) creates the
       * profiles row automatically; we poll briefly to let it propagate.
       */
      signUp: async (email, password, name, role) => {
        set({ isLoading: true, error: null });
        try {
          const { data, error } = await supabase.auth.signUp({
            email,
            password,
            options: { data: { name, role } },
          });
          if (error) throw error;
          if (!data.user) throw new Error('No user returned from sign-up');

          // Poll until the trigger-created profile row appears (max ~2.5 s).
          let profileRow: Record<string, unknown> | null = null;
          for (let attempt = 0; attempt < 5; attempt++) {
            await new Promise<void>((resolve) => setTimeout(resolve, 500));
            const { data: row, error: profileError } = await supabase
              .from('profiles')
              .select('*')
              .eq('id', data.user.id)
              .single();
            if (!profileError && row) {
              profileRow = row as Record<string, unknown>;
              break;
            }
          }

          if (!profileRow) {
            throw new Error('Profile creation timed out. Please try signing in.');
          }

          const { user, profile } = mapRowToUserAndProfile(profileRow);
          set({ user, profile, isAuthenticated: true, isLoading: false });
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : 'Sign-up failed';
          set({ error: message, isLoading: false, isAuthenticated: false });
          throw err;
        }
      },

      signOut: async () => {
        set({ isLoading: true, error: null });
        try {
          const { error } = await supabase.auth.signOut();
          if (error) throw error;
          set({ user: null, profile: null, isAuthenticated: false, isLoading: false });
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : 'Sign-out failed';
          set({ error: message, isLoading: false });
          throw err;
        }
      },

      /**
       * Re-hydrate user + profile from Supabase session. Call on app launch.
       */
      loadUser: async () => {
        set({ isLoading: true, error: null });
        try {
          const {
            data: { session },
            error: sessionError,
          } = await supabase.auth.getSession();

          if (sessionError) throw sessionError;

          if (!session?.user) {
            set({ user: null, profile: null, isAuthenticated: false, isLoading: false });
            return;
          }

          const { data: profileRow, error: profileError } = await supabase
            .from('profiles')
            .select('*')
            .eq('id', session.user.id)
            .single();

          if (profileError) throw profileError;

          const { user, profile } = mapRowToUserAndProfile(profileRow as Record<string, unknown>);
          set({ user, profile, isAuthenticated: true, isLoading: false });
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : 'Failed to load user';
          set({ error: message, isLoading: false, isAuthenticated: false });
        }
      },

      /**
       * Update any combination of user/profile fields in the profiles table.
       * Returns optimistically after the Supabase upsert succeeds.
       */
      updateProfile: async (updates) => {
        const { user } = get();
        if (!user) throw new Error('Not authenticated');

        set({ isLoading: true, error: null });
        try {
          // Allowed mutable profile fields (guards against inadvertent id overwrite).
          const allowedFields: Array<keyof (User & Profile)> = [
            'name',
            'avatar_url',
            'language_pref',
            'xp',
            'level',
            'streak',
            'subscription_tier',
            'last_activity',
          ];

          const profileUpdates: Record<string, unknown> = {};
          for (const key of allowedFields) {
            if (key in updates) {
              profileUpdates[key] = updates[key as keyof typeof updates];
            }
          }

          const { data: updatedRow, error } = await supabase
            .from('profiles')
            .update(profileUpdates)
            .eq('id', user.id)
            .select('*')
            .single();

          if (error) throw error;

          const { user: updatedUser, profile: updatedProfile } = mapRowToUserAndProfile(
            updatedRow as Record<string, unknown>,
          );
          set({ user: updatedUser, profile: updatedProfile, isLoading: false });
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : 'Profile update failed';
          set({ error: message, isLoading: false });
          throw err;
        }
      },

      setLanguage: async (code) => {
        const { user } = get();
        if (!user) return;

        try {
          const { error } = await supabase
            .from('profiles')
            .update({ language_pref: code })
            .eq('id', user.id);

          if (error) throw error;

          set((state) => ({
            user: state.user ? { ...state.user, language_pref: code } : null,
          }));
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : 'Failed to set language';
          set({ error: message });
          throw err;
        }
      },

      clearError: () => set({ error: null }),
    }),

    {
      name: 'techsei-auth-store',
      storage: createJSONStorage(() => AsyncStorage),
      // Only persist stable user state — not transient flags.
      partialize: (state) => ({
        user: state.user,
        profile: state.profile,
        isAuthenticated: state.isAuthenticated,
      }),
    },
  ),
);

// ─── Auth State Listener ──────────────────────────────────────────────────────

/**
 * Keep the store in sync with Supabase's own auth lifecycle events
 * (token refresh, session expiry, sign-out from another tab, etc.).
 */
supabase.auth.onAuthStateChange(async (event, session) => {
  const store = useAuthStore.getState();

  if (event === 'SIGNED_OUT' || !session) {
    useAuthStore.setState({ user: null, profile: null, isAuthenticated: false });
    return;
  }

  if (event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') {
    // Re-fetch profile only when the signed-in user changed or we have no data.
    if (!store.user || store.user.id !== session.user.id) {
      await store.loadUser();
    }
  }
});
