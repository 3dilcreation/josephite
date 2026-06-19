// ============================================================
// TechSei LMS — Brand Color Palette
// ============================================================
// All colours are defined here and nowhere else.  Import this
// file whenever a component, style, or theme needs a colour.

export const Colors = {
  // ── Brand ────────────────────────────────────────────────────────────────
  /** Primary brand purple — buttons, active tabs, links. */
  primary: '#6C63FF',
  /** Coral/pink accent — secondary CTAs, highlights. */
  secondary: '#FF6584',
  /** Electric green — XP gains, success states. */
  accent: '#43E97B',
  /** Warm orange — streak counters, warnings. */
  warning: '#FFB84C',

  // ── Backgrounds ──────────────────────────────────────────────────────────
  /** Deep dark blue-black — root app background. */
  background: '#0A0A1A',
  /** Dark card background — modals, sheets, most cards. */
  surface: '#141428',
  /** Slightly lighter surface — nested cards, input fields. */
  surfaceLight: '#1E1E3A',
  /** Subtle border/divider colour. */
  border: '#2A2A4A',

  // ── Text ─────────────────────────────────────────────────────────────────
  /** Primary text — headings, body copy on dark backgrounds. */
  text: '#FFFFFF',
  /** Secondary text — labels, metadata, captions. */
  textSecondary: '#9494B8',
  /** Muted/disabled text — placeholders, less-important info. */
  textMuted: '#5A5A7A',

  // ── Semantic ─────────────────────────────────────────────────────────────
  /** Success / completed state (same hue as accent). */
  success: '#43E97B',
  /** Error / destructive actions. */
  error: '#FF4B4B',
  /** Gold — badges, achievements, top-rank indicators. */
  gold: '#FFD700',

  // ── Overlay ──────────────────────────────────────────────────────────────
  /** Semi-transparent dark scrim for modals/bottom-sheets. */
  overlay: 'rgba(0, 0, 0, 0.6)',
  /** Faint highlight used for pressed/hovered states. */
  highlight: 'rgba(108, 99, 255, 0.15)',

  // ── Gradients ────────────────────────────────────────────────────────────
  /**
   * Two-stop gradient arrays compatible with expo-linear-gradient's `colors`
   * prop and React Native's experimental LinearGradient.
   */
  gradients: {
    /** Primary brand gradient — hero sections, CTA buttons. */
    primary: ['#6C63FF', '#8B5CF6'] as const,
    /** Secondary/coral gradient — badges, premium indicators. */
    secondary: ['#FF6584', '#FF4B4B'] as const,
    /** Success gradient — XP animations, completion banners. */
    success: ['#43E97B', '#38F9D7'] as const,
    /** Streak / fire gradient — streak counters, daily goal. */
    streak: ['#FFB84C', '#FF6B35'] as const,
    /** Dark gradient — full-screen backgrounds, onboarding. */
    dark: ['#0A0A1A', '#141428'] as const,
    /** Card gradient — elevated card backgrounds. */
    card: ['#141428', '#1E1E3A'] as const,
    /** Purple-to-pink banner gradient. */
    banner: ['#6C63FF', '#FF6584'] as const,
    /** Gold gradient — achievement / certificate accents. */
    gold: ['#FFD700', '#FFA500'] as const,
  },
} as const;

// ── Type helpers ──────────────────────────────────────────────────────────────

export type ColorKey = keyof Omit<typeof Colors, 'gradients'>;
export type GradientKey = keyof typeof Colors.gradients;

/** Convenience re-export so consumers can do `import { Colors, getGradient } from './colors'` */
export function getGradient(key: GradientKey): readonly [string, string] {
  return Colors.gradients[key];
}
