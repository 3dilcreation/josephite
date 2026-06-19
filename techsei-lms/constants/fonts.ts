import { TextStyle } from 'react-native';

// ============================================================
// TechSei LMS — Typography System
// ============================================================
// Centralises every font family name and every reusable text
// style so that screens and components never hard-code sizes.
//
// Font loading:
//   Load the fonts below via expo-font in your root _layout.tsx:
//
//   const [fontsLoaded] = useFonts({
//     [Fonts.regular]:  require('../assets/fonts/Inter-Regular.ttf'),
//     [Fonts.medium]:   require('../assets/fonts/Inter-Medium.ttf'),
//     [Fonts.semiBold]: require('../assets/fonts/Inter-SemiBold.ttf'),
//     [Fonts.bold]:     require('../assets/fonts/Inter-Bold.ttf'),
//     [Fonts.extraBold]:require('../assets/fonts/Inter-ExtraBold.ttf'),
//     [Fonts.mono]:     require('../assets/fonts/JetBrainsMono-Regular.ttf'),
//     [Fonts.monoBold]: require('../assets/fonts/JetBrainsMono-Bold.ttf'),
//   });

// ── Font Family Names ────────────────────────────────────────────────────────

/**
 * Font family string constants.  These must match the keys passed to
 * `useFonts()` exactly.
 */
export const Fonts = {
  /** Inter 400 — body copy, captions, input text. */
  regular: 'Inter-Regular',
  /** Inter 500 — labels, secondary headings, button text. */
  medium: 'Inter-Medium',
  /** Inter 600 — sub-headings, emphasized UI text. */
  semiBold: 'Inter-SemiBold',
  /** Inter 700 — headings, tab labels, badge text. */
  bold: 'Inter-Bold',
  /** Inter 800 — hero numbers, XP counters, large display text. */
  extraBold: 'Inter-ExtraBold',
  /** JetBrains Mono 400 — code snippets, lesson content (quiz answers). */
  mono: 'JetBrainsMono-Regular',
  /** JetBrains Mono 700 — highlighted code, inline code in headings. */
  monoBold: 'JetBrainsMono-Bold',
} as const;

export type FontKey = keyof typeof Fonts;
export type FontFamily = (typeof Fonts)[FontKey];

// ── Type Scale ───────────────────────────────────────────────────────────────

/**
 * Numeric font-size scale.  Derived from a 4 pt baseline grid.
 */
export const FontSize = {
  /** 10 — micro labels, timestamps, footnotes. */
  xs: 10,
  /** 12 — captions, helper text, badge text. */
  sm: 12,
  /** 14 — body text, list items, input placeholders. */
  md: 14,
  /** 16 — standard body, card titles. */
  base: 16,
  /** 18 — section sub-headings. */
  lg: 18,
  /** 20 — screen headings, modal titles. */
  xl: 20,
  /** 24 — page titles, large card headers. */
  '2xl': 24,
  /** 28 — hero text, XP display. */
  '3xl': 28,
  /** 32 — onboarding headlines. */
  '4xl': 32,
  /** 40 — large numeric counters (streak, level). */
  '5xl': 40,
} as const;

export type FontSizeKey = keyof typeof FontSize;

// ── Line Height Scale ────────────────────────────────────────────────────────

/**
 * Line-height values paired with the font-size scale for consistent vertical
 * rhythm (approximately 1.4–1.6× the font size).
 */
export const LineHeight = {
  xs: 14,
  sm: 18,
  md: 20,
  base: 24,
  lg: 28,
  xl: 30,
  '2xl': 34,
  '3xl': 40,
  '4xl': 44,
  '5xl': 52,
} as const;

export type LineHeightKey = keyof typeof LineHeight;

// ── Letter Spacing ───────────────────────────────────────────────────────────

export const LetterSpacing = {
  tighter: -0.8,
  tight: -0.4,
  normal: 0,
  wide: 0.4,
  wider: 0.8,
  /** All-caps labels and overlines. */
  widest: 1.5,
} as const;

// ── Reusable Text Styles ─────────────────────────────────────────────────────

/**
 * Pre-composed TextStyle objects.  These map onto a semantic naming scheme
 * (displayXL → smallest text, not component names) so they can be composed
 * with colour and margin overrides at the call site.
 *
 * Usage:
 *   <Text style={[TextStyles.headingLg, { color: Colors.text }]}>Title</Text>
 */
export const TextStyles = {
  // ── Display / Hero ────────────────────────────────────────────────────────
  display: {
    fontFamily: Fonts.extraBold,
    fontSize: FontSize['4xl'],
    lineHeight: LineHeight['4xl'],
    letterSpacing: LetterSpacing.tight,
  } satisfies TextStyle,

  displaySm: {
    fontFamily: Fonts.extraBold,
    fontSize: FontSize['3xl'],
    lineHeight: LineHeight['3xl'],
    letterSpacing: LetterSpacing.tight,
  } satisfies TextStyle,

  // ── Headings ──────────────────────────────────────────────────────────────
  headingXl: {
    fontFamily: Fonts.bold,
    fontSize: FontSize['2xl'],
    lineHeight: LineHeight['2xl'],
    letterSpacing: LetterSpacing.tight,
  } satisfies TextStyle,

  headingLg: {
    fontFamily: Fonts.bold,
    fontSize: FontSize.xl,
    lineHeight: LineHeight.xl,
    letterSpacing: LetterSpacing.tight,
  } satisfies TextStyle,

  headingMd: {
    fontFamily: Fonts.semiBold,
    fontSize: FontSize.lg,
    lineHeight: LineHeight.lg,
    letterSpacing: LetterSpacing.normal,
  } satisfies TextStyle,

  headingSm: {
    fontFamily: Fonts.semiBold,
    fontSize: FontSize.base,
    lineHeight: LineHeight.base,
    letterSpacing: LetterSpacing.normal,
  } satisfies TextStyle,

  // ── Body ──────────────────────────────────────────────────────────────────
  bodyLg: {
    fontFamily: Fonts.regular,
    fontSize: FontSize.base,
    lineHeight: LineHeight.base,
    letterSpacing: LetterSpacing.normal,
  } satisfies TextStyle,

  body: {
    fontFamily: Fonts.regular,
    fontSize: FontSize.md,
    lineHeight: LineHeight.md,
    letterSpacing: LetterSpacing.normal,
  } satisfies TextStyle,

  bodySm: {
    fontFamily: Fonts.regular,
    fontSize: FontSize.sm,
    lineHeight: LineHeight.sm,
    letterSpacing: LetterSpacing.normal,
  } satisfies TextStyle,

  // ── Labels & UI ───────────────────────────────────────────────────────────
  labelLg: {
    fontFamily: Fonts.medium,
    fontSize: FontSize.base,
    lineHeight: LineHeight.base,
    letterSpacing: LetterSpacing.wide,
  } satisfies TextStyle,

  label: {
    fontFamily: Fonts.medium,
    fontSize: FontSize.md,
    lineHeight: LineHeight.md,
    letterSpacing: LetterSpacing.wide,
  } satisfies TextStyle,

  labelSm: {
    fontFamily: Fonts.medium,
    fontSize: FontSize.sm,
    lineHeight: LineHeight.sm,
    letterSpacing: LetterSpacing.wider,
  } satisfies TextStyle,

  /** All-caps overline text — section category labels. */
  overline: {
    fontFamily: Fonts.semiBold,
    fontSize: FontSize.xs,
    lineHeight: LineHeight.xs,
    letterSpacing: LetterSpacing.widest,
    textTransform: 'uppercase',
  } satisfies TextStyle,

  // ── Captions & Meta ───────────────────────────────────────────────────────
  caption: {
    fontFamily: Fonts.regular,
    fontSize: FontSize.xs,
    lineHeight: LineHeight.xs,
    letterSpacing: LetterSpacing.normal,
  } satisfies TextStyle,

  // ── Buttons ───────────────────────────────────────────────────────────────
  buttonLg: {
    fontFamily: Fonts.semiBold,
    fontSize: FontSize.base,
    lineHeight: LineHeight.base,
    letterSpacing: LetterSpacing.wide,
  } satisfies TextStyle,

  button: {
    fontFamily: Fonts.semiBold,
    fontSize: FontSize.md,
    lineHeight: LineHeight.md,
    letterSpacing: LetterSpacing.wide,
  } satisfies TextStyle,

  buttonSm: {
    fontFamily: Fonts.medium,
    fontSize: FontSize.sm,
    lineHeight: LineHeight.sm,
    letterSpacing: LetterSpacing.wide,
  } satisfies TextStyle,

  // ── Numeric / Stat Display ────────────────────────────────────────────────
  /** Large XP / streak / level counter — gamification UI. */
  statDisplay: {
    fontFamily: Fonts.extraBold,
    fontSize: FontSize['5xl'],
    lineHeight: LineHeight['5xl'],
    letterSpacing: LetterSpacing.tighter,
  } satisfies TextStyle,

  statLg: {
    fontFamily: Fonts.bold,
    fontSize: FontSize['2xl'],
    lineHeight: LineHeight['2xl'],
    letterSpacing: LetterSpacing.tight,
  } satisfies TextStyle,

  // ── Code ─────────────────────────────────────────────────────────────────
  code: {
    fontFamily: Fonts.mono,
    fontSize: FontSize.sm,
    lineHeight: LineHeight.sm,
    letterSpacing: LetterSpacing.normal,
  } satisfies TextStyle,

  codeLg: {
    fontFamily: Fonts.mono,
    fontSize: FontSize.md,
    lineHeight: LineHeight.md,
    letterSpacing: LetterSpacing.normal,
  } satisfies TextStyle,
} as const;

export type TextStyleKey = keyof typeof TextStyles;

// ── Font Map (for expo-font useFonts) ────────────────────────────────────────

/**
 * Ready-to-spread object for `useFonts()`.  Fill in the require() paths that
 * match your assets/fonts/ directory layout.
 *
 * Example in _layout.tsx:
 *   const [fontsLoaded] = useFonts(FONT_MAP);
 */
export const FONT_MAP = {
  [Fonts.regular]:   require('../assets/fonts/Inter-Regular.ttf'),
  [Fonts.medium]:    require('../assets/fonts/Inter-Medium.ttf'),
  [Fonts.semiBold]:  require('../assets/fonts/Inter-SemiBold.ttf'),
  [Fonts.bold]:      require('../assets/fonts/Inter-Bold.ttf'),
  [Fonts.extraBold]: require('../assets/fonts/Inter-ExtraBold.ttf'),
  [Fonts.mono]:      require('../assets/fonts/JetBrainsMono-Regular.ttf'),
  [Fonts.monoBold]:  require('../assets/fonts/JetBrainsMono-Bold.ttf'),
} as const;
