// ============================================================
// TechSei LMS — Onboarding Screen (3-step new-user flow)
// ============================================================
import React, { useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Animated,
  Dimensions,
  ActivityIndicator,
  Alert,
  Platform,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '../../stores/authStore';
import { supabase } from '../../lib/supabase';
import type { LanguageCode } from '../../types';

// ── Constants ─────────────────────────────────────────────────────────────────

const { width: SCREEN_WIDTH } = Dimensions.get('window');

const COLORS = {
  bg: '#0A0A1A',
  surface: '#141428',
  surfaceHighlight: '#1C1C38',
  primary: '#6C63FF',
  primaryLight: '#8B5CF6',
  primaryDim: '#4a4480',
  accent: '#43E97B',
  accentDim: '#2AAB58',
  border: '#2A2A44',
  borderSelected: '#6C63FF',
  text: '#FFFFFF',
  textMuted: '#7777AA',
  textDim: '#55557A',
  inactive: '#3A3A5A',
};

const TOTAL_STEPS = 3;

// ── Step 1 — Learning goal data ───────────────────────────────────────────────

interface GoalOption {
  id: string;
  label: string;
  icon: React.ComponentProps<typeof Ionicons>['name'];
  color: string;
}

const GOAL_OPTIONS: GoalOption[] = [
  { id: 'web-development', label: 'Web Development', icon: 'globe-outline', color: '#6C63FF' },
  { id: 'data-science', label: 'Data Science', icon: 'bar-chart-outline', color: '#43E97B' },
  { id: 'mobile-apps', label: 'Mobile Apps', icon: 'phone-portrait-outline', color: '#FF6B6B' },
  { id: 'ai-ml', label: 'AI / ML', icon: 'hardware-chip-outline', color: '#FFB547' },
  { id: 'cybersecurity', label: 'Cybersecurity', icon: 'shield-checkmark-outline', color: '#00C9FF' },
  { id: 'cloud-computing', label: 'Cloud Computing', icon: 'cloud-outline', color: '#A78BFA' },
];

// ── Step 2 — Language data ────────────────────────────────────────────────────

interface LanguageOption {
  code: LanguageCode;
  name: string;
  nativeName: string;
  flag: string;
}

const LANGUAGE_OPTIONS: LanguageOption[] = [
  { code: 'en', name: 'English', nativeName: 'English', flag: '🇺🇸' },
  { code: 'hi', name: 'Hindi', nativeName: 'हिन्दी', flag: '🇮🇳' },
  { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்', flag: '🇮🇳' },
  { code: 'te', name: 'Telugu', nativeName: 'తెలుగు', flag: '🇮🇳' },
  { code: 'bn', name: 'Bengali', nativeName: 'বাংলা', flag: '🇧🇩' },
  { code: 'fr', name: 'French', nativeName: 'Français', flag: '🇫🇷' },
  { code: 'es', name: 'Spanish', nativeName: 'Español', flag: '🇪🇸' },
  { code: 'ar', name: 'Arabic', nativeName: 'العربية', flag: '🇸🇦' },
];

// ── Step 3 — Daily goal data ──────────────────────────────────────────────────

interface DailyGoalOption {
  id: string;
  label: string;
  minutes: number;
  emoji: string;
  motivational: string;
}

const DAILY_GOAL_OPTIONS: DailyGoalOption[] = [
  {
    id: '10min',
    label: '10 min / day',
    minutes: 10,
    emoji: '🌱',
    motivational: 'Perfect for getting started. Small steps lead to big changes!',
  },
  {
    id: '20min',
    label: '20 min / day',
    minutes: 20,
    emoji: '🔥',
    motivational: 'Great balance! Consistent daily practice builds lasting skills.',
  },
  {
    id: '30min',
    label: '30 min / day',
    minutes: 30,
    emoji: '⚡',
    motivational: 'Ambitious! You're on the fast track to becoming a pro.',
  },
  {
    id: '60min',
    label: '1 hour / day',
    minutes: 60,
    emoji: '🚀',
    motivational: 'Incredible commitment! You'll be mastering new skills in no time.',
  },
];

// ── Sub-components ────────────────────────────────────────────────────────────

interface StepHeaderProps {
  step: number;
  title: string;
  subtitle: string;
}

function StepHeader({ step, title, subtitle }: StepHeaderProps) {
  return (
    <View style={headerStyles.container}>
      <View style={headerStyles.stepBadge}>
        <Text style={headerStyles.stepBadgeText}>{step} / {TOTAL_STEPS}</Text>
      </View>
      <Text style={headerStyles.title}>{title}</Text>
      <Text style={headerStyles.subtitle}>{subtitle}</Text>
    </View>
  );
}

const headerStyles = StyleSheet.create({
  container: {
    marginBottom: 28,
  },
  stepBadge: {
    alignSelf: 'flex-start',
    backgroundColor: COLORS.surface,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: COLORS.border,
    paddingHorizontal: 12,
    paddingVertical: 4,
    marginBottom: 16,
  },
  stepBadgeText: {
    color: COLORS.textMuted,
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  title: {
    fontSize: 26,
    fontWeight: '800',
    color: COLORS.text,
    marginBottom: 8,
    lineHeight: 32,
  },
  subtitle: {
    fontSize: 15,
    color: COLORS.textMuted,
    lineHeight: 22,
  },
});

// ── Main component ────────────────────────────────────────────────────────────

export default function OnboardingScreen() {
  const router = useRouter();
  const { user, updateProfile, setLanguage } = useAuthStore();

  const [currentStep, setCurrentStep] = useState(0); // 0-indexed
  const [isSaving, setIsSaving] = useState(false);

  // Step selections
  const [selectedGoal, setSelectedGoal] = useState<string | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<LanguageCode>('en');
  const [selectedDailyGoal, setSelectedDailyGoal] = useState<string | null>(null);

  // Slide animation
  const slideAnim = useRef(new Animated.Value(0)).current;
  const fadeAnim = useRef(new Animated.Value(1)).current;

  const animateToStep = useCallback(
    (nextStep: number) => {
      const direction = nextStep > currentStep ? 1 : -1;

      // Fade + slide out
      Animated.parallel([
        Animated.timing(fadeAnim, { toValue: 0, duration: 150, useNativeDriver: true }),
        Animated.timing(slideAnim, {
          toValue: direction * -40,
          duration: 150,
          useNativeDriver: true,
        }),
      ]).start(() => {
        setCurrentStep(nextStep);
        slideAnim.setValue(direction * 40);

        // Fade + slide in
        Animated.parallel([
          Animated.timing(fadeAnim, { toValue: 1, duration: 250, useNativeDriver: true }),
          Animated.timing(slideAnim, { toValue: 0, duration: 250, useNativeDriver: true }),
        ]).start();
      });
    },
    [currentStep, fadeAnim, slideAnim],
  );

  const handleNext = () => {
    if (currentStep < TOTAL_STEPS - 1) {
      animateToStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      animateToStep(currentStep - 1);
    }
  };

  const handleSkip = () => {
    if (currentStep < TOTAL_STEPS - 1) {
      animateToStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handleComplete = async () => {
    if (!user) {
      router.replace('/student/home');
      return;
    }

    setIsSaving(true);
    try {
      // Persist language preference
      await setLanguage(selectedLanguage);

      // Persist learning goal + daily goal minutes to profile via upsert
      const dailyGoalObj = DAILY_GOAL_OPTIONS.find((g) => g.id === selectedDailyGoal);
      const dailyMinutes = dailyGoalObj?.minutes ?? 20;

      await supabase.from('profiles').update({
        language_pref: selectedLanguage,
        daily_goal_minutes: dailyMinutes,
        learning_goal: selectedGoal,
      }).eq('id', user.id);
    } catch (err) {
      // Non-fatal — preferences can be set later in profile
      console.warn('[Onboarding] Failed to save preferences:', err);
    } finally {
      setIsSaving(false);
      router.replace('/student/home');
    }
  };

  // ── Can proceed? ──────────────────────────────────────────────────────────

  const canProceed = () => {
    if (currentStep === 0) return selectedGoal !== null;
    if (currentStep === 1) return true; // language always has a default
    if (currentStep === 2) return selectedDailyGoal !== null;
    return true;
  };

  const isLastStep = currentStep === TOTAL_STEPS - 1;

  // ── Render steps ──────────────────────────────────────────────────────────

  function renderStep0() {
    return (
      <>
        <StepHeader
          step={1}
          title="What's your learning goal?"
          subtitle="Pick the area you're most excited to explore. You can always change this later."
        />
        <View style={styles.goalGrid}>
          {GOAL_OPTIONS.map((goal) => {
            const selected = selectedGoal === goal.id;
            return (
              <TouchableOpacity
                key={goal.id}
                style={[styles.goalCard, selected && styles.goalCardSelected]}
                onPress={() => setSelectedGoal(goal.id)}
                activeOpacity={0.75}
              >
                {selected && (
                  <LinearGradient
                    colors={[`${goal.color}22`, `${goal.color}08`]}
                    style={StyleSheet.absoluteFillObject}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                  />
                )}
                <View
                  style={[
                    styles.goalIconCircle,
                    { backgroundColor: `${goal.color}20` },
                    selected && { backgroundColor: `${goal.color}30` },
                  ]}
                >
                  <Ionicons
                    name={goal.icon}
                    size={24}
                    color={selected ? goal.color : COLORS.textMuted}
                  />
                </View>
                <Text
                  style={[
                    styles.goalLabel,
                    selected && { color: goal.color, fontWeight: '700' },
                  ]}
                >
                  {goal.label}
                </Text>
                {selected && (
                  <View style={[styles.goalCheckmark, { backgroundColor: goal.color }]}>
                    <Ionicons name="checkmark" size={10} color="#fff" />
                  </View>
                )}
              </TouchableOpacity>
            );
          })}
        </View>
      </>
    );
  }

  function renderStep1() {
    return (
      <>
        <StepHeader
          step={2}
          title="Choose your language"
          subtitle="We'll deliver content in your preferred language for the best learning experience."
        />
        <View style={styles.langGrid}>
          {LANGUAGE_OPTIONS.map((lang) => {
            const selected = selectedLanguage === lang.code;
            return (
              <TouchableOpacity
                key={lang.code}
                style={[styles.langCard, selected && styles.langCardSelected]}
                onPress={() => setSelectedLanguage(lang.code)}
                activeOpacity={0.75}
              >
                <Text style={styles.langFlag}>{lang.flag}</Text>
                <Text style={[styles.langName, selected && styles.langNameSelected]}>
                  {lang.name}
                </Text>
                <Text style={styles.langNative}>{lang.nativeName}</Text>
                {selected && (
                  <View style={styles.langSelectedDot} />
                )}
              </TouchableOpacity>
            );
          })}
        </View>
      </>
    );
  }

  function renderStep2() {
    const selected = DAILY_GOAL_OPTIONS.find((g) => g.id === selectedDailyGoal);
    return (
      <>
        <StepHeader
          step={3}
          title="Set your daily goal"
          subtitle="Consistent practice is the key to mastery. How much time can you commit each day?"
        />

        {/* Motivational message */}
        {selected && (
          <View style={styles.motivationBox}>
            <Text style={styles.motivationEmoji}>{selected.emoji}</Text>
            <Text style={styles.motivationText}>{selected.motivational}</Text>
          </View>
        )}

        <View style={styles.dailyGoalList}>
          {DAILY_GOAL_OPTIONS.map((option) => {
            const isSelected = selectedDailyGoal === option.id;
            return (
              <TouchableOpacity
                key={option.id}
                style={[styles.dailyGoalCard, isSelected && styles.dailyGoalCardSelected]}
                onPress={() => setSelectedDailyGoal(option.id)}
                activeOpacity={0.75}
              >
                {isSelected && (
                  <LinearGradient
                    colors={['#6C63FF18', '#6C63FF08']}
                    style={StyleSheet.absoluteFillObject}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                  />
                )}
                <Text style={styles.dailyGoalEmoji}>{option.emoji}</Text>
                <Text style={[styles.dailyGoalLabel, isSelected && styles.dailyGoalLabelSelected]}>
                  {option.label}
                </Text>
                {isSelected && (
                  <View style={styles.dailyGoalCheck}>
                    <Ionicons name="checkmark-circle" size={22} color={COLORS.primary} />
                  </View>
                )}
              </TouchableOpacity>
            );
          })}
        </View>
      </>
    );
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <View style={styles.root}>
      {/* Background gradient */}
      <LinearGradient
        colors={['#6C63FF14', '#0A0A1A00']}
        style={styles.bgGradient}
        start={{ x: 0.5, y: 0 }}
        end={{ x: 0.5, y: 0.5 }}
      />

      {/* ── Header row ──────────────────────────────────────── */}
      <View style={styles.topBar}>
        {currentStep > 0 ? (
          <TouchableOpacity
            style={styles.topBackBtn}
            onPress={handleBack}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          >
            <Ionicons name="arrow-back" size={20} color="#AAAACC" />
          </TouchableOpacity>
        ) : (
          <View style={styles.topBackBtn} />
        )}

        <TouchableOpacity
          onPress={handleSkip}
          hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
        >
          <Text style={styles.skipText}>Skip</Text>
        </TouchableOpacity>
      </View>

      {/* ── Logo mark ───────────────────────────────────────── */}
      <View style={styles.logoRow}>
        <LinearGradient
          colors={[COLORS.primary, COLORS.primaryLight]}
          style={styles.logoMark}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        >
          <Text style={styles.logoMarkText}>T</Text>
        </LinearGradient>
        <Text style={styles.logoLabel}>
          Tech<Text style={styles.logoAccent}>Sei</Text>
        </Text>
      </View>

      {/* ── Animated step content ───────────────────────────── */}
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        <Animated.View
          style={[
            styles.stepContent,
            { opacity: fadeAnim, transform: [{ translateY: slideAnim }] },
          ]}
        >
          {currentStep === 0 && renderStep0()}
          {currentStep === 1 && renderStep1()}
          {currentStep === 2 && renderStep2()}
        </Animated.View>
      </ScrollView>

      {/* ── Bottom bar ──────────────────────────────────────── */}
      <View style={styles.bottomBar}>
        {/* Progress dots */}
        <View style={styles.progressDots}>
          {Array.from({ length: TOTAL_STEPS }).map((_, i) => (
            <View
              key={i}
              style={[
                styles.dot,
                i === currentStep
                  ? styles.dotActive
                  : i < currentStep
                  ? styles.dotDone
                  : styles.dotInactive,
              ]}
            />
          ))}
        </View>

        {/* Next / Get Started button */}
        <TouchableOpacity
          style={[styles.nextBtnWrapper, !canProceed() && { opacity: 0.5 }]}
          onPress={handleNext}
          disabled={!canProceed() || isSaving}
          activeOpacity={0.85}
        >
          <LinearGradient
            colors={
              isSaving
                ? [COLORS.primaryDim, '#5e4fa3']
                : [COLORS.primary, COLORS.primaryLight]
            }
            style={styles.nextBtn}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            {isSaving ? (
              <ActivityIndicator color="#fff" size="small" />
            ) : (
              <View style={styles.nextBtnInner}>
                <Text style={styles.nextBtnText}>
                  {isLastStep ? 'Get Started' : 'Next'}
                </Text>
                <Ionicons
                  name={isLastStep ? 'rocket-outline' : 'arrow-forward'}
                  size={18}
                  color="#fff"
                />
              </View>
            )}
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: COLORS.bg,
  },
  bgGradient: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 300,
  },

  // ── Top bar ──
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 56 : 32,
    paddingBottom: 8,
  },
  topBackBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: COLORS.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  skipText: {
    color: COLORS.textMuted,
    fontSize: 14,
    fontWeight: '600',
  },

  // ── Logo ──
  logoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 24,
    marginBottom: 8,
    gap: 10,
  },
  logoMark: {
    width: 36,
    height: 36,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoMarkText: {
    fontSize: 18,
    fontWeight: '800',
    color: '#fff',
  },
  logoLabel: {
    fontSize: 20,
    fontWeight: '800',
    color: COLORS.text,
  },
  logoAccent: {
    color: COLORS.primary,
  },

  // ── Scroll content ──
  scrollContent: {
    paddingHorizontal: 24,
    paddingTop: 16,
    paddingBottom: 20,
  },
  stepContent: {
    flex: 1,
  },

  // ── Step 1: Goal grid ──
  goalGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  goalCard: {
    width: (SCREEN_WIDTH - 48 - 12) / 2,
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    padding: 18,
    alignItems: 'flex-start',
    gap: 12,
    overflow: 'hidden',
    position: 'relative',
  },
  goalCardSelected: {
    borderColor: COLORS.primary,
  },
  goalIconCircle: {
    width: 48,
    height: 48,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  goalLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: COLORS.textMuted,
    lineHeight: 18,
  },
  goalCheckmark: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 18,
    height: 18,
    borderRadius: 9,
    alignItems: 'center',
    justifyContent: 'center',
  },

  // ── Step 2: Language grid ──
  langGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  langCard: {
    width: (SCREEN_WIDTH - 48 - 10) / 2,
    backgroundColor: COLORS.surface,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    padding: 16,
    alignItems: 'center',
    gap: 6,
    position: 'relative',
  },
  langCardSelected: {
    borderColor: COLORS.primary,
    backgroundColor: COLORS.surfaceHighlight,
  },
  langFlag: {
    fontSize: 28,
  },
  langName: {
    fontSize: 14,
    fontWeight: '700',
    color: COLORS.textMuted,
  },
  langNameSelected: {
    color: COLORS.text,
  },
  langNative: {
    fontSize: 11,
    color: COLORS.inactive,
    fontWeight: '500',
  },
  langSelectedDot: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.primary,
  },

  // ── Step 3: Daily goal ──
  motivationBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#6C63FF14',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#6C63FF30',
    padding: 16,
    marginBottom: 20,
    gap: 12,
  },
  motivationEmoji: {
    fontSize: 24,
    lineHeight: 28,
  },
  motivationText: {
    flex: 1,
    fontSize: 13,
    color: '#AAAADD',
    lineHeight: 20,
    fontStyle: 'italic',
  },
  dailyGoalList: {
    gap: 10,
  },
  dailyGoalCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    padding: 18,
    gap: 14,
    overflow: 'hidden',
    position: 'relative',
  },
  dailyGoalCardSelected: {
    borderColor: COLORS.primary,
  },
  dailyGoalEmoji: {
    fontSize: 24,
    width: 32,
    textAlign: 'center',
  },
  dailyGoalLabel: {
    flex: 1,
    fontSize: 16,
    fontWeight: '600',
    color: COLORS.textMuted,
  },
  dailyGoalLabelSelected: {
    color: COLORS.text,
  },
  dailyGoalCheck: {
    marginLeft: 'auto',
  },

  // ── Bottom bar ──
  bottomBar: {
    paddingHorizontal: 24,
    paddingBottom: Platform.OS === 'ios' ? 40 : 24,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#1C1C34',
    backgroundColor: COLORS.bg,
    gap: 20,
  },
  progressDots: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 8,
  },
  dot: {
    height: 8,
    borderRadius: 4,
  },
  dotActive: {
    width: 24,
    backgroundColor: COLORS.primary,
  },
  dotDone: {
    width: 8,
    backgroundColor: '#6C63FF66',
  },
  dotInactive: {
    width: 8,
    backgroundColor: COLORS.inactive,
  },

  // ── Next button ──
  nextBtnWrapper: {
    borderRadius: 14,
    overflow: 'hidden',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.35,
    shadowRadius: 12,
    elevation: 8,
  },
  nextBtn: {
    height: 54,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 14,
  },
  nextBtnInner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  nextBtnText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
});
