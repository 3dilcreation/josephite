// ============================================================
// TechSei LMS — Student Progress & Achievements Screen
// ============================================================
import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Animated,
  TouchableOpacity,
  Dimensions,
  FlatList,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Circle, Defs, LinearGradient as SvgGradient, Stop } from 'react-native-svg';
import { useGamificationStore, LEVEL_THRESHOLDS, selectLevelProgress } from '../../stores/gamificationStore';
import { useAuthStore } from '../../stores/authStore';
import BadgeCard from '../../components/student/BadgeCard';
import type { Badge, StudentBadge } from '../../types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CARD_PADDING = 16;
const CONTENT_WIDTH = SCREEN_WIDTH - CARD_PADDING * 2;

// ─── Colors ───────────────────────────────────────────────────────────────────
const C = {
  bg: '#0A0A1A',
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  primary: '#6C63FF',
  accent: '#43E97B',
  gold: '#FFD700',
  warning: '#FFB84C',
  text: '#FFFFFF',
  textSec: '#9494B8',
  textMuted: '#5A5A7A',
  border: '#2A2A4A',
  error: '#FF4B4B',
};

// ─── Level title map ──────────────────────────────────────────────────────────
const LEVEL_TITLES: Record<number, string> = {
  1: 'Newcomer', 2: 'Explorer', 3: 'Apprentice', 4: 'Scholar',
  5: 'Rising Star', 6: 'Adept', 7: 'Expert', 8: 'Master',
  9: 'Grandmaster', 10: 'Legend',
};

// ─── Circular XP Ring ─────────────────────────────────────────────────────────
const RING_SIZE = 160;
const STROKE = 14;
const RADIUS = (RING_SIZE - STROKE) / 2;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

interface XPRingProps {
  progress: number; // 0–1
}

function XPRing({ progress }: XPRingProps) {
  const animProgress = useRef(new Animated.Value(0)).current;
  const [displayProgress, setDisplayProgress] = useState(0);

  useEffect(() => {
    Animated.timing(animProgress, {
      toValue: progress,
      duration: 1200,
      useNativeDriver: false,
    }).start();

    const id = animProgress.addListener(({ value }) => setDisplayProgress(value));
    return () => animProgress.removeListener(id);
  }, [progress, animProgress]);

  const strokeDashoffset = CIRCUMFERENCE * (1 - displayProgress);

  return (
    <Svg width={RING_SIZE} height={RING_SIZE}>
      <Defs>
        <SvgGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <Stop offset="0%" stopColor="#6C63FF" />
          <Stop offset="100%" stopColor="#43E97B" />
        </SvgGradient>
      </Defs>
      {/* Background track */}
      <Circle
        cx={RING_SIZE / 2}
        cy={RING_SIZE / 2}
        r={RADIUS}
        stroke="#2A2A4A"
        strokeWidth={STROKE}
        fill="none"
      />
      {/* Progress arc */}
      <Circle
        cx={RING_SIZE / 2}
        cy={RING_SIZE / 2}
        r={RADIUS}
        stroke="url(#ringGrad)"
        strokeWidth={STROKE}
        fill="none"
        strokeDasharray={CIRCUMFERENCE}
        strokeDashoffset={strokeDashoffset}
        strokeLinecap="round"
        rotation="-90"
        origin={`${RING_SIZE / 2}, ${RING_SIZE / 2}`}
      />
    </Svg>
  );
}

// ─── Skills data ──────────────────────────────────────────────────────────────
const SKILLS = [
  { name: 'Web Dev', icon: '🌐', pct: 45, color: C.primary },
  { name: 'Data Science', icon: '📊', pct: 20, color: '#38F9D7' },
  { name: 'Mobile Dev', icon: '📱', pct: 30, color: '#FF6584' },
  { name: 'Cloud', icon: '☁️', pct: 15, color: C.warning },
  { name: 'AI / ML', icon: '🤖', pct: 10, color: C.accent },
  { name: 'Cyber Security', icon: '🔐', pct: 5, color: C.gold },
];

// ─── Static mock badges ───────────────────────────────────────────────────────
const ALL_BADGES: Badge[] = [
  { id: 'b1', name: 'First Step', description: 'Complete your first lesson', icon: '👣', xp_required: 0, badge_type: 'achievement', rarity: 'common' },
  { id: 'b2', name: 'Week Warrior', description: 'Maintain a 7-day streak', icon: '⚡', xp_required: 250, badge_type: 'streak', rarity: 'rare' },
  { id: 'b3', name: 'Quiz Master', description: 'Score 100% on 5 quizzes', icon: '🎯', xp_required: 500, badge_type: 'achievement', rarity: 'epic' },
  { id: 'b4', name: 'Course Champion', description: 'Complete a full course', icon: '🏆', xp_required: 1000, badge_type: 'course', rarity: 'legendary' },
  { id: 'b5', name: 'Speed Learner', description: 'Finish 3 lessons in one day', icon: '🚀', xp_required: 300, badge_type: 'achievement', rarity: 'rare' },
  { id: 'b6', name: 'Night Owl', description: 'Study after midnight', icon: '🦉', xp_required: 100, badge_type: 'achievement', rarity: 'common' },
];

const MOCK_CERTIFICATES = [
  { id: 'c1', title: 'Intro to Web Development', date: '2025-03-10', instructor: 'Prof. Rajan' },
  { id: 'c2', title: 'Python for Beginners', date: '2025-01-22', instructor: 'Dr. Meera' },
];

// ─── Weekly XP mock data ──────────────────────────────────────────────────────
const WEEKLY_XP = [
  { day: 'Mon', xp: 120 },
  { day: 'Tue', xp: 75 },
  { day: 'Wed', xp: 200 },
  { day: 'Thu', xp: 50 },
  { day: 'Fri', xp: 175 },
  { day: 'Sat', xp: 300 },
  { day: 'Sun', xp: 240 },
];
const MAX_WEEKLY_XP = Math.max(...WEEKLY_XP.map((d) => d.xp));

// ─── Streak calendar (30 days) ────────────────────────────────────────────────
function generateCalendar(streak: number): { date: string; level: number }[] {
  const today = new Date();
  return Array.from({ length: 30 }, (_, i) => {
    const d = new Date(today);
    d.setDate(today.getDate() - (29 - i));
    const daysFromEnd = 29 - i;
    const level = daysFromEnd < streak ? (daysFromEnd < streak * 0.5 ? 3 : 2) : (Math.random() < 0.2 ? 1 : 0);
    return { date: d.toISOString().slice(0, 10), level };
  });
}

const CALENDAR_COLORS = ['#1E1E3A', '#4A3F8F', '#6C63FF', '#B9B5FF'];

// ─── Main Screen ──────────────────────────────────────────────────────────────
export default function ProgressScreen() {
  const { xp, level, streak, badges: earnedBadges } = useGamificationStore();
  const { profile } = useAuthStore();

  const rawProgress = selectLevelProgress({ xp, level, streak, badges: earnedBadges, dailyGoalCompleted: false, lastStreakDate: null });
  const currentThreshold = LEVEL_THRESHOLDS[level - 1] ?? 0;
  const nextThreshold = LEVEL_THRESHOLDS[level] ?? LEVEL_THRESHOLDS[LEVEL_THRESHOLDS.length - 1];
  const xpIntoLevel = xp - currentThreshold;
  const xpNeeded = nextThreshold - currentThreshold;

  const levelTitle = LEVEL_TITLES[level] ?? 'Legend';
  const totalLessons = profile?.total_lessons_completed ?? 0;
  const totalCourses = profile?.total_courses_completed ?? 0;
  const hoursLearned = Math.round((profile?.total_time_spent_seconds ?? 0) / 3600);

  const earnedBadgeIds = new Set(earnedBadges.map((b) => b.badge_id));
  const calendarData = generateCalendar(streak);

  const statCards = [
    { label: 'Lessons', value: totalLessons, icon: 'book-outline' as const, color: C.primary },
    { label: 'Courses', value: totalCourses, icon: 'school-outline' as const, color: '#38F9D7' },
    { label: 'Hours', value: hoursLearned, icon: 'time-outline' as const, color: C.warning },
    { label: 'Streak', value: streak, icon: 'flame-outline' as const, color: C.accent },
  ];

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>My Progress</Text>
          <TouchableOpacity style={styles.headerBtn}>
            <Ionicons name="share-social-outline" size={22} color={C.textSec} />
          </TouchableOpacity>
        </View>

        {/* ── Level & XP Card ─────────────────────────────────────────────── */}
        <LinearGradient
          colors={['#141428', '#1E1E3A']}
          style={styles.levelCard}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        >
          <View style={styles.levelCardInner}>
            {/* Ring */}
            <View style={styles.ringWrapper}>
              <XPRing progress={rawProgress} />
              <View style={styles.ringCenter}>
                <Text style={styles.ringLevel}>{level}</Text>
                <Text style={styles.ringLevelLabel}>LVL</Text>
              </View>
            </View>

            {/* Text info */}
            <View style={styles.levelInfo}>
              <LinearGradient
                colors={[C.primary, '#8B5CF6']}
                style={styles.levelBadgeGrad}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
              >
                <Text style={styles.levelBadgeText}>Level {level}</Text>
              </LinearGradient>
              <Text style={styles.levelTitle}>{levelTitle}</Text>

              {/* XP bar */}
              <View style={styles.xpBarTrack}>
                <Animated.View
                  style={[styles.xpBarFill, { width: `${rawProgress * 100}%` }]}
                />
              </View>
              <Text style={styles.xpText}>
                {xpIntoLevel.toLocaleString()} / {xpNeeded.toLocaleString()} XP to Level {level + 1}
              </Text>
              <Text style={styles.totalXPText}>Total: {xp.toLocaleString()} XP</Text>
            </View>
          </View>
        </LinearGradient>

        {/* ── Stats Row ────────────────────────────────────────────────────── */}
        <View style={styles.statsRow}>
          {statCards.map((s) => (
            <View key={s.label} style={styles.statCard}>
              <Ionicons name={s.icon} size={22} color={s.color} />
              <Text style={[styles.statValue, { color: s.color }]}>{s.value}</Text>
              <Text style={styles.statLabel}>{s.label}</Text>
            </View>
          ))}
        </View>

        {/* ── Skills Map ───────────────────────────────────────────────────── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Skills Map</Text>
          <View style={styles.skillsGrid}>
            {SKILLS.map((skill) => (
              <View key={skill.name} style={styles.skillCard}>
                <Text style={styles.skillIcon}>{skill.icon}</Text>
                <Text style={styles.skillName}>{skill.name}</Text>
                <View style={styles.skillTrack}>
                  <View
                    style={[
                      styles.skillFill,
                      { width: `${skill.pct}%`, backgroundColor: skill.color },
                    ]}
                  />
                </View>
                <Text style={[styles.skillPct, { color: skill.color }]}>{skill.pct}%</Text>
              </View>
            ))}
          </View>
        </View>

        {/* ── Streak Calendar ──────────────────────────────────────────────── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Activity Calendar</Text>
          <View style={styles.calendarCard}>
            <View style={styles.calendarGrid}>
              {calendarData.map((day, idx) => (
                <View
                  key={idx}
                  style={[
                    styles.calendarCell,
                    { backgroundColor: CALENDAR_COLORS[day.level] },
                  ]}
                />
              ))}
            </View>
            <View style={styles.calendarLegend}>
              <Text style={styles.legendLabel}>Less</Text>
              {CALENDAR_COLORS.map((c, i) => (
                <View key={i} style={[styles.legendDot, { backgroundColor: c }]} />
              ))}
              <Text style={styles.legendLabel}>More</Text>
            </View>
          </View>
        </View>

        {/* ── Weekly XP Chart ──────────────────────────────────────────────── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>This Week's XP</Text>
          <View style={styles.chartCard}>
            <View style={styles.barChart}>
              {WEEKLY_XP.map((d) => {
                const height = Math.max(6, (d.xp / MAX_WEEKLY_XP) * 100);
                return (
                  <View key={d.day} style={styles.barColumn}>
                    <Text style={styles.barXP}>{d.xp}</Text>
                    <LinearGradient
                      colors={[C.primary, '#8B5CF6']}
                      style={[styles.bar, { height }]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 0, y: 1 }}
                    />
                    <Text style={styles.barDay}>{d.day}</Text>
                  </View>
                );
              })}
            </View>
          </View>
        </View>

        {/* ── Badges ───────────────────────────────────────────────────────── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Badges</Text>

          {/* Earned */}
          <Text style={styles.subsectionLabel}>Earned ({earnedBadgeIds.size})</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.badgeRow}>
            {ALL_BADGES.filter((b) => earnedBadgeIds.has(b.id)).length === 0 ? (
              <View style={styles.emptyBadge}>
                <Text style={styles.emptyBadgeText}>Complete lessons to earn badges!</Text>
              </View>
            ) : (
              ALL_BADGES.filter((b) => earnedBadgeIds.has(b.id)).map((badge) => {
                const earned = earnedBadges.find((eb) => eb.badge_id === badge.id);
                return (
                  <BadgeCard
                    key={badge.id}
                    badge={badge}
                    earned
                    earnedAt={earned?.earned_at}
                    size="medium"
                  />
                );
              })
            )}
          </ScrollView>

          {/* Locked */}
          <Text style={styles.subsectionLabel}>Locked</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.badgeRow}>
            {ALL_BADGES.filter((b) => !earnedBadgeIds.has(b.id)).map((badge) => (
              <BadgeCard key={badge.id} badge={badge} earned={false} size="medium" />
            ))}
          </ScrollView>
        </View>

        {/* ── Certificates ─────────────────────────────────────────────────── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Certificates</Text>
          {MOCK_CERTIFICATES.length === 0 ? (
            <View style={styles.emptyCerts}>
              <Ionicons name="ribbon-outline" size={40} color={C.textMuted} />
              <Text style={styles.emptyCertsText}>Complete a course to earn your first certificate!</Text>
            </View>
          ) : (
            MOCK_CERTIFICATES.map((cert) => (
              <View key={cert.id} style={styles.certCard}>
                <LinearGradient
                  colors={[C.gold, '#FFA500']}
                  style={styles.certIcon}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                >
                  <Ionicons name="ribbon" size={24} color="#0A0A1A" />
                </LinearGradient>
                <View style={styles.certInfo}>
                  <Text style={styles.certTitle}>{cert.title}</Text>
                  <Text style={styles.certMeta}>
                    {cert.instructor} · {new Date(cert.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                  </Text>
                </View>
                <TouchableOpacity style={styles.downloadBtn}>
                  <Ionicons name="download-outline" size={16} color={C.primary} />
                  <Text style={styles.downloadBtnText}>PDF</Text>
                </TouchableOpacity>
              </View>
            ))
          )}
        </View>

        <View style={{ height: 32 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: C.bg },
  scroll: { flex: 1 },
  content: { paddingHorizontal: CARD_PADDING, paddingTop: 8 },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  headerTitle: { fontSize: 26, fontWeight: '800', color: C.text },
  headerBtn: {
    width: 40, height: 40, borderRadius: 20,
    backgroundColor: C.surfaceLight,
    alignItems: 'center', justifyContent: 'center',
  },

  // Level card
  levelCard: {
    borderRadius: 24,
    padding: 20,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: C.border,
  },
  levelCardInner: { flexDirection: 'row', alignItems: 'center', gap: 20 },
  ringWrapper: { position: 'relative', width: RING_SIZE, height: RING_SIZE },
  ringCenter: {
    position: 'absolute', inset: 0,
    alignItems: 'center', justifyContent: 'center',
  },
  ringLevel: { fontSize: 38, fontWeight: '900', color: C.text },
  ringLevelLabel: { fontSize: 11, color: C.textSec, fontWeight: '700', letterSpacing: 2 },
  levelInfo: { flex: 1, gap: 6 },
  levelBadgeGrad: {
    alignSelf: 'flex-start',
    paddingHorizontal: 10, paddingVertical: 4,
    borderRadius: 10,
  },
  levelBadgeText: { color: C.text, fontWeight: '700', fontSize: 12 },
  levelTitle: { color: C.text, fontSize: 20, fontWeight: '800' },
  xpBarTrack: {
    height: 8, borderRadius: 4,
    backgroundColor: C.surfaceLight,
    overflow: 'hidden',
  },
  xpBarFill: {
    height: '100%', borderRadius: 4,
    backgroundColor: C.primary,
  },
  xpText: { fontSize: 12, color: C.textSec, fontWeight: '600' },
  totalXPText: { fontSize: 11, color: C.textMuted },

  // Stats
  statsRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 24,
  },
  statCard: {
    flex: 1,
    backgroundColor: C.surface,
    borderRadius: 16,
    paddingVertical: 14,
    alignItems: 'center',
    gap: 4,
    borderWidth: 1,
    borderColor: C.border,
  },
  statValue: { fontSize: 20, fontWeight: '800' },
  statLabel: { fontSize: 10, color: C.textSec, fontWeight: '600' },

  // Section
  section: { marginBottom: 28 },
  sectionTitle: {
    fontSize: 18, fontWeight: '800', color: C.text,
    marginBottom: 14,
  },
  subsectionLabel: {
    fontSize: 13, fontWeight: '600', color: C.textSec,
    marginBottom: 10, marginTop: 4,
  },

  // Skills
  skillsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  skillCard: {
    width: (CONTENT_WIDTH - 10) / 2,
    backgroundColor: C.surface,
    borderRadius: 16,
    padding: 14,
    gap: 6,
    borderWidth: 1,
    borderColor: C.border,
  },
  skillIcon: { fontSize: 22 },
  skillName: { fontSize: 13, fontWeight: '700', color: C.text },
  skillTrack: {
    height: 6, borderRadius: 3,
    backgroundColor: C.surfaceLight,
    overflow: 'hidden',
  },
  skillFill: { height: '100%', borderRadius: 3 },
  skillPct: { fontSize: 12, fontWeight: '700' },

  // Calendar
  calendarCard: {
    backgroundColor: C.surface,
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: C.border,
  },
  calendarGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 5,
    marginBottom: 12,
  },
  calendarCell: {
    width: (CONTENT_WIDTH - 32 - 5 * 6) / 7,
    height: (CONTENT_WIDTH - 32 - 5 * 6) / 7,
    borderRadius: 3,
  },
  calendarLegend: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    justifyContent: 'flex-end',
  },
  legendLabel: { fontSize: 10, color: C.textMuted },
  legendDot: {
    width: 10, height: 10, borderRadius: 2,
  },

  // Bar chart
  chartCard: {
    backgroundColor: C.surface,
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: C.border,
  },
  barChart: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    height: 140,
  },
  barColumn: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 4,
  },
  barXP: { fontSize: 9, color: C.textMuted, fontWeight: '600' },
  bar: {
    width: 22,
    borderRadius: 6,
    minHeight: 6,
  },
  barDay: { fontSize: 11, color: C.textSec, fontWeight: '600' },

  // Badges
  badgeRow: { marginBottom: 4 },
  emptyBadge: {
    paddingVertical: 20,
    paddingHorizontal: 16,
  },
  emptyBadgeText: { color: C.textMuted, fontSize: 13 },

  // Certificates
  emptyCerts: {
    backgroundColor: C.surface,
    borderRadius: 16,
    padding: 28,
    alignItems: 'center',
    gap: 12,
    borderWidth: 1,
    borderColor: C.border,
  },
  emptyCertsText: {
    color: C.textMuted, fontSize: 13, textAlign: 'center', maxWidth: 220,
  },
  certCard: {
    backgroundColor: C.surface,
    borderRadius: 16,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: C.border,
  },
  certIcon: {
    width: 52, height: 52, borderRadius: 14,
    alignItems: 'center', justifyContent: 'center',
  },
  certInfo: { flex: 1 },
  certTitle: { color: C.text, fontSize: 14, fontWeight: '700', marginBottom: 2 },
  certMeta: { color: C.textSec, fontSize: 12 },
  downloadBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: C.surfaceLight,
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 10,
  },
  downloadBtnText: { color: C.primary, fontSize: 12, fontWeight: '700' },
});
