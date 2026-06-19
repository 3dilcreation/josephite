// ============================================================
// TechSei LMS — Student Home Screen
// ============================================================
import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Image,
  Dimensions,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuthStore } from '../../stores/authStore';
import { useCourseStore } from '../../stores/courseStore';
import {
  useGamificationStore,
  LEVEL_THRESHOLDS,
  selectLevelProgress,
  XP_REWARDS,
} from '../../stores/gamificationStore';
import { CourseCard } from '../../components/common/CourseCard';
import { ProgressBar } from '../../components/common/ProgressBar';
import type { Course, LeaderboardEntry } from '../../types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

const COLORS = {
  background: '#0A0A1A',
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  primary: '#6C63FF',
  accent: '#43E97B',
  warning: '#FFB84C',
  text: '#FFFFFF',
  textMuted: '#8A8AAA',
  border: '#2A2A4A',
  gold: '#FFD700',
  silver: '#C0C0C0',
  bronze: '#CD7F32',
};

const DAILY_XP_GOAL = 200;

// ── Skeleton Loader ───────────────────────────────────────────────────────────
function SkeletonBox({ width, height, style }: { width?: number | string; height: number; style?: object }) {
  const shimmer = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(shimmer, { toValue: 1, duration: 900, useNativeDriver: true }),
        Animated.timing(shimmer, { toValue: 0, duration: 900, useNativeDriver: true }),
      ])
    ).start();
  }, [shimmer]);
  const opacity = shimmer.interpolate({ inputRange: [0, 1], outputRange: [0.3, 0.7] });
  return (
    <Animated.View
      style={[
        { width: width ?? '100%', height, borderRadius: 10, backgroundColor: COLORS.surfaceLight, opacity },
        style,
      ]}
    />
  );
}

function HomeSkeleton() {
  return (
    <View style={{ paddingHorizontal: 20, paddingTop: 16 }}>
      <View style={styles.skeletonHeader}>
        <SkeletonBox width={48} height={48} style={{ borderRadius: 24 }} />
        <View style={{ flex: 1, marginLeft: 12, gap: 8 }}>
          <SkeletonBox width={160} height={16} />
          <SkeletonBox width={100} height={12} />
        </View>
      </View>
      <SkeletonBox height={100} style={{ marginTop: 16, borderRadius: 16 }} />
      <SkeletonBox height={80} style={{ marginTop: 12, borderRadius: 16 }} />
      <SkeletonBox height={140} style={{ marginTop: 12, borderRadius: 16 }} />
      <View style={{ flexDirection: 'row', gap: 12, marginTop: 12 }}>
        <SkeletonBox width={(SCREEN_WIDTH - 52) / 2} height={180} style={{ borderRadius: 16 }} />
        <SkeletonBox width={(SCREEN_WIDTH - 52) / 2} height={180} style={{ borderRadius: 16 }} />
      </View>
    </View>
  );
}

// ── XP Pop Animation ─────────────────────────────────────────────────────────
function XPBadge({ xp }: { xp: number }) {
  const float = useRef(new Animated.Value(0)).current;
  const opacity = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    Animated.parallel([
      Animated.timing(float, { toValue: -40, duration: 1200, useNativeDriver: true }),
      Animated.timing(opacity, { toValue: 0, duration: 1200, useNativeDriver: true }),
    ]).start();
  }, [float, opacity]);
  return (
    <Animated.View style={[styles.xpBadge, { transform: [{ translateY: float }], opacity }]}>
      <Text style={styles.xpBadgeText}>+{xp} XP</Text>
    </Animated.View>
  );
}

// ── Leaderboard Row ───────────────────────────────────────────────────────────
function LeaderboardRow({ entry, index }: { entry: LeaderboardEntry; index: number }) {
  const medalColors = [COLORS.gold, COLORS.silver, COLORS.bronze];
  const rankColor = index < 3 ? medalColors[index] : COLORS.textMuted;

  return (
    <View style={styles.leaderboardRow}>
      <View style={[styles.rankBadge, { borderColor: rankColor }]}>
        <Text style={[styles.rankText, { color: rankColor }]}>{entry.rank}</Text>
      </View>
      {entry.avatar_url ? (
        <Image source={{ uri: entry.avatar_url }} style={styles.leaderboardAvatar} />
      ) : (
        <View style={[styles.leaderboardAvatar, styles.avatarPlaceholder]}>
          <Text style={styles.avatarInitial}>{entry.name.charAt(0).toUpperCase()}</Text>
        </View>
      )}
      <View style={{ flex: 1 }}>
        <Text style={styles.leaderboardName} numberOfLines={1}>{entry.name}</Text>
        <Text style={styles.leaderboardLevel}>Level {entry.level}</Text>
      </View>
      <View style={styles.xpChip}>
        <Ionicons name="flash" size={12} color={COLORS.accent} />
        <Text style={styles.xpChipText}>{entry.xp.toLocaleString()}</Text>
      </View>
    </View>
  );
}

// ── Mock daily challenge (would come from API in production) ──────────────────
const DAILY_CHALLENGE = {
  title: "JavaScript Fundamentals",
  description: "Complete today's 5-question quiz for bonus XP",
  bonusXP: 100,
  timeLimit: "10 min",
};

// ── Mock leaderboard (would come from Supabase in production) ─────────────────
const MOCK_LEADERBOARD: LeaderboardEntry[] = [
  { rank: 1, user_id: '1', name: 'Priya Sharma', avatar_url: null, xp: 4850, level: 7, streak: 15 },
  { rank: 2, user_id: '2', name: 'Rahul Verma', avatar_url: null, xp: 3920, level: 6, streak: 8 },
  { rank: 3, user_id: '3', name: 'Ananya Patel', avatar_url: null, xp: 3210, level: 5, streak: 12 },
];

// ── Main Screen ───────────────────────────────────────────────────────────────
export default function HomeScreen() {
  const router = useRouter();
  const { user, profile } = useAuthStore();
  const {
    courses,
    enrolledCourses,
    isLoading: coursesLoading,
    fetchCourses,
    fetchEnrolledCourses,
    getCourseProgress,
  } = useCourseStore();
  const {
    xp,
    level,
    streak,
    badges,
    dailyGoalCompleted,
    loadGamification,
    checkAndUpdateStreak,
  } = useGamificationStore();

  const [refreshing, setRefreshing] = useState(false);
  const [showXPPop, setShowXPPop] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);

  const streakPulse = useRef(new Animated.Value(1)).current;

  // Streak pulse animation when streak > 0
  useEffect(() => {
    if (streak > 0) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(streakPulse, { toValue: 1.06, duration: 800, useNativeDriver: true }),
          Animated.timing(streakPulse, { toValue: 1, duration: 800, useNativeDriver: true }),
        ])
      ).start();
    }
  }, [streak, streakPulse]);

  const loadData = useCallback(async () => {
    if (!user) return;
    try {
      await Promise.all([
        fetchCourses(),
        fetchEnrolledCourses(user.id),
        loadGamification(user.id),
      ]);
      await checkAndUpdateStreak();
    } catch (e) {
      // Silent fail — data shown from cache
    } finally {
      setInitialLoading(false);
    }
  }, [user, fetchCourses, fetchEnrolledCourses, loadGamification, checkAndUpdateStreak]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  }, [loadData]);

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  };

  const firstName = user?.name?.split(' ')[0] ?? 'Learner';

  // Daily XP progress (simplified: XP % daily goal)
  const todayXP = xp % DAILY_XP_GOAL;
  const dailyXPProgress = Math.min((todayXP / DAILY_XP_GOAL) * 100, 100);

  // Last accessed course — first enrolled course
  const continueCoure = enrolledCourses[0] ?? null;
  const continueProgress = continueCoure ? getCourseProgress(continueCoure.id) : 0;

  // Recommended: courses not yet enrolled, limited to 5
  const enrolledIds = new Set(enrolledCourses.map((c) => c.id));
  const recommended = courses.filter((c) => !enrolledIds.has(c.id)).slice(0, 5);

  // Recent badge
  const recentBadge = badges.length > 0 ? badges[badges.length - 1] : null;
  const badgeInfo = recentBadge?.badge;

  if (initialLoading && coursesLoading) {
    return (
      <SafeAreaView style={styles.root}>
        <HomeSkeleton />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.root} edges={['top']}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={COLORS.primary}
            colors={[COLORS.primary]}
          />
        }
      >
        {/* ── Header ──────────────────────────────────────────── */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <TouchableOpacity onPress={() => router.push('/student/profile' as any)} style={styles.avatarContainer}>
              {user?.avatar_url ? (
                <Image source={{ uri: user.avatar_url }} style={styles.avatar} />
              ) : (
                <LinearGradient colors={['#6C63FF', '#8B5CF6']} style={styles.avatar}>
                  <Text style={styles.avatarInitialLarge}>{firstName.charAt(0).toUpperCase()}</Text>
                </LinearGradient>
              )}
              <View style={styles.levelBadgeSmall}>
                <Text style={styles.levelBadgeText}>{level}</Text>
              </View>
            </TouchableOpacity>
            <View style={{ marginLeft: 12 }}>
              <Text style={styles.greetingText}>{getGreeting()},</Text>
              <Text style={styles.nameText}>{firstName}! 👋</Text>
            </View>
          </View>
          <View style={styles.headerRight}>
            <TouchableOpacity style={styles.iconBtn} onPress={() => {}}>
              <Ionicons name="notifications-outline" size={22} color={COLORS.text} />
              <View style={styles.notifDot} />
            </TouchableOpacity>
            <TouchableOpacity style={styles.iconBtn} onPress={() => {}}>
              <Ionicons name="settings-outline" size={22} color={COLORS.text} />
            </TouchableOpacity>
          </View>
        </View>

        {/* ── Streak Banner ───────────────────────────────────── */}
        <Animated.View style={[{ transform: [{ scale: streakPulse }] }, styles.streakWrapper]}>
          <LinearGradient
            colors={['#FFB84C', '#FF6B35']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.streakCard}
          >
            <View style={styles.streakLeft}>
              <Text style={styles.streakEmoji}>{streak > 0 ? '🔥' : '⚡'}</Text>
              <View>
                <Text style={styles.streakTitle}>{streak > 0 ? `${streak} Day Streak!` : 'Start Your Streak!'}</Text>
                <Text style={styles.streakSub}>
                  {streak > 0 ? 'Keep it up — you\'re on fire!' : 'Complete a lesson to begin'}
                </Text>
              </View>
            </View>
            <View style={styles.streakRight}>
              <Text style={styles.streakCount}>{streak}</Text>
              <Text style={styles.streakDays}>days</Text>
            </View>
          </LinearGradient>
        </Animated.View>

        {/* ── Daily XP Progress ───────────────────────────────── */}
        <View style={styles.card}>
          <View style={styles.xpHeader}>
            <View style={styles.xpTitleRow}>
              <Ionicons name="flash" size={18} color={COLORS.accent} />
              <Text style={styles.cardTitle}>Daily XP Progress</Text>
            </View>
            <Text style={styles.xpCount}>
              <Text style={styles.xpCurrent}>{todayXP}</Text>
              <Text style={styles.xpMuted}>/{DAILY_XP_GOAL} XP</Text>
            </Text>
          </View>
          <ProgressBar
            progress={dailyXPProgress}
            colorStart="#43E97B"
            colorEnd="#38F9D7"
            height={10}
            animated
          />
          {dailyGoalCompleted && (
            <View style={styles.goalCompletedBadge}>
              <Ionicons name="checkmark-circle" size={14} color={COLORS.accent} />
              <Text style={styles.goalCompletedText}>Daily goal reached! 🎉</Text>
            </View>
          )}
        </View>

        {/* ── Continue Learning ────────────────────────────────── */}
        {continueCoure && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Continue Learning</Text>
            <TouchableOpacity
              style={styles.continueCard}
              activeOpacity={0.85}
              onPress={() => router.push(`/student/courses/${continueCoure.id}` as any)}
            >
              <View style={styles.continueThumbnail}>
                {continueCoure.thumbnail_url ? (
                  <Image source={{ uri: continueCoure.thumbnail_url }} style={StyleSheet.absoluteFill} />
                ) : (
                  <LinearGradient colors={['#6C63FF', '#A855F7']} style={StyleSheet.absoluteFill}>
                    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
                      <Ionicons name="book" size={32} color="#FFFFFF60" />
                    </View>
                  </LinearGradient>
                )}
              </View>
              <View style={styles.continueBody}>
                <Text style={styles.continueTitle} numberOfLines={2}>{continueCoure.title}</Text>
                <Text style={styles.continueInstructor}>{continueCoure.instructor_name}</Text>
                <View style={styles.continueProgressRow}>
                  <View style={{ flex: 1 }}>
                    <ProgressBar progress={continueProgress} colorStart="#6C63FF" colorEnd="#8B5CF6" height={6} animated />
                  </View>
                  <Text style={styles.continueProgressText}>{continueProgress}%</Text>
                </View>
                <TouchableOpacity
                  style={styles.resumeBtn}
                  onPress={() => router.push(`/student/courses/${continueCoure.id}` as any)}
                >
                  <LinearGradient colors={['#6C63FF', '#8B5CF6']} style={styles.resumeBtnGrad} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}>
                    <Ionicons name="play" size={14} color="#fff" />
                    <Text style={styles.resumeBtnText}>Resume</Text>
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            </TouchableOpacity>
          </View>
        )}

        {/* ── Enrolled Courses ─────────────────────────────────── */}
        {enrolledCourses.length > 0 && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>My Courses</Text>
              <TouchableOpacity onPress={() => router.push('/student/courses' as any)}>
                <Text style={styles.seeAll}>See all</Text>
              </TouchableOpacity>
            </View>
            <FlatList
              horizontal
              data={enrolledCourses}
              keyExtractor={(item) => item.id}
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={{ paddingRight: 20 }}
              renderItem={({ item }) => (
                <CourseCard
                  course={item}
                  variant="horizontal"
                  enrollmentProgress={getCourseProgress(item.id)}
                  onPress={() => router.push(`/student/courses/${item.id}` as any)}
                />
              )}
            />
          </View>
        )}

        {/* ── Recommended Courses ──────────────────────────────── */}
        {recommended.length > 0 && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Recommended for You</Text>
              <TouchableOpacity onPress={() => router.push('/student/courses' as any)}>
                <Text style={styles.seeAll}>See all</Text>
              </TouchableOpacity>
            </View>
            <FlatList
              horizontal
              data={recommended}
              keyExtractor={(item) => item.id}
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={{ paddingRight: 20 }}
              renderItem={({ item }) => (
                <CourseCard
                  course={item}
                  variant="horizontal"
                  onPress={() => router.push(`/student/courses/${item.id}` as any)}
                />
              )}
            />
          </View>
        )}

        {/* ── Daily Challenge ──────────────────────────────────── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Daily Challenge</Text>
          <TouchableOpacity activeOpacity={0.88} onPress={() => {}}>
            <LinearGradient
              colors={['#6C63FF', '#8B5CF6', '#A855F7']}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={styles.challengeCard}
            >
              <View style={styles.challengeTop}>
                <View style={styles.challengeIcon}>
                  <Text style={{ fontSize: 28 }}>⚡</Text>
                </View>
                <View style={styles.challengeXPBadge}>
                  <Ionicons name="flash" size={12} color={COLORS.accent} />
                  <Text style={styles.challengeXPText}>+{DAILY_CHALLENGE.bonusXP} XP</Text>
                </View>
              </View>
              <Text style={styles.challengeTitle}>{DAILY_CHALLENGE.title}</Text>
              <Text style={styles.challengeDesc}>{DAILY_CHALLENGE.description}</Text>
              <View style={styles.challengeMeta}>
                <View style={styles.challengeMetaItem}>
                  <Ionicons name="time-outline" size={14} color="rgba(255,255,255,0.7)" />
                  <Text style={styles.challengeMetaText}>{DAILY_CHALLENGE.timeLimit}</Text>
                </View>
                <TouchableOpacity style={styles.challengeBtn} onPress={() => {}}>
                  <Text style={styles.challengeBtnText}>Start Challenge</Text>
                  <Ionicons name="arrow-forward" size={16} color={COLORS.primary} />
                </TouchableOpacity>
              </View>
            </LinearGradient>
          </TouchableOpacity>
        </View>

        {/* ── Leaderboard Preview ──────────────────────────────── */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Leaderboard</Text>
            <TouchableOpacity onPress={() => router.push('/student/progress' as any)}>
              <Text style={styles.seeAll}>Full board</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.leaderboardCard}>
            {MOCK_LEADERBOARD.map((entry, idx) => (
              <LeaderboardRow key={entry.user_id} entry={entry} index={idx} />
            ))}
          </View>
        </View>

        {/* ── Recent Achievement ───────────────────────────────── */}
        {badgeInfo && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Recent Achievement</Text>
            <LinearGradient
              colors={['#FFD70022', '#FFD70008']}
              style={styles.achievementCard}
            >
              <View style={styles.achievementIcon}>
                <Text style={{ fontSize: 36 }}>{badgeInfo.icon ?? '🏆'}</Text>
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.achievementTitle}>{badgeInfo.name ?? 'Badge Earned!'}</Text>
                <Text style={styles.achievementDesc}>{badgeInfo.description ?? 'You earned a new badge!'}</Text>
              </View>
              <View style={styles.newBadge}>
                <Text style={styles.newBadgeText}>NEW</Text>
              </View>
            </LinearGradient>
          </View>
        )}

        <View style={{ height: 20 }} />
      </ScrollView>

      {/* XP Pop Animation */}
      {showXPPop && (
        <View style={styles.xpPopContainer}>
          <XPBadge xp={XP_REWARDS.complete_lesson} />
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 20,
  },
  skeletonHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },

  // ── Header ──
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 12,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  headerRight: {
    flexDirection: 'row',
    gap: 8,
  },
  avatarContainer: {
    position: 'relative',
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: COLORS.primary,
  },
  avatarInitialLarge: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '700',
  },
  levelBadgeSmall: {
    position: 'absolute',
    bottom: -4,
    right: -4,
    backgroundColor: COLORS.primary,
    borderRadius: 10,
    minWidth: 20,
    height: 20,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: COLORS.background,
    paddingHorizontal: 4,
  },
  levelBadgeText: {
    color: '#fff',
    fontSize: 9,
    fontWeight: '800',
  },
  greetingText: {
    color: COLORS.textMuted,
    fontSize: 13,
  },
  nameText: {
    color: COLORS.text,
    fontSize: 18,
    fontWeight: '700',
  },
  iconBtn: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: COLORS.surface,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  notifDot: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.accent,
    borderWidth: 2,
    borderColor: COLORS.background,
  },

  // ── Streak ──
  streakWrapper: {
    marginHorizontal: 20,
    marginBottom: 12,
    borderRadius: 16,
    overflow: 'hidden',
    shadowColor: '#FFB84C',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 8,
  },
  streakCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 16,
  },
  streakLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1,
  },
  streakEmoji: {
    fontSize: 28,
  },
  streakTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '800',
  },
  streakSub: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 12,
    marginTop: 2,
  },
  streakRight: {
    alignItems: 'center',
  },
  streakCount: {
    color: '#fff',
    fontSize: 28,
    fontWeight: '900',
  },
  streakDays: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 11,
    marginTop: -4,
  },

  // ── Daily XP Card ──
  card: {
    marginHorizontal: 20,
    marginBottom: 12,
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  xpHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  xpTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  cardTitle: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: '700',
  },
  xpCount: {
    flexDirection: 'row',
    alignItems: 'baseline',
  },
  xpCurrent: {
    color: COLORS.accent,
    fontSize: 18,
    fontWeight: '800',
  },
  xpMuted: {
    color: COLORS.textMuted,
    fontSize: 13,
  },
  goalCompletedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 8,
  },
  goalCompletedText: {
    color: COLORS.accent,
    fontSize: 12,
    fontWeight: '600',
  },

  // ── Section ──
  section: {
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    marginBottom: 12,
  },
  sectionTitle: {
    color: COLORS.text,
    fontSize: 18,
    fontWeight: '700',
    paddingHorizontal: 20,
    marginBottom: 12,
  },
  seeAll: {
    color: COLORS.primary,
    fontSize: 13,
    fontWeight: '600',
  },

  // ── Continue Learning ──
  continueCard: {
    marginHorizontal: 20,
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    overflow: 'hidden',
    flexDirection: 'row',
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  continueThumbnail: {
    width: 100,
    height: 120,
    position: 'relative',
    overflow: 'hidden',
  },
  continueBody: {
    flex: 1,
    padding: 14,
    justifyContent: 'space-between',
  },
  continueTitle: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: '700',
    lineHeight: 20,
    marginBottom: 4,
  },
  continueInstructor: {
    color: COLORS.textMuted,
    fontSize: 12,
    marginBottom: 8,
  },
  continueProgressRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 10,
  },
  continueProgressText: {
    color: COLORS.accent,
    fontSize: 12,
    fontWeight: '700',
    minWidth: 32,
  },
  resumeBtn: {
    borderRadius: 10,
    overflow: 'hidden',
    alignSelf: 'flex-start',
  },
  resumeBtnGrad: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 10,
  },
  resumeBtnText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '700',
  },

  // ── Daily Challenge ──
  challengeCard: {
    marginHorizontal: 20,
    borderRadius: 20,
    padding: 20,
  },
  challengeTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  challengeIcon: {
    width: 52,
    height: 52,
    borderRadius: 14,
    backgroundColor: 'rgba(255,255,255,0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  challengeXPBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  challengeXPText: {
    color: COLORS.accent,
    fontSize: 14,
    fontWeight: '800',
  },
  challengeTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '800',
    marginBottom: 6,
  },
  challengeDesc: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 13,
    lineHeight: 19,
    marginBottom: 16,
  },
  challengeMeta: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  challengeMetaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  challengeMetaText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 13,
  },
  challengeBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#fff',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 10,
  },
  challengeBtnText: {
    color: COLORS.primary,
    fontSize: 13,
    fontWeight: '700',
  },

  // ── Leaderboard ──
  leaderboardCard: {
    marginHorizontal: 20,
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  leaderboardRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
    gap: 10,
  },
  rankBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rankText: {
    fontSize: 12,
    fontWeight: '800',
  },
  leaderboardAvatar: {
    width: 36,
    height: 36,
    borderRadius: 18,
  },
  avatarPlaceholder: {
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarInitial: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  leaderboardName: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: '600',
  },
  leaderboardLevel: {
    color: COLORS.textMuted,
    fontSize: 11,
  },
  xpChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
    backgroundColor: COLORS.surfaceLight,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  xpChipText: {
    color: COLORS.accent,
    fontSize: 12,
    fontWeight: '700',
  },

  // ── Achievement ──
  achievementCard: {
    marginHorizontal: 20,
    borderRadius: 16,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    borderWidth: 1,
    borderColor: 'rgba(255,215,0,0.3)',
  },
  achievementIcon: {
    width: 56,
    height: 56,
    borderRadius: 14,
    backgroundColor: 'rgba(255,215,0,0.1)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  achievementTitle: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 4,
  },
  achievementDesc: {
    color: COLORS.textMuted,
    fontSize: 12,
    lineHeight: 16,
  },
  newBadge: {
    backgroundColor: COLORS.gold,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  newBadgeText: {
    color: '#000',
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.5,
  },

  // ── XP Pop ──
  xpPopContainer: {
    position: 'absolute',
    bottom: 100,
    right: 24,
    alignItems: 'center',
    pointerEvents: 'none',
  },
  xpBadge: {
    backgroundColor: COLORS.accent,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    shadowColor: COLORS.accent,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.5,
    shadowRadius: 8,
    elevation: 10,
  },
  xpBadgeText: {
    color: '#000',
    fontSize: 15,
    fontWeight: '900',
  },
});
