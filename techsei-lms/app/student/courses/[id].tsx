// ============================================================
// TechSei LMS — Course Detail Screen
// ============================================================
import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Image,
  Dimensions,
  Animated,
  ActivityIndicator,
  Share,
  Alert,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useCourseStore } from '../../../stores/courseStore';
import { useAuthStore } from '../../../stores/authStore';
import { ProgressBar } from '../../../components/common/ProgressBar';
import type { Course, Module, Lesson, ContentType } from '../../../types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const HERO_HEIGHT = 260;

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
  error: '#FF6B6B',
};

type TabKey = 'overview' | 'curriculum' | 'reviews';

// ── Mock reviews (in production, fetch from Supabase) ─────────────────────────
const MOCK_REVIEWS = [
  { id: '1', name: 'Priya S.', rating: 5, comment: 'Excellent course! The content is well-structured and easy to follow.', date: '2 weeks ago', avatar: null },
  { id: '2', name: 'Rahul V.', rating: 4, comment: 'Very informative. Loved the hands-on projects and quizzes.', date: '1 month ago', avatar: null },
  { id: '3', name: 'Ananya K.', rating: 5, comment: 'Best investment I\'ve made for my career. Highly recommend!', date: '3 weeks ago', avatar: null },
];

// ── Mock module data (in production this comes from supabase via course.modules) ──
function getMockModules(courseId: string): Module[] {
  return [
    {
      id: 'm1',
      course_id: courseId,
      title: 'Getting Started',
      order_index: 0,
      lessons: [
        { id: 'l1', module_id: 'm1', title: 'Introduction & Setup', content_type: 'video', content_url: null, duration_minutes: 8, order_index: 0, is_preview: true },
        { id: 'l2', module_id: 'm1', title: 'Core Concepts Overview', content_type: 'text', content_url: null, duration_minutes: 5, order_index: 1, is_preview: true },
        { id: 'l3', module_id: 'm1', title: 'Knowledge Check', content_type: 'quiz', content_url: null, duration_minutes: 10, order_index: 2 },
      ],
    },
    {
      id: 'm2',
      course_id: courseId,
      title: 'Core Fundamentals',
      order_index: 1,
      lessons: [
        { id: 'l4', module_id: 'm2', title: 'Deep Dive — Part 1', content_type: 'video', content_url: null, duration_minutes: 20, order_index: 0 },
        { id: 'l5', module_id: 'm2', title: 'Deep Dive — Part 2', content_type: 'video', content_url: null, duration_minutes: 18, order_index: 1 },
        { id: 'l6', module_id: 'm2', title: 'Hands-on Practice', content_type: 'interactive', content_url: null, duration_minutes: 30, order_index: 2 },
      ],
    },
    {
      id: 'm3',
      course_id: courseId,
      title: 'Advanced Topics',
      order_index: 2,
      lessons: [
        { id: 'l7', module_id: 'm3', title: 'Advanced Patterns', content_type: 'video', content_url: null, duration_minutes: 25, order_index: 0 },
        { id: 'l8', module_id: 'm3', title: 'Final Assessment', content_type: 'quiz', content_url: null, duration_minutes: 20, order_index: 1 },
      ],
    },
  ];
}

// ── Content type icon helper ───────────────────────────────────────────────────
function contentTypeIcon(type: ContentType): React.ComponentProps<typeof Ionicons>['name'] {
  switch (type) {
    case 'video': return 'play-circle-outline';
    case 'text': return 'document-text-outline';
    case 'quiz': return 'help-circle-outline';
    case 'interactive': return 'code-slash-outline';
    default: return 'document-outline';
  }
}

function contentTypeColor(type: ContentType): string {
  switch (type) {
    case 'video': return COLORS.primary;
    case 'text': return COLORS.accent;
    case 'quiz': return COLORS.warning;
    case 'interactive': return '#F472B6';
    default: return COLORS.textMuted;
  }
}

// ── Star Rating ───────────────────────────────────────────────────────────────
function StarRating({ rating, size = 14 }: { rating: number; size?: number }) {
  const full = Math.floor(rating);
  const half = rating - full >= 0.5;
  return (
    <View style={{ flexDirection: 'row', gap: 2 }}>
      {[...Array(5)].map((_, i) => (
        <Ionicons
          key={i}
          name={i < full ? 'star' : i === full && half ? 'star-half' : 'star-outline'}
          size={size}
          color={i < full || (i === full && half) ? COLORS.warning : COLORS.textMuted}
        />
      ))}
    </View>
  );
}

// ── Curriculum Module ─────────────────────────────────────────────────────────
function CurriculumModule({
  module,
  isSubscribed,
  completedLessonIds,
  onPressLesson,
}: {
  module: Module;
  isSubscribed: boolean;
  completedLessonIds: Set<string>;
  onPressLesson: (lesson: Lesson) => void;
}) {
  const [expanded, setExpanded] = useState(true);
  const anim = useRef(new Animated.Value(1)).current;
  const totalDuration = module.lessons.reduce((acc, l) => acc + l.duration_minutes, 0);

  const toggle = () => {
    Animated.timing(anim, {
      toValue: expanded ? 0 : 1,
      duration: 250,
      useNativeDriver: false,
    }).start();
    setExpanded((v) => !v);
  };

  return (
    <View style={styles.moduleContainer}>
      <TouchableOpacity style={styles.moduleHeader} onPress={toggle} activeOpacity={0.8}>
        <View style={{ flex: 1 }}>
          <Text style={styles.moduleTitle}>{module.title}</Text>
          <Text style={styles.moduleMeta}>
            {module.lessons.length} lessons · {totalDuration} min
          </Text>
        </View>
        <Ionicons
          name={expanded ? 'chevron-up' : 'chevron-down'}
          size={20}
          color={COLORS.textMuted}
        />
      </TouchableOpacity>
      {expanded && (
        <View style={styles.lessonList}>
          {module.lessons.map((lesson) => {
            const isLocked = !isSubscribed && !lesson.is_preview;
            const isCompleted = completedLessonIds.has(lesson.id);
            return (
              <TouchableOpacity
                key={lesson.id}
                style={styles.lessonRow}
                onPress={() => !isLocked && onPressLesson(lesson)}
                activeOpacity={isLocked ? 1 : 0.75}
              >
                <View style={[styles.lessonTypeIcon, { backgroundColor: `${contentTypeColor(lesson.content_type)}22` }]}>
                  <Ionicons
                    name={contentTypeIcon(lesson.content_type)}
                    size={16}
                    color={contentTypeColor(lesson.content_type)}
                  />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={[styles.lessonTitle, isLocked && { color: COLORS.textMuted }]} numberOfLines={1}>
                    {lesson.title}
                  </Text>
                  <Text style={styles.lessonDuration}>{lesson.duration_minutes} min</Text>
                </View>
                {lesson.is_preview && !isLocked && (
                  <View style={styles.previewBadge}>
                    <Text style={styles.previewBadgeText}>Preview</Text>
                  </View>
                )}
                {isCompleted ? (
                  <Ionicons name="checkmark-circle" size={20} color={COLORS.accent} />
                ) : isLocked ? (
                  <Ionicons name="lock-closed" size={16} color={COLORS.textMuted} />
                ) : null}
              </TouchableOpacity>
            );
          })}
        </View>
      )}
    </View>
  );
}

// ── Review Card ───────────────────────────────────────────────────────────────
function ReviewCard({ review }: { review: typeof MOCK_REVIEWS[0] }) {
  return (
    <View style={styles.reviewCard}>
      <View style={styles.reviewHeader}>
        <View style={styles.reviewAvatar}>
          <Text style={styles.reviewAvatarText}>{review.name.charAt(0)}</Text>
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.reviewName}>{review.name}</Text>
          <Text style={styles.reviewDate}>{review.date}</Text>
        </View>
        <StarRating rating={review.rating} size={13} />
      </View>
      <Text style={styles.reviewComment}>{review.comment}</Text>
    </View>
  );
}

// ── Main Screen ───────────────────────────────────────────────────────────────
export default function CourseDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const { user, profile } = useAuthStore();
  const {
    courses,
    enrolledCourses,
    lessonProgress,
    isLoading,
    enrollInCourse,
    fetchCourseProgress,
    getCourseProgress,
    setCurrentCourse,
  } = useCourseStore();

  const [activeTab, setActiveTab] = useState<TabKey>('overview');
  const [enrolling, setEnrolling] = useState(false);
  const scrollY = useRef(new Animated.Value(0)).current;

  const course = courses.find((c) => c.id === id) ?? enrolledCourses.find((c) => c.id === id);
  const isEnrolled = enrolledCourses.some((c) => c.id === id);
  const isSubscribed = (profile?.subscription_tier ?? 'free') !== 'free';
  const canAccess = !course?.is_premium || isSubscribed || isEnrolled;
  const progress = course ? getCourseProgress(course.id) : 0;

  const modules = course ? getMockModules(course.id) : [];
  const totalLessons = modules.reduce((acc, m) => acc + m.lessons.length, 0);

  const completedLessonIds = new Set(
    Object.values(lessonProgress)
      .filter((p) => p.completed)
      .map((p) => p.lesson_id)
  );

  useEffect(() => {
    if (course && user) {
      setCurrentCourse(course);
      if (isEnrolled) {
        fetchCourseProgress(course.id, user.id);
      }
    }
  }, [course, user, isEnrolled, setCurrentCourse, fetchCourseProgress]);

  const handleEnroll = useCallback(async () => {
    if (!course || !user) return;
    if (course.is_premium && !isSubscribed) {
      Alert.alert(
        'Premium Course',
        'Subscribe to TechSei Pro to access this course.',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Upgrade', onPress: () => {} },
        ]
      );
      return;
    }
    try {
      setEnrolling(true);
      await enrollInCourse(course.id, user.id);
    } catch (e) {
      Alert.alert('Error', 'Failed to enroll. Please try again.');
    } finally {
      setEnrolling(false);
    }
  }, [course, user, isSubscribed, enrollInCourse]);

  const handleShare = useCallback(async () => {
    if (!course) return;
    await Share.share({ message: `Check out "${course.title}" on TechSei LMS!` });
  }, [course]);

  const handlePressLesson = (lesson: Lesson) => {
    router.push(`/student/courses/lesson/${lesson.id}` as any);
  };

  const heroOpacity = scrollY.interpolate({ inputRange: [0, HERO_HEIGHT * 0.5], outputRange: [1, 0], extrapolate: 'clamp' });
  const navBgOpacity = scrollY.interpolate({ inputRange: [HERO_HEIGHT * 0.4, HERO_HEIGHT * 0.7], outputRange: [0, 1], extrapolate: 'clamp' });

  if (!course) {
    return (
      <SafeAreaView style={[styles.root, { alignItems: 'center', justifyContent: 'center' }]}>
        <ActivityIndicator color={COLORS.primary} size="large" />
        <Text style={[styles.textMuted, { marginTop: 16 }]}>Loading course...</Text>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.root}>
      {/* ── Sticky Nav Background ────────────────────────────── */}
      <Animated.View style={[styles.navBg, { opacity: navBgOpacity }]} />

      {/* ── Nav Bar ──────────────────────────────────────────── */}
      <SafeAreaView style={styles.navBar} edges={['top']}>
        <TouchableOpacity onPress={() => router.back()} style={styles.navBtn}>
          <Ionicons name="arrow-back" size={22} color={COLORS.text} />
        </TouchableOpacity>
        <Animated.Text style={[styles.navTitle, { opacity: navBgOpacity }]} numberOfLines={1}>
          {course.title}
        </Animated.Text>
        <TouchableOpacity onPress={handleShare} style={styles.navBtn}>
          <Ionicons name="share-outline" size={22} color={COLORS.text} />
        </TouchableOpacity>
      </SafeAreaView>

      <Animated.ScrollView
        style={styles.scroll}
        onScroll={Animated.event([{ nativeEvent: { contentOffset: { y: scrollY } } }], { useNativeDriver: false })}
        scrollEventThrottle={16}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{ paddingBottom: 120 }}
      >
        {/* ── Hero ────────────────────────────────────────────── */}
        <View style={styles.hero}>
          <Animated.View style={[StyleSheet.absoluteFill, { opacity: heroOpacity }]}>
            {course.thumbnail_url ? (
              <Image source={{ uri: course.thumbnail_url }} style={StyleSheet.absoluteFill} resizeMode="cover" />
            ) : (
              <LinearGradient colors={['#6C63FF', '#A855F7', '#EC4899']} style={StyleSheet.absoluteFill} />
            )}
          </Animated.View>
          <LinearGradient
            colors={['transparent', 'rgba(10,10,26,0.85)', COLORS.background]}
            style={styles.heroGradient}
          >
            {course.is_premium && (
              <View style={styles.proBadge}>
                <Ionicons name="star" size={12} color="#000" />
                <Text style={styles.proBadgeText}>PRO</Text>
              </View>
            )}
          </LinearGradient>
        </View>

        {/* ── Course Info ──────────────────────────────────────── */}
        <View style={styles.infoSection}>
          <Text style={styles.courseTitle}>{course.title}</Text>

          <View style={styles.instructorRow}>
            <View style={styles.instructorAvatar}>
              <Ionicons name="person" size={16} color={COLORS.primary} />
            </View>
            <Text style={styles.instructorName}>{course.instructor_name ?? 'TechSei Instructor'}</Text>
          </View>

          <View style={styles.ratingRow}>
            <StarRating rating={course.rating} size={15} />
            <Text style={styles.ratingNumber}>{course.rating.toFixed(1)}</Text>
            <Text style={styles.ratingCount}>({course.total_students.toLocaleString()} students)</Text>
          </View>

          {/* ── Stats ──────────────────────────────────────────── */}
          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <Ionicons name="time-outline" size={16} color={COLORS.primary} />
              <Text style={styles.statLabel}>{course.duration_hours}h</Text>
              <Text style={styles.statSub}>Duration</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Ionicons name="layers-outline" size={16} color={COLORS.accent} />
              <Text style={styles.statLabel}>{totalLessons}</Text>
              <Text style={styles.statSub}>Lessons</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Ionicons name="bar-chart-outline" size={16} color={COLORS.warning} />
              <Text style={styles.statLabel}>Beginner</Text>
              <Text style={styles.statSub}>Level</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Ionicons name="people-outline" size={16} color={COLORS.textMuted} />
              <Text style={styles.statLabel}>{(course.total_students / 1000).toFixed(1)}k</Text>
              <Text style={styles.statSub}>Enrolled</Text>
            </View>
          </View>

          {/* ── Progress if enrolled ────────────────────────────── */}
          {isEnrolled && (
            <View style={styles.progressSection}>
              <View style={styles.progressHeader}>
                <Text style={styles.progressLabel}>Your Progress</Text>
                <Text style={styles.progressPct}>{progress}%</Text>
              </View>
              <ProgressBar progress={progress} colorStart="#43E97B" colorEnd="#38F9D7" height={8} animated />
            </View>
          )}
        </View>

        {/* ── Tab Switcher ─────────────────────────────────────── */}
        <View style={styles.tabRow}>
          {(['overview', 'curriculum', 'reviews'] as TabKey[]).map((tab) => (
            <TouchableOpacity
              key={tab}
              style={[styles.tab, activeTab === tab && styles.tabActive]}
              onPress={() => setActiveTab(tab)}
            >
              <Text style={[styles.tabText, activeTab === tab && styles.tabTextActive]}>
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </Text>
              {activeTab === tab && <View style={styles.tabUnderline} />}
            </TouchableOpacity>
          ))}
        </View>

        {/* ── Overview Tab ─────────────────────────────────────── */}
        {activeTab === 'overview' && (
          <View style={styles.tabContent}>
            <Text style={styles.sectionTitle}>About this Course</Text>
            <Text style={styles.description}>{course.description}</Text>

            <Text style={styles.sectionTitle}>What You'll Learn</Text>
            {[
              'Build real-world projects from scratch',
              'Understand core concepts and best practices',
              'Master industry-standard tools and workflows',
              'Deploy production-ready applications',
              'Solve common coding challenges confidently',
            ].map((item, i) => (
              <View key={i} style={styles.bulletRow}>
                <View style={styles.bulletDot}>
                  <Ionicons name="checkmark" size={12} color={COLORS.accent} />
                </View>
                <Text style={styles.bulletText}>{item}</Text>
              </View>
            ))}

            <Text style={styles.sectionTitle}>Requirements</Text>
            {[
              'Basic understanding of computers',
              'A device with internet connection',
              'Eagerness to learn — no prior experience needed!',
            ].map((item, i) => (
              <View key={i} style={styles.bulletRow}>
                <View style={[styles.bulletDot, { backgroundColor: `${COLORS.warning}22` }]}>
                  <Ionicons name="arrow-forward" size={12} color={COLORS.warning} />
                </View>
                <Text style={styles.bulletText}>{item}</Text>
              </View>
            ))}
          </View>
        )}

        {/* ── Curriculum Tab ───────────────────────────────────── */}
        {activeTab === 'curriculum' && (
          <View style={styles.tabContent}>
            <View style={styles.curriculumHeader}>
              <Text style={styles.curriculumMeta}>
                {modules.length} modules · {totalLessons} lessons · {course.duration_hours}h total
              </Text>
            </View>
            {modules.map((mod) => (
              <CurriculumModule
                key={mod.id}
                module={mod}
                isSubscribed={canAccess}
                completedLessonIds={completedLessonIds}
                onPressLesson={handlePressLesson}
              />
            ))}
          </View>
        )}

        {/* ── Reviews Tab ──────────────────────────────────────── */}
        {activeTab === 'reviews' && (
          <View style={styles.tabContent}>
            {/* Rating breakdown */}
            <View style={styles.ratingBreakdown}>
              <View style={styles.ratingBigNumber}>
                <Text style={styles.ratingBig}>{course.rating.toFixed(1)}</Text>
                <StarRating rating={course.rating} size={18} />
                <Text style={styles.ratingMeta}>{course.total_students.toLocaleString()} ratings</Text>
              </View>
              <View style={styles.ratingBars}>
                {[5, 4, 3, 2, 1].map((star) => {
                  const pct = star === 5 ? 70 : star === 4 ? 20 : star === 3 ? 7 : star === 2 ? 2 : 1;
                  return (
                    <View key={star} style={styles.ratingBarRow}>
                      <Text style={styles.ratingBarLabel}>{star}</Text>
                      <Ionicons name="star" size={11} color={COLORS.warning} />
                      <View style={{ flex: 1 }}>
                        <ProgressBar progress={pct} height={6} colorStart="#FFB84C" colorEnd="#FF6B35" animated={false} />
                      </View>
                      <Text style={styles.ratingBarPct}>{pct}%</Text>
                    </View>
                  );
                })}
              </View>
            </View>

            {MOCK_REVIEWS.map((review) => (
              <ReviewCard key={review.id} review={review} />
            ))}
          </View>
        )}
      </Animated.ScrollView>

      {/* ── Sticky Bottom Bar ────────────────────────────────── */}
      <View style={styles.stickyBar}>
        {course.is_premium && !isSubscribed && (
          <Text style={styles.stickyPrice}>From ₹499/month</Text>
        )}
        <TouchableOpacity
          style={[styles.ctaBtn, enrolling && { opacity: 0.7 }]}
          onPress={isEnrolled ? () => handlePressLesson(modules[0]?.lessons[0]) : handleEnroll}
          disabled={enrolling}
          activeOpacity={0.85}
        >
          <LinearGradient
            colors={['#6C63FF', '#8B5CF6']}
            style={styles.ctaBtnGrad}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            {enrolling ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <>
                <Ionicons
                  name={isEnrolled ? 'play' : course.is_premium && !isSubscribed ? 'card-outline' : 'add-circle-outline'}
                  size={20}
                  color="#fff"
                />
                <Text style={styles.ctaBtnText}>
                  {isEnrolled
                    ? progress > 0 ? 'Continue Learning' : 'Start Learning'
                    : course.is_premium && !isSubscribed
                    ? 'Upgrade to Pro'
                    : 'Enroll Now — Free'}
                </Text>
              </>
            )}
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
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
  textMuted: {
    color: COLORS.textMuted,
    fontSize: 14,
  },

  // ── Nav ──
  navBg: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: COLORS.surface,
    zIndex: 10,
    height: 90,
  },
  navBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 20,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingBottom: 10,
  },
  navBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(20,20,40,0.7)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  navTitle: {
    flex: 1,
    color: COLORS.text,
    fontSize: 15,
    fontWeight: '700',
    textAlign: 'center',
    marginHorizontal: 8,
  },

  // ── Hero ──
  hero: {
    height: HERO_HEIGHT,
    position: 'relative',
  },
  heroGradient: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'flex-end',
    padding: 20,
  },
  proBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#FFD700',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  proBadgeText: {
    color: '#000',
    fontSize: 11,
    fontWeight: '800',
  },

  // ── Info ──
  infoSection: {
    paddingHorizontal: 20,
    paddingTop: 12,
  },
  courseTitle: {
    color: COLORS.text,
    fontSize: 22,
    fontWeight: '800',
    lineHeight: 30,
    marginBottom: 12,
  },
  instructorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 10,
  },
  instructorAvatar: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: `${COLORS.primary}22`,
    alignItems: 'center',
    justifyContent: 'center',
  },
  instructorName: {
    color: COLORS.textMuted,
    fontSize: 14,
    fontWeight: '600',
  },
  ratingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 16,
  },
  ratingNumber: {
    color: COLORS.warning,
    fontSize: 15,
    fontWeight: '700',
  },
  ratingCount: {
    color: COLORS.textMuted,
    fontSize: 13,
  },

  // ── Stats ──
  statsRow: {
    flexDirection: 'row',
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: COLORS.border,
    marginBottom: 16,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
    gap: 4,
  },
  statLabel: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: '700',
    marginTop: 4,
  },
  statSub: {
    color: COLORS.textMuted,
    fontSize: 11,
  },
  statDivider: {
    width: 1,
    backgroundColor: COLORS.border,
    marginVertical: 4,
  },

  // ── Progress ──
  progressSection: {
    marginBottom: 8,
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  progressLabel: {
    color: COLORS.textMuted,
    fontSize: 13,
    fontWeight: '600',
  },
  progressPct: {
    color: COLORS.accent,
    fontSize: 13,
    fontWeight: '700',
  },

  // ── Tabs ──
  tabRow: {
    flexDirection: 'row',
    marginHorizontal: 20,
    marginTop: 8,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  tab: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
    position: 'relative',
  },
  tabActive: {},
  tabText: {
    color: COLORS.textMuted,
    fontSize: 14,
    fontWeight: '600',
  },
  tabTextActive: {
    color: COLORS.primary,
  },
  tabUnderline: {
    position: 'absolute',
    bottom: -1,
    left: '20%',
    right: '20%',
    height: 2,
    backgroundColor: COLORS.primary,
    borderRadius: 1,
  },
  tabContent: {
    padding: 20,
  },

  // ── Overview ──
  sectionTitle: {
    color: COLORS.text,
    fontSize: 17,
    fontWeight: '700',
    marginBottom: 12,
    marginTop: 8,
  },
  description: {
    color: COLORS.textMuted,
    fontSize: 14,
    lineHeight: 22,
    marginBottom: 20,
  },
  bulletRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginBottom: 10,
  },
  bulletDot: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: `${COLORS.accent}22`,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 1,
  },
  bulletText: {
    color: COLORS.textMuted,
    fontSize: 14,
    lineHeight: 20,
    flex: 1,
  },

  // ── Curriculum ──
  curriculumHeader: {
    marginBottom: 16,
  },
  curriculumMeta: {
    color: COLORS.textMuted,
    fontSize: 13,
  },
  moduleContainer: {
    marginBottom: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 14,
    overflow: 'hidden',
  },
  moduleHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: COLORS.surface,
    gap: 8,
  },
  moduleTitle: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 3,
  },
  moduleMeta: {
    color: COLORS.textMuted,
    fontSize: 12,
  },
  lessonList: {
    backgroundColor: COLORS.background,
  },
  lessonRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    gap: 10,
  },
  lessonTypeIcon: {
    width: 32,
    height: 32,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  lessonTitle: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: '600',
    marginBottom: 3,
  },
  lessonDuration: {
    color: COLORS.textMuted,
    fontSize: 11,
  },
  previewBadge: {
    backgroundColor: `${COLORS.accent}22`,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  previewBadgeText: {
    color: COLORS.accent,
    fontSize: 10,
    fontWeight: '700',
  },

  // ── Reviews ──
  ratingBreakdown: {
    flexDirection: 'row',
    marginBottom: 20,
    gap: 16,
  },
  ratingBigNumber: {
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 80,
    gap: 4,
  },
  ratingBig: {
    color: COLORS.text,
    fontSize: 42,
    fontWeight: '900',
  },
  ratingMeta: {
    color: COLORS.textMuted,
    fontSize: 11,
    textAlign: 'center',
    marginTop: 4,
  },
  ratingBars: {
    flex: 1,
    gap: 6,
  },
  ratingBarRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  ratingBarLabel: {
    color: COLORS.textMuted,
    fontSize: 12,
    fontWeight: '600',
    width: 10,
  },
  ratingBarPct: {
    color: COLORS.textMuted,
    fontSize: 11,
    width: 28,
    textAlign: 'right',
  },
  reviewCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 14,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  reviewHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 10,
  },
  reviewAvatar: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  reviewAvatarText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  reviewName: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 2,
  },
  reviewDate: {
    color: COLORS.textMuted,
    fontSize: 11,
  },
  reviewComment: {
    color: COLORS.textMuted,
    fontSize: 13,
    lineHeight: 20,
  },

  // ── Sticky Bar ──
  stickyBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: COLORS.surface,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    paddingHorizontal: 20,
    paddingVertical: 14,
    paddingBottom: 28,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 20,
  },
  stickyPrice: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: '700',
    minWidth: 110,
  },
  ctaBtn: {
    flex: 1,
    borderRadius: 14,
    overflow: 'hidden',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 10,
    elevation: 8,
  },
  ctaBtnGrad: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    height: 52,
    borderRadius: 14,
  },
  ctaBtnText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '700',
  },
});
