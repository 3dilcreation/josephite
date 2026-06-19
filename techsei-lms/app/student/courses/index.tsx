// ============================================================
// TechSei LMS — Courses Browse Screen
// ============================================================
import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  TextInput,
  Animated,
  RefreshControl,
  Dimensions,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useCourseStore } from '../../../stores/courseStore';
import { CourseCard } from '../../../components/common/CourseCard';
import type { Course, CourseCategory } from '../../../types';

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
};

// ── Category chips ────────────────────────────────────────────────────────────
interface CategoryItem {
  label: string;
  value: string;
  icon: React.ComponentProps<typeof Ionicons>['name'];
}

const CATEGORIES: CategoryItem[] = [
  { label: 'All', value: 'all', icon: 'apps-outline' },
  { label: 'Web Dev', value: 'web-development', icon: 'globe-outline' },
  { label: 'Data Science', value: 'data-science', icon: 'analytics-outline' },
  { label: 'Mobile', value: 'mobile-development', icon: 'phone-portrait-outline' },
  { label: 'AI/ML', value: 'ai-ml', icon: 'hardware-chip-outline' },
  { label: 'Cybersecurity', value: 'cybersecurity', icon: 'shield-checkmark-outline' },
  { label: 'Cloud', value: 'cloud', icon: 'cloud-outline' },
];

type SortOption = 'popular' | 'newest' | 'rating';

// ── Skeleton ──────────────────────────────────────────────────────────────────
function SkeletonCard() {
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
  const cardWidth = (SCREEN_WIDTH - 48) / 2;
  return (
    <Animated.View style={[styles.skeletonCard, { width: cardWidth, opacity }]}>
      <View style={styles.skeletonThumb} />
      <View style={{ padding: 10, gap: 6 }}>
        <View style={[styles.skeletonLine, { width: '90%', height: 14 }]} />
        <View style={[styles.skeletonLine, { width: '60%', height: 11 }]} />
        <View style={[styles.skeletonLine, { width: '40%', height: 11 }]} />
      </View>
    </Animated.View>
  );
}

function CourseGridSkeleton() {
  return (
    <View style={styles.skeletonGrid}>
      {[...Array(6)].map((_, i) => <SkeletonCard key={i} />)}
    </View>
  );
}

// ── Empty State ───────────────────────────────────────────────────────────────
function EmptyState({ onReset }: { onReset: () => void }) {
  return (
    <View style={styles.emptyState}>
      <View style={styles.emptyIcon}>
        <Ionicons name="search-outline" size={48} color={COLORS.textMuted} />
      </View>
      <Text style={styles.emptyTitle}>No Courses Found</Text>
      <Text style={styles.emptyDesc}>Try adjusting your filters or search query.</Text>
      <TouchableOpacity style={styles.emptyBtn} onPress={onReset}>
        <Text style={styles.emptyBtnText}>Clear Filters</Text>
      </TouchableOpacity>
    </View>
  );
}

// ── Main Screen ───────────────────────────────────────────────────────────────
export default function CoursesScreen() {
  const router = useRouter();
  const { courses, enrolledCourses, isLoading, fetchCourses, getCourseProgress } = useCourseStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [sortBy, setSortBy] = useState<SortOption>('popular');
  const [premiumFilter, setPremiumFilter] = useState<'all' | 'free' | 'pro'>('all');
  const [showSortMenu, setShowSortMenu] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const enrolledIds = new Set(enrolledCourses.map((c) => c.id));

  useEffect(() => {
    fetchCourses(selectedCategory !== 'all' ? selectedCategory : undefined);
  }, [selectedCategory, fetchCourses]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchCourses(selectedCategory !== 'all' ? selectedCategory : undefined);
    setRefreshing(false);
  }, [selectedCategory, fetchCourses]);

  const resetFilters = () => {
    setSearchQuery('');
    setSelectedCategory('all');
    setSortBy('popular');
    setPremiumFilter('all');
  };

  // Filter and sort
  const filteredCourses = courses
    .filter((c) => {
      const matchesSearch =
        searchQuery.trim() === '' ||
        c.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        c.instructor_name?.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesPremium =
        premiumFilter === 'all' ||
        (premiumFilter === 'free' && !c.is_premium) ||
        (premiumFilter === 'pro' && c.is_premium);
      return matchesSearch && matchesPremium;
    })
    .sort((a, b) => {
      if (sortBy === 'popular') return b.total_students - a.total_students;
      if (sortBy === 'rating') return b.rating - a.rating;
      if (sortBy === 'newest') return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      return 0;
    });

  const SORT_LABELS: Record<SortOption, string> = {
    popular: 'Most Popular',
    newest: 'Newest First',
    rating: 'Top Rated',
  };

  return (
    <SafeAreaView style={styles.root} edges={['top']}>
      {/* ── Header ──────────────────────────────────────────────── */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Browse Courses</Text>
        <Text style={styles.headerSub}>{filteredCourses.length} courses available</Text>
      </View>

      {/* ── Search Bar ──────────────────────────────────────────── */}
      <View style={styles.searchRow}>
        <View style={styles.searchBar}>
          <Ionicons name="search-outline" size={18} color={COLORS.textMuted} style={{ marginRight: 8 }} />
          <TextInput
            style={styles.searchInput}
            placeholder="Search courses..."
            placeholderTextColor={COLORS.textMuted}
            value={searchQuery}
            onChangeText={setSearchQuery}
            returnKeyType="search"
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity onPress={() => setSearchQuery('')}>
              <Ionicons name="close-circle" size={18} color={COLORS.textMuted} />
            </TouchableOpacity>
          )}
        </View>
        <TouchableOpacity
          style={styles.filterBtn}
          onPress={() => setShowSortMenu((v) => !v)}
        >
          <Ionicons name="options-outline" size={20} color={COLORS.primary} />
        </TouchableOpacity>
      </View>

      {/* ── Sort menu dropdown ───────────────────────────────────── */}
      {showSortMenu && (
        <View style={styles.sortMenu}>
          {(['popular', 'newest', 'rating'] as SortOption[]).map((opt) => (
            <TouchableOpacity
              key={opt}
              style={[styles.sortOption, sortBy === opt && styles.sortOptionActive]}
              onPress={() => { setSortBy(opt); setShowSortMenu(false); }}
            >
              <Text style={[styles.sortOptionText, sortBy === opt && { color: COLORS.primary }]}>
                {SORT_LABELS[opt]}
              </Text>
              {sortBy === opt && <Ionicons name="checkmark" size={16} color={COLORS.primary} />}
            </TouchableOpacity>
          ))}
        </View>
      )}

      {/* ── Category Chips ──────────────────────────────────────── */}
      <FlatList
        horizontal
        data={CATEGORIES}
        keyExtractor={(item) => item.value}
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.categoryList}
        style={styles.categoryRow}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={[styles.chip, selectedCategory === item.value && styles.chipActive]}
            onPress={() => setSelectedCategory(item.value)}
          >
            <Ionicons
              name={item.icon}
              size={14}
              color={selectedCategory === item.value ? '#fff' : COLORS.textMuted}
            />
            <Text style={[styles.chipText, selectedCategory === item.value && styles.chipTextActive]}>
              {item.label}
            </Text>
          </TouchableOpacity>
        )}
      />

      {/* ── Free / Pro Toggle ───────────────────────────────────── */}
      <View style={styles.premiumToggleRow}>
        {(['all', 'free', 'pro'] as const).map((opt) => (
          <TouchableOpacity
            key={opt}
            style={[styles.premiumToggle, premiumFilter === opt && styles.premiumToggleActive]}
            onPress={() => setPremiumFilter(opt)}
          >
            <Text style={[styles.premiumToggleText, premiumFilter === opt && { color: '#fff' }]}>
              {opt === 'all' ? 'All' : opt === 'free' ? '✓ Free' : '★ Pro'}
            </Text>
          </TouchableOpacity>
        ))}
        <View style={{ flex: 1, alignItems: 'flex-end' }}>
          <Text style={styles.sortLabel}>
            <Ionicons name="swap-vertical-outline" size={13} color={COLORS.textMuted} />{' '}
            {SORT_LABELS[sortBy]}
          </Text>
        </View>
      </View>

      {/* ── Course Grid ─────────────────────────────────────────── */}
      {isLoading && courses.length === 0 ? (
        <CourseGridSkeleton />
      ) : filteredCourses.length === 0 ? (
        <EmptyState onReset={resetFilters} />
      ) : (
        <FlatList
          data={filteredCourses}
          keyExtractor={(item) => item.id}
          numColumns={2}
          contentContainerStyle={styles.grid}
          columnWrapperStyle={styles.gridRow}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor={COLORS.primary}
              colors={[COLORS.primary]}
            />
          }
          renderItem={({ item }) => (
            <CourseCard
              course={item}
              variant="grid"
              enrollmentProgress={enrolledIds.has(item.id) ? getCourseProgress(item.id) : undefined}
              onPress={() => router.push(`/student/courses/${item.id}` as any)}
            />
          )}
          ListFooterComponent={
            isLoading ? (
              <View style={{ padding: 20, alignItems: 'center' }}>
                <ActivityIndicator color={COLORS.primary} />
              </View>
            ) : null
          }
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: COLORS.background,
  },

  // ── Header ──
  header: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 8,
  },
  headerTitle: {
    color: COLORS.text,
    fontSize: 24,
    fontWeight: '800',
  },
  headerSub: {
    color: COLORS.textMuted,
    fontSize: 13,
    marginTop: 2,
  },

  // ── Search ──
  searchRow: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    gap: 10,
    marginBottom: 12,
    marginTop: 8,
  },
  searchBar: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderRadius: 12,
    paddingHorizontal: 14,
    height: 46,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  searchInput: {
    flex: 1,
    color: COLORS.text,
    fontSize: 14,
    height: '100%',
  },
  filterBtn: {
    width: 46,
    height: 46,
    borderRadius: 12,
    backgroundColor: COLORS.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  // ── Sort Menu ──
  sortMenu: {
    position: 'absolute',
    top: 168,
    right: 20,
    backgroundColor: COLORS.surface,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
    zIndex: 100,
    elevation: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    overflow: 'hidden',
  },
  sortOption: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 24,
  },
  sortOptionActive: {
    backgroundColor: COLORS.surfaceLight,
  },
  sortOptionText: {
    color: COLORS.text,
    fontSize: 14,
  },

  // ── Categories ──
  categoryRow: {
    marginBottom: 10,
  },
  categoryList: {
    paddingHorizontal: 20,
    gap: 8,
  },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  chipActive: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  chipText: {
    color: COLORS.textMuted,
    fontSize: 13,
    fontWeight: '600',
  },
  chipTextActive: {
    color: '#fff',
  },

  // ── Premium Toggle ──
  premiumToggleRow: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    gap: 8,
    marginBottom: 14,
    alignItems: 'center',
  },
  premiumToggle: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  premiumToggleActive: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  premiumToggleText: {
    color: COLORS.textMuted,
    fontSize: 12,
    fontWeight: '600',
  },
  sortLabel: {
    color: COLORS.textMuted,
    fontSize: 12,
  },

  // ── Grid ──
  grid: {
    paddingHorizontal: 16,
    paddingBottom: 100,
  },
  gridRow: {
    gap: 12,
    marginBottom: 0,
  },

  // ── Skeleton ──
  skeletonGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 16,
    gap: 12,
  },
  skeletonCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  skeletonThumb: {
    width: '100%',
    height: 110,
    backgroundColor: COLORS.surfaceLight,
  },
  skeletonLine: {
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 6,
  },

  // ── Empty State ──
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 40,
    paddingTop: 60,
  },
  emptyIcon: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: COLORS.surface,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  emptyTitle: {
    color: COLORS.text,
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 8,
    textAlign: 'center',
  },
  emptyDesc: {
    color: COLORS.textMuted,
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 24,
  },
  emptyBtn: {
    paddingHorizontal: 24,
    paddingVertical: 12,
    backgroundColor: COLORS.primary,
    borderRadius: 12,
  },
  emptyBtnText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '700',
  },
});
