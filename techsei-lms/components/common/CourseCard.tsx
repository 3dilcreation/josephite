// ============================================================
// TechSei LMS — Reusable Course Card Component
// ============================================================
import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Image,
  StyleSheet,
  ViewStyle,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import type { Course } from '../../types';
import { ProgressBar } from './ProgressBar';

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
};

interface CourseCardProps {
  course: Course;
  onPress: () => void;
  variant?: 'grid' | 'horizontal' | 'featured';
  enrollmentProgress?: number;
  style?: ViewStyle;
}

// ── Star Rating ───────────────────────────────────────────────────────────────
function StarRating({ rating }: { rating: number }) {
  const stars = [];
  const fullStars = Math.floor(rating);
  const hasHalf = rating - fullStars >= 0.5;

  for (let i = 0; i < 5; i++) {
    if (i < fullStars) {
      stars.push(<Ionicons key={i} name="star" size={11} color={COLORS.warning} />);
    } else if (i === fullStars && hasHalf) {
      stars.push(<Ionicons key={i} name="star-half" size={11} color={COLORS.warning} />);
    } else {
      stars.push(<Ionicons key={i} name="star-outline" size={11} color={COLORS.textMuted} />);
    }
  }
  return <View style={styles.starsRow}>{stars}</View>;
}

// ── Price / Tier Badge ────────────────────────────────────────────────────────
function PriceTag({ isPremium }: { isPremium: boolean }) {
  if (isPremium) {
    return (
      <View style={styles.proBadge}>
        <Ionicons name="star" size={9} color="#000" />
        <Text style={styles.proBadgeText}>PRO</Text>
      </View>
    );
  }
  return (
    <View style={styles.freeBadge}>
      <Text style={styles.freeBadgeText}>FREE</Text>
    </View>
  );
}

// ── Thumbnail Placeholder ─────────────────────────────────────────────────────
function ThumbnailPlaceholder({ size }: { size: 'small' | 'large' }) {
  return (
    <LinearGradient
      colors={['#6C63FF', '#A855F7']}
      style={StyleSheet.absoluteFill}
    >
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
        <Ionicons name="book" size={size === 'large' ? 60 : 32} color="#FFFFFF30" />
      </View>
    </LinearGradient>
  );
}

// ── Grid Variant ──────────────────────────────────────────────────────────────
function GridCard({ course, onPress, enrollmentProgress }: CourseCardProps) {
  const cardWidth = (SCREEN_WIDTH - 48) / 2;
  return (
    <TouchableOpacity
      onPress={onPress}
      style={[styles.gridCard, { width: cardWidth }]}
      activeOpacity={0.85}
    >
      <View style={styles.gridThumbnailContainer}>
        {course.thumbnail_url ? (
          <Image
            source={{ uri: course.thumbnail_url }}
            style={styles.gridThumbnail}
            resizeMode="cover"
          />
        ) : (
          <View style={styles.gridThumbnail}>
            <ThumbnailPlaceholder size="small" />
          </View>
        )}
        <View style={styles.thumbnailBadge}>
          <PriceTag isPremium={course.is_premium} />
        </View>
      </View>
      <View style={styles.gridBody}>
        <Text style={styles.gridTitle} numberOfLines={2}>
          {course.title}
        </Text>
        <Text style={styles.instructorText} numberOfLines={1}>
          {course.instructor_name ?? 'TechSei Instructor'}
        </Text>
        <View style={styles.ratingRow}>
          <StarRating rating={course.rating} />
          <Text style={styles.ratingText}> {course.rating.toFixed(1)}</Text>
        </View>
        <View style={styles.gridStats}>
          <View style={styles.statChip}>
            <Ionicons name="time-outline" size={10} color={COLORS.textMuted} />
            <Text style={styles.statChipText}>{course.duration_hours}h</Text>
          </View>
          <View style={styles.statChip}>
            <Ionicons name="people-outline" size={10} color={COLORS.textMuted} />
            <Text style={styles.statChipText}>{formatStudentCount(course.total_students)}</Text>
          </View>
        </View>
        {enrollmentProgress !== undefined && (
          <View style={{ marginTop: 8 }}>
            <ProgressBar
              progress={enrollmentProgress}
              height={4}
              colorStart="#43E97B"
              colorEnd="#38F9D7"
            />
          </View>
        )}
      </View>
    </TouchableOpacity>
  );
}

// ── Horizontal Variant ────────────────────────────────────────────────────────
function HorizontalCard({ course, onPress, enrollmentProgress }: CourseCardProps) {
  return (
    <TouchableOpacity onPress={onPress} style={styles.horizontalCard} activeOpacity={0.85}>
      <View style={styles.horizontalThumbnailContainer}>
        {course.thumbnail_url ? (
          <Image
            source={{ uri: course.thumbnail_url }}
            style={styles.horizontalThumbnail}
            resizeMode="cover"
          />
        ) : (
          <View style={styles.horizontalThumbnail}>
            <ThumbnailPlaceholder size="small" />
          </View>
        )}
        <View style={styles.thumbnailBadgeSmall}>
          <PriceTag isPremium={course.is_premium} />
        </View>
      </View>
      <View style={styles.horizontalBody}>
        <Text style={styles.horizontalTitle} numberOfLines={2}>
          {course.title}
        </Text>
        <Text style={styles.instructorText} numberOfLines={1}>
          {course.instructor_name ?? 'TechSei Instructor'}
        </Text>
        <View style={styles.ratingRow}>
          <StarRating rating={course.rating} />
          <Text style={styles.ratingText}> {course.rating.toFixed(1)}</Text>
          <Text style={styles.dotSeparator}> · </Text>
          <Text style={styles.ratingText}>{formatStudentCount(course.total_students)}</Text>
        </View>
        <View style={styles.statsRow}>
          <View style={styles.statChip}>
            <Ionicons name="time-outline" size={11} color={COLORS.textMuted} />
            <Text style={styles.statChipText}>{course.duration_hours}h</Text>
          </View>
          <View style={styles.statChip}>
            <Ionicons name="layers-outline" size={11} color={COLORS.textMuted} />
            <Text style={styles.statChipText}>Course</Text>
          </View>
        </View>
        {enrollmentProgress !== undefined && (
          <View style={{ marginTop: 6 }}>
            <ProgressBar
              progress={enrollmentProgress}
              height={4}
              colorStart="#43E97B"
              colorEnd="#38F9D7"
            />
          </View>
        )}
      </View>
    </TouchableOpacity>
  );
}

// ── Featured Variant ──────────────────────────────────────────────────────────
function FeaturedCard({ course, onPress, enrollmentProgress }: CourseCardProps) {
  return (
    <TouchableOpacity onPress={onPress} style={styles.featuredCard} activeOpacity={0.88}>
      <View style={StyleSheet.absoluteFill}>
        {course.thumbnail_url ? (
          <Image
            source={{ uri: course.thumbnail_url }}
            style={StyleSheet.absoluteFill}
            resizeMode="cover"
          />
        ) : (
          <ThumbnailPlaceholder size="large" />
        )}
      </View>
      <LinearGradient
        colors={['transparent', 'rgba(10,10,26,0.85)', 'rgba(10,10,26,0.98)']}
        style={styles.featuredOverlay}
      >
        <View style={styles.featuredContent}>
          <View style={styles.featuredBadgeRow}>
            <PriceTag isPremium={course.is_premium} />
            <View style={styles.categoryChip}>
              <Text style={styles.categoryChipText}>{formatCategory(course.category)}</Text>
            </View>
          </View>
          <Text style={styles.featuredTitle} numberOfLines={2}>
            {course.title}
          </Text>
          <Text style={styles.featuredInstructor} numberOfLines={1}>
            by {course.instructor_name ?? 'TechSei Instructor'}
          </Text>
          <View style={styles.featuredMeta}>
            <View style={styles.ratingRow}>
              <StarRating rating={course.rating} />
              <Text style={styles.ratingText}> {course.rating.toFixed(1)}</Text>
              <Text style={styles.dotSeparator}> · </Text>
              <Text style={styles.ratingText}>{formatStudentCount(course.total_students)} students</Text>
            </View>
            <View style={styles.durationChip}>
              <Ionicons name="time-outline" size={12} color={COLORS.textMuted} />
              <Text style={styles.statChipText}>{course.duration_hours}h</Text>
            </View>
          </View>
          {enrollmentProgress !== undefined && (
            <View style={{ marginTop: 10 }}>
              <ProgressBar
                progress={enrollmentProgress}
                height={5}
                colorStart="#43E97B"
                colorEnd="#38F9D7"
                showLabel
                labelText={`${Math.round(enrollmentProgress)}% complete`}
              />
            </View>
          )}
        </View>
      </LinearGradient>
    </TouchableOpacity>
  );
}

// ── Main Export ───────────────────────────────────────────────────────────────
export function CourseCard({
  course,
  onPress,
  variant = 'grid',
  enrollmentProgress,
  style,
}: CourseCardProps) {
  const props = { course, onPress, variant, enrollmentProgress };
  if (variant === 'featured') return <FeaturedCard {...props} />;
  if (variant === 'horizontal') return <HorizontalCard {...props} />;
  return <GridCard {...props} />;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function formatStudentCount(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}k`;
  return String(count);
}

function formatCategory(cat: string): string {
  return cat.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  // ── Grid ──
  gridCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 4,
  },
  gridThumbnailContainer: {
    position: 'relative',
    height: 120,
    overflow: 'hidden',
  },
  gridThumbnail: {
    width: '100%',
    height: '100%',
  },
  thumbnailBadge: {
    position: 'absolute',
    top: 8,
    right: 8,
  },
  thumbnailBadgeSmall: {
    position: 'absolute',
    top: 6,
    right: 6,
  },
  gridBody: {
    padding: 12,
  },
  gridTitle: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: '700',
    lineHeight: 18,
    marginBottom: 4,
  },
  gridStats: {
    flexDirection: 'row',
    gap: 6,
    marginTop: 6,
  },

  // ── Horizontal ──
  horizontalCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    flexDirection: 'row',
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: COLORS.border,
    width: 290,
    marginRight: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 4,
  },
  horizontalThumbnailContainer: {
    position: 'relative',
    width: 100,
    overflow: 'hidden',
  },
  horizontalThumbnail: {
    width: 100,
    height: '100%',
  },
  horizontalBody: {
    flex: 1,
    padding: 12,
    justifyContent: 'center',
  },
  horizontalTitle: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: '700',
    lineHeight: 18,
    marginBottom: 4,
  },

  // ── Featured ──
  featuredCard: {
    borderRadius: 20,
    overflow: 'hidden',
    height: 220,
    position: 'relative',
    borderWidth: 1,
    borderColor: COLORS.border,
    shadowColor: '#6C63FF',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 12,
    elevation: 8,
  },
  featuredOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'flex-end',
  },
  featuredContent: {
    padding: 16,
  },
  featuredBadgeRow: {
    flexDirection: 'row',
    marginBottom: 8,
    gap: 8,
  },
  featuredTitle: {
    color: COLORS.text,
    fontSize: 18,
    fontWeight: '800',
    lineHeight: 24,
    marginBottom: 4,
  },
  featuredInstructor: {
    color: COLORS.textMuted,
    fontSize: 13,
    marginBottom: 8,
  },
  featuredMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  durationChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(255,255,255,0.1)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },

  // ── Shared ──
  instructorText: {
    color: COLORS.textMuted,
    fontSize: 11,
    marginBottom: 6,
  },
  ratingRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  starsRow: {
    flexDirection: 'row',
    gap: 1,
  },
  ratingText: {
    color: COLORS.textMuted,
    fontSize: 11,
    fontWeight: '600',
  },
  dotSeparator: {
    color: COLORS.textMuted,
    fontSize: 11,
  },
  statsRow: {
    flexDirection: 'row',
    marginTop: 6,
    gap: 6,
  },
  statChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
    backgroundColor: COLORS.surfaceLight,
    paddingHorizontal: 7,
    paddingVertical: 3,
    borderRadius: 6,
  },
  statChipText: {
    color: COLORS.textMuted,
    fontSize: 10,
    fontWeight: '600',
  },

  // ── Badges ──
  proBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
    backgroundColor: COLORS.gold,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    shadowColor: COLORS.gold,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.4,
    shadowRadius: 4,
    elevation: 4,
  },
  proBadgeText: {
    color: '#000',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  freeBadge: {
    backgroundColor: COLORS.accent,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  freeBadgeText: {
    color: '#000',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  categoryChip: {
    backgroundColor: 'rgba(108,99,255,0.25)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(108,99,255,0.4)',
  },
  categoryChipText: {
    color: '#A5A0FF',
    fontSize: 10,
    fontWeight: '700',
  },
});

export default CourseCard;
