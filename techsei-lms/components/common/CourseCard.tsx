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
import { Course } from '../../stores/courseStore';
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

function PriceTag({ isPremium }: { isPremium: boolean }) {
  if (isPremium) {
    return (
      <View style={styles.proBadge}>
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

// ── Grid variant ─────────────────────────────────────────────────────────────
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
          <Image source={{ uri: course.thumbnail_url }} style={styles.gridThumbnail} />
        ) : (
          <LinearGradient
            colors={['#6C63FF', '#A855F7']}
            style={styles.gridThumbnail}
          >
            <Ionicons name="book" size={32} color="#FFFFFF60" />
          </LinearGradient>
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
        {enrollmentProgress !== undefined && (
          <View style={{ marginTop: 6 }}>
            <ProgressBar progress={enrollmentProgress} height={4} colorStart="#43E97B" colorEnd="#38F9D7" />
          </View>
        )}
      </View>
    </TouchableOpacity>
  );
}

// ── Horizontal variant ───────────────────────────────────────────────────────
function HorizontalCard({ course, onPress, enrollmentProgress }: CourseCardProps) {
  return (
    <TouchableOpacity onPress={onPress} style={styles.horizontalCard} activeOpacity={0.85}>
      <View style={styles.horizontalThumbnailContainer}>
        {course.thumbnail_url ? (
          <Image source={{ uri: course.thumbnail_url }} style={styles.horizontalThumbnail} />
        ) : (
          <LinearGradient
            colors={['#6C63FF', '#A855F7']}
            style={styles.horizontalThumbnail}
          >
            <Ionicons name="book" size={28} color="#FFFFFF60" />
          </LinearGradient>
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
        </View>
        {enrollmentProgress !== undefined && (
          <View style={{ marginTop: 6 }}>
            <ProgressBar progress={enrollmentProgress} height={4} colorStart="#43E97B" colorEnd="#38F9D7" />
          </View>
        )}
      </View>
    </TouchableOpacity>
  );
}

// ── Featured variant ─────────────────────────────────────────────────────────
function FeaturedCard({ course, onPress, enrollmentProgress }: CourseCardProps) {
  return (
    <TouchableOpacity onPress={onPress} style={styles.featuredCard} activeOpacity={0.88}>
      {course.thumbnail_url ? (
        <Image source={{ uri: course.thumbnail_url }} style={styles.featuredThumbnail} />
      ) : (
        <LinearGradient
          colors={['#6C63FF', '#A855F7', '#EC4899']}
          style={styles.featuredThumbnail}
        >
          <Ionicons name="book" size={60} color="#FFFFFF30" />
        </LinearGradient>
      )}
      <LinearGradient
        colors={['transparent', 'rgba(10,10,26,0.95)']}
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
            {course.instructor_name ?? 'TechSei Instructor'}
          </Text>
          <View style={styles.ratingRow}>
            <StarRating rating={course.rating} />
            <Text style={styles.ratingText}> {course.rating.toFixed(1)}</Text>
            <Text style={styles.dotSeparator}> · </Text>
            <Text style={styles.ratingText}>{formatStudentCount(course.total_students)} students</Text>
          </View>
          {enrollmentProgress !== undefined && (
            <View style={{ marginTop: 8 }}>
              <ProgressBar progress={enrollmentProgress} height={5} colorStart="#43E97B" colorEnd="#38F9D7" />
            </View>
          )}
        </View>
      </LinearGradient>
    </TouchableOpacity>
  );
}

// ── Main export ──────────────────────────────────────────────────────────────
export function CourseCard({ course, onPress, variant = 'grid', enrollmentProgress, style }: CourseCardProps) {
  const props = { course, onPress, variant, enrollmentProgress };

  if (variant === 'featured') return <FeaturedCard {...props} />;
  if (variant === 'horizontal') return <HorizontalCard {...props} />;
  return <GridCard {...props} />;
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function formatStudentCount(count: number): string {
  if (count >= 1000) return `${(count / 1000).toFixed(1)}k`;
  return String(count);
}

function formatCategory(cat: string): string {
  return cat.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

// ── Styles ───────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  // Grid
  gridCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  gridThumbnailContainer: {
    position: 'relative',
  },
  gridThumbnail: {
    width: '100%',
    height: 110,
    alignItems: 'center',
    justifyContent: 'center',
    resizeMode: 'cover',
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
    padding: 10,
  },
  gridTitle: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: '700',
    lineHeight: 18,
    marginBottom: 4,
  },
  // Horizontal
  horizontalCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    flexDirection: 'row',
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: COLORS.border,
    width: 280,
    marginRight: 12,
  },
  horizontalThumbnailContainer: {
    position: 'relative',
  },
  horizontalThumbnail: {
    width: 95,
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
    resizeMode: 'cover',
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
    marginBottom: 3,
  },
  // Featured
  featuredCard: {
    borderRadius: 20,
    overflow: 'hidden',
    height: 200,
    position: 'relative',
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  featuredThumbnail: {
    ...StyleSheet.absoluteFillObject,
    resizeMode: 'cover',
    alignItems: 'center',
    justifyContent: 'center',
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
    marginBottom: 6,
  },
  // Shared
  instructorText: {
    color: COLORS.textMuted,
    fontSize: 11,
    marginBottom: 5,
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
    gap: 8,
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
  // Badges
  proBadge: {
    backgroundColor: COLORS.gold,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
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
    backgroundColor: 'rgba(108,99,255,0.3)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(108,99,255,0.5)',
  },
  categoryChipText: {
    color: COLORS.primary,
    fontSize: 10,
    fontWeight: '700',
  },
});

export default CourseCard;
