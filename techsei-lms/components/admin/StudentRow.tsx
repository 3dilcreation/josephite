// ============================================================
// TechSei LMS — Admin Student Row Component
// ============================================================
import React, { useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Animated,
  StyleSheet,
  PanResponder,
  Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const SWIPE_THRESHOLD = 60;
const ACTION_WIDTH = 70;

const COLORS = {
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  primary: '#6C63FF',
  accent: '#43E97B',
  text: '#FFFFFF',
  textMuted: '#8A8AAA',
  border: '#2A2A4A',
  warning: '#FFB84C',
  error: '#FF6B6B',
  blue: '#4FC3F7',
  gold: '#FFD700',
  selected: '#6C63FF22',
};

export interface Student {
  id: string;
  name: string;
  email: string;
  level: number;
  subscriptionTier: 'free' | 'pro' | 'enterprise';
  lastActive: string;
  xp: number;
  avatar?: string;
  isActive: boolean;
  joinDate: string;
  coursesEnrolled: number;
  coursesCompleted: number;
}

interface StudentRowProps {
  student: Student;
  onPress: () => void;
  onLongPress?: () => void;
  onEdit?: () => void;
  onDelete?: () => void;
  selected?: boolean;
  selectionMode?: boolean;
}

function AvatarCircle({ name, size = 44 }: { name: string; size?: number }) {
  const initials = name
    .split(' ')
    .map((n) => n[0])
    .slice(0, 2)
    .join('')
    .toUpperCase();

  // Deterministic color from name
  const hue = name.charCodeAt(0) * 15 % 360;
  const bgColor = `hsl(${hue}, 60%, 35%)`;

  return (
    <View
      style={[
        styles.avatarCircle,
        {
          width: size,
          height: size,
          borderRadius: size / 2,
          backgroundColor: bgColor,
        },
      ]}
    >
      <Text style={[styles.avatarText, { fontSize: size * 0.36 }]}>
        {initials}
      </Text>
    </View>
  );
}

function LevelBadge({ level }: { level: number }) {
  return (
    <View style={styles.levelBadge}>
      <Text style={styles.levelBadgeText}>Lv.{level}</Text>
    </View>
  );
}

function TierBadge({ tier }: { tier: Student['subscriptionTier'] }) {
  const config = {
    pro: { colors: ['#FFD700', '#FFA500'] as [string, string], label: 'PRO', textColor: '#000' },
    enterprise: { colors: ['#6C63FF', '#A855F7'] as [string, string], label: 'ENT', textColor: '#fff' },
    free: { colors: ['#2A2A4A', '#1E1E3A'] as [string, string], label: 'FREE', textColor: '#8A8AAA' },
  };
  const c = config[tier];
  return (
    <LinearGradient colors={c.colors} style={styles.tierBadge} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}>
      <Text style={[styles.tierBadgeText, { color: c.textColor }]}>{c.label}</Text>
    </LinearGradient>
  );
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days}d ago`;
  if (days < 30) return `${Math.floor(days / 7)}w ago`;
  return `${Math.floor(days / 30)}mo ago`;
}

function formatXP(xp: number): string {
  if (xp >= 1000) return `${(xp / 1000).toFixed(1)}k`;
  return String(xp);
}

export function StudentRow({
  student,
  onPress,
  onLongPress,
  onEdit,
  onDelete,
  selected = false,
  selectionMode = false,
}: StudentRowProps) {
  const translateX = useRef(new Animated.Value(0)).current;
  const swipeOpen = useRef(false);

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gs) =>
        !selectionMode && Math.abs(gs.dx) > 8 && Math.abs(gs.dy) < 20,
      onPanResponderMove: (_, gs) => {
        // Only allow left swipe (negative dx) to reveal actions
        if (gs.dx < 0) {
          const clamped = Math.max(gs.dx, -(ACTION_WIDTH * 2 + 16));
          translateX.setValue(swipeOpen.current ? clamped - ACTION_WIDTH * 2 : clamped);
        }
      },
      onPanResponderRelease: (_, gs) => {
        if (gs.dx < -SWIPE_THRESHOLD) {
          // Open actions
          Animated.spring(translateX, {
            toValue: -(ACTION_WIDTH * 2 + 8),
            useNativeDriver: true,
            tension: 100,
            friction: 10,
          }).start();
          swipeOpen.current = true;
        } else {
          // Close
          Animated.spring(translateX, {
            toValue: 0,
            useNativeDriver: true,
            tension: 100,
            friction: 10,
          }).start();
          swipeOpen.current = false;
        }
      },
    })
  ).current;

  const closeSwipe = () => {
    Animated.spring(translateX, {
      toValue: 0,
      useNativeDriver: true,
      tension: 100,
      friction: 10,
    }).start();
    swipeOpen.current = false;
  };

  return (
    <View style={styles.outerContainer}>
      {/* Swipe action buttons (revealed on left swipe) */}
      <View style={styles.actionsContainer}>
        <TouchableOpacity
          style={[styles.actionBtn, styles.editBtn]}
          onPress={() => { closeSwipe(); onEdit?.(); }}
        >
          <Ionicons name="pencil" size={18} color="#fff" />
          <Text style={styles.actionBtnText}>Edit</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.actionBtn, styles.deleteBtn]}
          onPress={() => { closeSwipe(); onDelete?.(); }}
        >
          <Ionicons name="trash" size={18} color="#fff" />
          <Text style={styles.actionBtnText}>Delete</Text>
        </TouchableOpacity>
      </View>

      {/* Main row */}
      <Animated.View
        style={[
          styles.rowContainer,
          selected && styles.rowSelected,
          { transform: [{ translateX }] },
        ]}
        {...panResponder.panHandlers}
      >
        <TouchableOpacity
          style={styles.rowContent}
          onPress={() => {
            if (swipeOpen.current) {
              closeSwipe();
              return;
            }
            onPress();
          }}
          onLongPress={onLongPress}
          activeOpacity={0.8}
        >
          {/* Selection checkbox */}
          {selectionMode && (
            <View style={[styles.checkbox, selected && styles.checkboxSelected]}>
              {selected && <Ionicons name="checkmark" size={14} color="#fff" />}
            </View>
          )}

          {/* Avatar */}
          <View style={styles.avatarWrapper}>
            <AvatarCircle name={student.name} />
            <View
              style={[
                styles.onlineDot,
                { backgroundColor: student.isActive ? COLORS.accent : COLORS.textMuted },
              ]}
            />
          </View>

          {/* Info */}
          <View style={styles.infoColumn}>
            <View style={styles.nameRow}>
              <Text style={styles.name} numberOfLines={1}>
                {student.name}
              </Text>
              <LevelBadge level={student.level} />
            </View>
            <Text style={styles.email} numberOfLines={1}>
              {student.email}
            </Text>
            <View style={styles.metaRow}>
              <Ionicons name="time-outline" size={11} color={COLORS.textMuted} />
              <Text style={styles.metaText}>{formatDate(student.lastActive)}</Text>
              <View style={styles.dot} />
              <Ionicons name="flash-outline" size={11} color={COLORS.warning} />
              <Text style={[styles.metaText, { color: COLORS.warning }]}>
                {formatXP(student.xp)} XP
              </Text>
            </View>
          </View>

          {/* Right side */}
          <View style={styles.rightColumn}>
            <TierBadge tier={student.subscriptionTier} />
            <Ionicons
              name="chevron-forward"
              size={16}
              color={COLORS.textMuted}
              style={{ marginTop: 8 }}
            />
          </View>
        </TouchableOpacity>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  outerContainer: {
    position: 'relative',
    marginHorizontal: 16,
    marginBottom: 8,
  },
  actionsContainer: {
    position: 'absolute',
    right: 0,
    top: 0,
    bottom: 0,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingRight: 4,
  },
  actionBtn: {
    width: ACTION_WIDTH - 4,
    height: '88%',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
  },
  editBtn: {
    backgroundColor: COLORS.blue,
  },
  deleteBtn: {
    backgroundColor: COLORS.error,
  },
  actionBtnText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '700',
  },
  rowContainer: {
    backgroundColor: COLORS.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 3,
  },
  rowSelected: {
    borderColor: COLORS.primary,
    backgroundColor: COLORS.selected,
  },
  rowContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    gap: 12,
  },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 2,
    borderColor: COLORS.textMuted,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxSelected: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  avatarWrapper: {
    position: 'relative',
  },
  avatarCircle: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  onlineDot: {
    position: 'absolute',
    bottom: 1,
    right: 1,
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 2,
    borderColor: COLORS.surface,
  },
  infoColumn: {
    flex: 1,
    gap: 3,
  },
  nameRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  name: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: '700',
    flex: 1,
  },
  email: {
    color: COLORS.textMuted,
    fontSize: 12,
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 2,
  },
  metaText: {
    color: COLORS.textMuted,
    fontSize: 11,
  },
  dot: {
    width: 3,
    height: 3,
    borderRadius: 1.5,
    backgroundColor: COLORS.textMuted,
    marginHorizontal: 2,
  },
  rightColumn: {
    alignItems: 'flex-end',
    justifyContent: 'center',
  },
  levelBadge: {
    backgroundColor: '#6C63FF33',
    borderWidth: 1,
    borderColor: '#6C63FF66',
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: 6,
  },
  levelBadgeText: {
    color: COLORS.primary,
    fontSize: 10,
    fontWeight: '700',
  },
  tierBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 7,
  },
  tierBadgeText: {
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
});

export default StudentRow;
