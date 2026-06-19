// ============================================================
// TechSei LMS — BadgeCard Component
// ============================================================
import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Modal,
  StyleSheet,
  Animated,
  Dimensions,
  ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import type { Badge } from '../../types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

const COLORS = {
  background: '#0A0A1A',
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  primary: '#6C63FF',
  accent: '#43E97B',
  gold: '#FFD700',
  warning: '#FFB84C',
  text: '#FFFFFF',
  textSecondary: '#9494B8',
  textMuted: '#5A5A7A',
  border: '#2A2A4A',
  error: '#FF4B4B',
  overlay: 'rgba(0,0,0,0.75)',
};

const RARITY_COLORS: Record<string, [string, string]> = {
  common: ['#9494B8', '#5A5A7A'],
  rare: ['#6C63FF', '#8B5CF6'],
  epic: ['#FF6584', '#C026D3'],
  legendary: ['#FFD700', '#FFA500'],
};

const RARITY_GLOW: Record<string, string> = {
  common: 'rgba(148,148,184,0.35)',
  rare: 'rgba(108,99,255,0.45)',
  epic: 'rgba(255,101,132,0.45)',
  legendary: 'rgba(255,215,0,0.55)',
};

export type BadgeSize = 'small' | 'medium' | 'large';

interface BadgeCardProps {
  badge: Badge;
  earned: boolean;
  earnedAt?: string;
  size?: BadgeSize;
}

const SIZE_CONFIG: Record<BadgeSize, { container: number; icon: number; fontSize: number }> = {
  small: { container: 64, icon: 24, fontSize: 10 },
  medium: { container: 88, icon: 32, fontSize: 12 },
  large: { container: 110, icon: 42, fontSize: 14 },
};

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString('en-US', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
    });
  } catch {
    return iso;
  }
}

export default function BadgeCard({ badge, earned, earnedAt, size = 'medium' }: BadgeCardProps) {
  const [modalVisible, setModalVisible] = useState(false);
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const glowAnim = useRef(new Animated.Value(0.5)).current;

  const cfg = SIZE_CONFIG[size];
  const rarity = badge.rarity ?? 'common';
  const gradientColors = earned ? RARITY_COLORS[rarity] : ['#2A2A4A', '#1E1E3A'];
  const glowColor = earned ? RARITY_GLOW[rarity] : 'transparent';

  React.useEffect(() => {
    if (earned) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(glowAnim, { toValue: 1, duration: 1400, useNativeDriver: true }),
          Animated.timing(glowAnim, { toValue: 0.5, duration: 1400, useNativeDriver: true }),
        ])
      ).start();
    }
  }, [earned, glowAnim]);

  function handlePressIn() {
    Animated.spring(scaleAnim, { toValue: 0.92, useNativeDriver: true }).start();
  }

  function handlePressOut() {
    Animated.spring(scaleAnim, { toValue: 1, friction: 3, useNativeDriver: true }).start();
    setModalVisible(true);
  }

  return (
    <>
      <TouchableOpacity
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        activeOpacity={1}
        style={styles.touchable}
      >
        <Animated.View
          style={[
            styles.badgeOuter,
            {
              width: cfg.container + 16,
              transform: [{ scale: scaleAnim }],
            },
          ]}
        >
          {/* Glow halo (earned only) */}
          {earned && (
            <Animated.View
              style={[
                styles.glowHalo,
                {
                  width: cfg.container + 24,
                  height: cfg.container + 24,
                  borderRadius: (cfg.container + 24) / 2,
                  backgroundColor: glowColor,
                  opacity: glowAnim,
                },
              ]}
            />
          )}

          <LinearGradient
            colors={gradientColors as [string, string]}
            style={[
              styles.badgeCircle,
              {
                width: cfg.container,
                height: cfg.container,
                borderRadius: cfg.container / 2,
              },
            ]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
          >
            <Text style={{ fontSize: cfg.icon, opacity: earned ? 1 : 0.35 }}>
              {badge.icon || '🏅'}
            </Text>
            {!earned && (
              <View style={styles.lockedOverlay}>
                <Ionicons name="lock-closed" size={cfg.icon * 0.6} color={COLORS.textMuted} />
              </View>
            )}
          </LinearGradient>

          {/* Rarity star (earned legendary/epic) */}
          {earned && (rarity === 'legendary' || rarity === 'epic') && (
            <View style={styles.rarityBadge}>
              <Text style={{ fontSize: 8 }}>{rarity === 'legendary' ? '⭐' : '✨'}</Text>
            </View>
          )}

          {size !== 'small' && (
            <Text
              style={[
                styles.badgeName,
                { fontSize: cfg.fontSize, color: earned ? COLORS.text : COLORS.textMuted },
              ]}
              numberOfLines={2}
            >
              {badge.name}
            </Text>
          )}
        </Animated.View>
      </TouchableOpacity>

      {/* Detail Modal */}
      <Modal
        visible={modalVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setModalVisible(false)}
      >
        <TouchableOpacity
          style={styles.modalBackdrop}
          activeOpacity={1}
          onPress={() => setModalVisible(false)}
        >
          <View style={styles.modalCard}>
            <LinearGradient
              colors={gradientColors as [string, string]}
              style={styles.modalIconCircle}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
            >
              <Text style={styles.modalIcon}>{badge.icon || '🏅'}</Text>
              {!earned && (
                <View style={styles.lockedOverlay}>
                  <Ionicons name="lock-closed" size={28} color={COLORS.textMuted} />
                </View>
              )}
            </LinearGradient>

            <View style={styles.rarityChip}>
              <Text style={[styles.rarityText, { color: earned ? RARITY_COLORS[rarity][0] : COLORS.textMuted }]}>
                {rarity.toUpperCase()}
              </Text>
            </View>

            <Text style={styles.modalTitle}>{badge.name}</Text>
            <Text style={styles.modalDesc}>{badge.description}</Text>

            {earned && earnedAt ? (
              <View style={styles.earnedRow}>
                <Ionicons name="checkmark-circle" size={16} color={COLORS.accent} />
                <Text style={styles.earnedText}>Earned on {formatDate(earnedAt)}</Text>
              </View>
            ) : (
              <View style={styles.lockedInfoBox}>
                <Ionicons name="lock-closed-outline" size={14} color={COLORS.textMuted} />
                <Text style={styles.lockedInfoText}>
                  {badge.xp_required > 0
                    ? `Earn ${badge.xp_required.toLocaleString()} XP to unlock`
                    : 'Complete special requirements to unlock'}
                </Text>
              </View>
            )}

            <TouchableOpacity style={styles.closeBtn} onPress={() => setModalVisible(false)}>
              <Text style={styles.closeBtnText}>Close</Text>
            </TouchableOpacity>
          </View>
        </TouchableOpacity>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  touchable: {
    alignItems: 'center',
  },
  badgeOuter: {
    alignItems: 'center',
    position: 'relative',
    paddingTop: 12,
    paddingBottom: 4,
  },
  glowHalo: {
    position: 'absolute',
    top: 0,
    alignSelf: 'center',
  },
  badgeCircle: {
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    elevation: 6,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
  },
  lockedOverlay: {
    position: 'absolute',
    inset: 0,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(10,10,26,0.6)',
    borderRadius: 999,
  },
  rarityBadge: {
    position: 'absolute',
    top: 4,
    right: 8,
    backgroundColor: COLORS.surface,
    borderRadius: 8,
    padding: 2,
  },
  badgeName: {
    marginTop: 6,
    textAlign: 'center',
    fontWeight: '600',
    maxWidth: 90,
  },
  // Modal
  modalBackdrop: {
    flex: 1,
    backgroundColor: COLORS.overlay,
    alignItems: 'center',
    justifyContent: 'center',
  },
  modalCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 24,
    padding: 28,
    width: SCREEN_WIDTH * 0.82,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  modalIconCircle: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    marginBottom: 16,
  },
  modalIcon: {
    fontSize: 44,
  },
  rarityChip: {
    backgroundColor: COLORS.surfaceLight,
    paddingHorizontal: 12,
    paddingVertical: 3,
    borderRadius: 12,
    marginBottom: 12,
  },
  rarityText: {
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1,
  },
  modalTitle: {
    color: COLORS.text,
    fontSize: 22,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 8,
  },
  modalDesc: {
    color: COLORS.textSecondary,
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 18,
  },
  earnedRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 20,
  },
  earnedText: {
    color: COLORS.accent,
    fontSize: 13,
    fontWeight: '600',
  },
  lockedInfoBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: COLORS.surfaceLight,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 12,
    marginBottom: 20,
  },
  lockedInfoText: {
    color: COLORS.textMuted,
    fontSize: 13,
    flex: 1,
  },
  closeBtn: {
    backgroundColor: COLORS.primary,
    borderRadius: 14,
    paddingHorizontal: 36,
    paddingVertical: 12,
  },
  closeBtnText: {
    color: COLORS.text,
    fontWeight: '700',
    fontSize: 15,
  },
});
