// ============================================================
// TechSei LMS — XPNotification (toast + level-up celebration)
// ============================================================
import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  Dimensions,
  Modal,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

const COLORS = {
  background: '#0A0A1A',
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  primary: '#6C63FF',
  accent: '#43E97B',
  gold: '#FFD700',
  text: '#FFFFFF',
  textSecondary: '#9494B8',
  overlay: 'rgba(0,0,0,0.85)',
};

// ─── Confetti Particle ────────────────────────────────────────────────────────

const CONFETTI_COLORS = [
  COLORS.accent,
  COLORS.primary,
  COLORS.gold,
  '#FF6584',
  '#38F9D7',
  '#FFB84C',
];

interface Particle {
  id: number;
  x: Animated.Value;
  y: Animated.Value;
  rotate: Animated.Value;
  opacity: Animated.Value;
  color: string;
  size: number;
}

function useParticles(count: number, active: boolean): Particle[] {
  const particles = useRef<Particle[]>([]);

  if (particles.current.length === 0) {
    for (let i = 0; i < count; i++) {
      particles.current.push({
        id: i,
        x: new Animated.Value(0),
        y: new Animated.Value(0),
        rotate: new Animated.Value(0),
        opacity: new Animated.Value(0),
        color: CONFETTI_COLORS[i % CONFETTI_COLORS.length],
        size: 6 + Math.random() * 8,
      });
    }
  }

  useEffect(() => {
    if (!active) return;
    const anims = particles.current.map((p) => {
      const startX = SCREEN_WIDTH * 0.3 + Math.random() * SCREEN_WIDTH * 0.4;
      const endX = startX + (Math.random() - 0.5) * SCREEN_WIDTH * 0.6;
      const startY = SCREEN_HEIGHT * 0.35;
      const endY = SCREEN_HEIGHT * 0.85;
      const delay = Math.random() * 600;

      p.x.setValue(startX);
      p.y.setValue(startY);
      p.rotate.setValue(0);
      p.opacity.setValue(0);

      return Animated.sequence([
        Animated.delay(delay),
        Animated.parallel([
          Animated.timing(p.opacity, { toValue: 1, duration: 200, useNativeDriver: true }),
          Animated.timing(p.x, { toValue: endX, duration: 1600, useNativeDriver: true }),
          Animated.timing(p.y, { toValue: endY, duration: 1600, useNativeDriver: true }),
          Animated.timing(p.rotate, { toValue: 6, duration: 1600, useNativeDriver: true }),
        ]),
        Animated.timing(p.opacity, { toValue: 0, duration: 400, useNativeDriver: true }),
      ]);
    });

    Animated.stagger(40, anims).start();
  }, [active]);

  return particles.current;
}

// ─── Props ────────────────────────────────────────────────────────────────────

export interface XPNotificationProps {
  xp: number;
  reason: string;
  visible: boolean;
  onHide: () => void;
  /** When true, shows the full-screen level-up celebration instead of the toast */
  levelUp?: boolean;
  newLevel?: number;
}

// ─── Toast variant ────────────────────────────────────────────────────────────

function XPToast({
  xp,
  reason,
  visible,
  onHide,
}: Omit<XPNotificationProps, 'levelUp' | 'newLevel'>) {
  const slideY = useRef(new Animated.Value(-120)).current;
  const opacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (visible) {
      Animated.parallel([
        Animated.spring(slideY, { toValue: 0, useNativeDriver: true, friction: 8 }),
        Animated.timing(opacity, { toValue: 1, duration: 250, useNativeDriver: true }),
      ]).start();

      const timer = setTimeout(() => {
        Animated.parallel([
          Animated.timing(slideY, { toValue: -120, duration: 300, useNativeDriver: true }),
          Animated.timing(opacity, { toValue: 0, duration: 300, useNativeDriver: true }),
        ]).start(() => onHide());
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [visible, slideY, opacity, onHide]);

  if (!visible) return null;

  return (
    <Animated.View
      style={[
        styles.toastContainer,
        { transform: [{ translateY: slideY }], opacity },
      ]}
      pointerEvents="none"
    >
      <LinearGradient
        colors={['#43E97B', '#38F9D7']}
        style={styles.toastGradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
      >
        <View style={styles.toastIcon}>
          <Ionicons name="flash" size={20} color="#0A0A1A" />
        </View>
        <View style={styles.toastTextGroup}>
          <Text style={styles.toastXP}>+{xp} XP</Text>
          <Text style={styles.toastReason} numberOfLines={1}>{reason}</Text>
        </View>
      </LinearGradient>
    </Animated.View>
  );
}

// ─── Level-Up Celebration ─────────────────────────────────────────────────────

function LevelUpCelebration({
  newLevel,
  visible,
  onHide,
}: {
  newLevel: number;
  visible: boolean;
  onHide: () => void;
}) {
  const scale = useRef(new Animated.Value(0.4)).current;
  const opacity = useRef(new Animated.Value(0)).current;
  const particles = useParticles(28, visible);

  useEffect(() => {
    if (visible) {
      Animated.parallel([
        Animated.spring(scale, { toValue: 1, useNativeDriver: true, friction: 5, tension: 80 }),
        Animated.timing(opacity, { toValue: 1, duration: 350, useNativeDriver: true }),
      ]).start();

      const timer = setTimeout(() => {
        Animated.parallel([
          Animated.timing(scale, { toValue: 0.8, duration: 300, useNativeDriver: true }),
          Animated.timing(opacity, { toValue: 0, duration: 300, useNativeDriver: true }),
        ]).start(() => onHide());
      }, 3500);

      return () => clearTimeout(timer);
    }
  }, [visible, scale, opacity, onHide]);

  return (
    <Modal visible={visible} transparent animationType="none">
      <View style={styles.celebrationBg}>
        {/* Confetti particles */}
        {particles.map((p) => (
          <Animated.View
            key={p.id}
            style={[
              styles.particle,
              {
                backgroundColor: p.color,
                width: p.size,
                height: p.size,
                borderRadius: p.size / 2,
                position: 'absolute',
                left: 0,
                top: 0,
                opacity: p.opacity,
                transform: [
                  { translateX: p.x },
                  { translateY: p.y },
                  {
                    rotate: p.rotate.interpolate({
                      inputRange: [0, 6],
                      outputRange: ['0deg', '1080deg'],
                    }),
                  },
                ],
              },
            ]}
          />
        ))}

        {/* Card */}
        <Animated.View
          style={[styles.celebrationCard, { transform: [{ scale }], opacity }]}
        >
          <LinearGradient
            colors={['#6C63FF', '#8B5CF6']}
            style={styles.celebrationGradient}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
          >
            <Text style={styles.celebrationEmoji}>🎉</Text>
            <Text style={styles.levelUpLabel}>LEVEL UP!</Text>
            <Text style={styles.levelNumber}>Level {newLevel}</Text>
            <Text style={styles.levelSubtext}>You're on fire! Keep it up!</Text>
          </LinearGradient>
        </Animated.View>
      </View>
    </Modal>
  );
}

// ─── Main export ──────────────────────────────────────────────────────────────

export default function XPNotification({
  xp,
  reason,
  visible,
  onHide,
  levelUp = false,
  newLevel = 1,
}: XPNotificationProps) {
  if (levelUp) {
    return (
      <LevelUpCelebration newLevel={newLevel} visible={visible} onHide={onHide} />
    );
  }

  return <XPToast xp={xp} reason={reason} visible={visible} onHide={onHide} />;
}

const styles = StyleSheet.create({
  // Toast
  toastContainer: {
    position: 'absolute',
    top: 52,
    left: 16,
    right: 16,
    zIndex: 9999,
    borderRadius: 16,
    overflow: 'hidden',
    elevation: 12,
    shadowColor: COLORS.accent,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.5,
    shadowRadius: 12,
  },
  toastGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    gap: 12,
  },
  toastIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(0,0,0,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  toastTextGroup: {
    flex: 1,
  },
  toastXP: {
    fontSize: 20,
    fontWeight: '800',
    color: '#0A0A1A',
  },
  toastReason: {
    fontSize: 12,
    color: 'rgba(10,10,26,0.7)',
    fontWeight: '600',
  },
  // Level-up
  celebrationBg: {
    flex: 1,
    backgroundColor: COLORS.overlay,
    alignItems: 'center',
    justifyContent: 'center',
  },
  particle: {},
  celebrationCard: {
    borderRadius: 28,
    overflow: 'hidden',
    elevation: 20,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.6,
    shadowRadius: 24,
  },
  celebrationGradient: {
    width: SCREEN_WIDTH * 0.78,
    padding: 36,
    alignItems: 'center',
  },
  celebrationEmoji: {
    fontSize: 64,
    marginBottom: 12,
  },
  levelUpLabel: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 14,
    fontWeight: '800',
    letterSpacing: 4,
    marginBottom: 8,
  },
  levelNumber: {
    color: COLORS.text,
    fontSize: 42,
    fontWeight: '900',
    marginBottom: 10,
  },
  levelSubtext: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 15,
    fontWeight: '600',
    textAlign: 'center',
  },
});
