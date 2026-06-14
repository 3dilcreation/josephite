// ============================================================
// TechSei LMS — StreakWidget Component
// ============================================================
import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';

const COLORS = {
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  primary: '#6C63FF',
  warning: '#FFB84C',
  text: '#FFFFFF',
  textSecondary: '#9494B8',
  textMuted: '#5A5A7A',
  border: '#2A2A4A',
  accent: '#43E97B',
};

interface StreakWidgetProps {
  streak: number;
  lastSevenDays?: boolean[]; // true = activity that day, index 0 = 6 days ago
}

export default function StreakWidget({ streak, lastSevenDays }: StreakWidgetProps) {
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const opacityAnim = useRef(new Animated.Value(1)).current;

  const isActive = streak > 0;

  useEffect(() => {
    if (isActive) {
      Animated.loop(
        Animated.sequence([
          Animated.parallel([
            Animated.timing(pulseAnim, {
              toValue: 1.12,
              duration: 800,
              useNativeDriver: true,
            }),
            Animated.timing(opacityAnim, {
              toValue: 0.8,
              duration: 800,
              useNativeDriver: true,
            }),
          ]),
          Animated.parallel([
            Animated.timing(pulseAnim, {
              toValue: 1,
              duration: 800,
              useNativeDriver: true,
            }),
            Animated.timing(opacityAnim, {
              toValue: 1,
              duration: 800,
              useNativeDriver: true,
            }),
          ]),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
      opacityAnim.setValue(1);
    }
  }, [isActive, pulseAnim, opacityAnim]);

  // Build 7-day dot data (Sun to Sat or last 7 days)
  const dots: boolean[] = lastSevenDays ?? Array(7).fill(false);
  // Day labels: show short day names
  const today = new Date();
  const dayLabels: string[] = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(today.getDate() - i);
    dayLabels.push(['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'][d.getDay()]);
  }

  return (
    <View style={styles.container}>
      {/* Fire emoji + streak count */}
      <View style={styles.streakRow}>
        <Animated.Text
          style={[
            styles.fireEmoji,
            isActive
              ? {
                  transform: [{ scale: pulseAnim }],
                  opacity: opacityAnim,
                }
              : styles.fireInactive,
          ]}
        >
          {isActive ? '🔥' : '💤'}
        </Animated.Text>

        <View style={styles.streakTextGroup}>
          <Text style={[styles.streakCount, !isActive && styles.inactiveCount]}>
            {streak}
          </Text>
          <Text style={styles.streakLabel}>day streak</Text>
        </View>
      </View>

      {/* Broken streak warning */}
      {!isActive && (
        <View style={styles.warningBanner}>
          <Text style={styles.warningText}>Start a new streak today!</Text>
        </View>
      )}

      {/* Mini 7-day calendar dots */}
      <View style={styles.dotsRow}>
        {dots.map((active, idx) => (
          <View key={idx} style={styles.dotColumn}>
            <View
              style={[
                styles.dot,
                active
                  ? { backgroundColor: COLORS.warning }
                  : { backgroundColor: COLORS.surfaceLight },
                idx === 6 && active && styles.dotToday,
              ]}
            />
            <Text style={styles.dayLabel}>{dayLabels[idx]}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  streakRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    gap: 12,
  },
  fireEmoji: {
    fontSize: 36,
  },
  fireInactive: {
    opacity: 0.4,
  },
  streakTextGroup: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 4,
  },
  streakCount: {
    fontSize: 32,
    fontWeight: '800',
    color: COLORS.warning,
  },
  inactiveCount: {
    color: COLORS.textMuted,
  },
  streakLabel: {
    fontSize: 14,
    color: COLORS.textSecondary,
    fontWeight: '500',
  },
  warningBanner: {
    backgroundColor: 'rgba(90,90,122,0.25)',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  warningText: {
    color: COLORS.textMuted,
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'center',
  },
  dotsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  dotColumn: {
    alignItems: 'center',
    gap: 4,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  dotToday: {
    width: 12,
    height: 12,
    borderRadius: 6,
    shadowColor: COLORS.warning,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.9,
    shadowRadius: 4,
    elevation: 4,
  },
  dayLabel: {
    fontSize: 9,
    color: COLORS.textMuted,
    fontWeight: '500',
  },
});
