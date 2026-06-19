// ============================================================
// TechSei LMS — Admin Stat Card Component
// ============================================================
import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  Animated,
  StyleSheet,
  ViewStyle,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

const COLORS = {
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  text: '#FFFFFF',
  textMuted: '#8A8AAA',
  border: '#2A2A4A',
  accent: '#43E97B',
  error: '#FF6B6B',
};

export interface TrendInfo {
  value: number;
  direction: 'up' | 'down';
}

export interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ComponentProps<typeof Ionicons>['name'];
  color: string;
  trend?: TrendInfo;
  subtitle?: string;
  style?: ViewStyle;
}

export function StatCard({
  title,
  value,
  icon,
  color,
  trend,
  subtitle,
  style,
}: StatCardProps) {
  const animatedValue = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.92)).current;
  const opacityAnim = useRef(new Animated.Value(0)).current;

  // Numeric counting animation
  const numericValue = typeof value === 'number' ? value : parseFloat(String(value).replace(/[^0-9.]/g, '')) || 0;
  const isNumeric = typeof value === 'number' || /^[\d.,]+$/.test(String(value).replace(/[^0-9.]/g, ''));

  useEffect(() => {
    Animated.parallel([
      Animated.spring(scaleAnim, {
        toValue: 1,
        useNativeDriver: true,
        tension: 80,
        friction: 8,
      }),
      Animated.timing(opacityAnim, {
        toValue: 1,
        duration: 400,
        useNativeDriver: true,
      }),
      Animated.timing(animatedValue, {
        toValue: numericValue,
        duration: 1200,
        useNativeDriver: false,
      }),
    ]).start();
  }, [numericValue]);

  const displayValue = isNumeric
    ? animatedValue.interpolate({
        inputRange: [0, numericValue || 1],
        outputRange: ['0', String(numericValue)],
        extrapolate: 'clamp',
      })
    : null;

  const gradientColors: [string, string] = [
    `${color}22`,
    `${color}08`,
  ];

  const trendColor =
    trend?.direction === 'up' ? COLORS.accent : COLORS.error;
  const trendIcon =
    trend?.direction === 'up' ? 'trending-up' : 'trending-down';

  return (
    <Animated.View
      style={[
        styles.container,
        {
          opacity: opacityAnim,
          transform: [{ scale: scaleAnim }],
        },
        style,
      ]}
    >
      <LinearGradient
        colors={gradientColors}
        style={styles.gradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        {/* Icon circle */}
        <View style={[styles.iconCircle, { backgroundColor: `${color}20` }]}>
          <LinearGradient
            colors={[color, `${color}CC`]}
            style={styles.iconGradient}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
          >
            <Ionicons name={icon} size={20} color="#FFFFFF" />
          </LinearGradient>
        </View>

        {/* Value */}
        <View style={styles.valueRow}>
          {isNumeric && displayValue ? (
            <Animated.Text style={[styles.value, { color }]}>
              {displayValue}
            </Animated.Text>
          ) : (
            <Text style={[styles.value, { color }]}>{value}</Text>
          )}
        </View>

        {/* Title */}
        <Text style={styles.title} numberOfLines={2}>
          {title}
        </Text>

        {/* Subtitle */}
        {subtitle && (
          <Text style={styles.subtitle} numberOfLines={1}>
            {subtitle}
          </Text>
        )}

        {/* Trend */}
        {trend && (
          <View style={styles.trendRow}>
            <Ionicons name={trendIcon} size={13} color={trendColor} />
            <Text style={[styles.trendText, { color: trendColor }]}>
              {trend.value}%{' '}
              <Text style={styles.trendLabel}>vs last month</Text>
            </Text>
          </View>
        )}

        {/* Accent corner bar */}
        <View style={[styles.cornerBar, { backgroundColor: color }]} />
      </LinearGradient>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: COLORS.border,
    flex: 1,
    minHeight: 140,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 6,
  },
  gradient: {
    flex: 1,
    padding: 16,
    justifyContent: 'space-between',
  },
  iconCircle: {
    width: 40,
    height: 40,
    borderRadius: 12,
    marginBottom: 12,
    overflow: 'hidden',
  },
  iconGradient: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  valueRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  value: {
    fontSize: 28,
    fontWeight: '800',
    letterSpacing: -0.5,
    lineHeight: 32,
  },
  title: {
    color: COLORS.textMuted,
    fontSize: 12,
    fontWeight: '600',
    marginTop: 4,
    letterSpacing: 0.2,
  },
  subtitle: {
    color: COLORS.textMuted,
    fontSize: 10,
    marginTop: 2,
    opacity: 0.7,
  },
  trendRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
    marginTop: 8,
  },
  trendText: {
    fontSize: 11,
    fontWeight: '700',
  },
  trendLabel: {
    color: COLORS.textMuted,
    fontWeight: '400',
  },
  cornerBar: {
    position: 'absolute',
    top: 0,
    right: 0,
    width: 3,
    height: '40%',
    borderBottomLeftRadius: 3,
  },
});

export default StatCard;
