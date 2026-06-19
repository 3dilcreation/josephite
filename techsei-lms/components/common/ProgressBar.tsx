// ============================================================
// TechSei LMS — Reusable Animated Progress Bar
// ============================================================
import React, { useEffect, useRef } from 'react';
import { View, Text, Animated, StyleSheet, ViewStyle } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

interface ProgressBarProps {
  /** 0–100 */
  progress: number;
  /** Gradient start color — defaults to accent green */
  colorStart?: string;
  /** Gradient end color */
  colorEnd?: string;
  height?: number;
  showLabel?: boolean;
  /** Override label text (default: "XX%") */
  labelText?: string;
  animated?: boolean;
  style?: ViewStyle;
  trackColor?: string;
  borderRadius?: number;
}

export function ProgressBar({
  progress,
  colorStart = '#43E97B',
  colorEnd = '#38F9D7',
  height = 8,
  showLabel = false,
  labelText,
  animated = true,
  style,
  trackColor = '#1E1E3A',
  borderRadius,
}: ProgressBarProps) {
  const clampedProgress = Math.min(100, Math.max(0, progress));
  const animatedWidth = useRef(new Animated.Value(0)).current;
  const radius = borderRadius ?? Math.ceil(height / 2);

  useEffect(() => {
    if (animated) {
      Animated.spring(animatedWidth, {
        toValue: clampedProgress,
        useNativeDriver: false,
        tension: 60,
        friction: 8,
      }).start();
    } else {
      animatedWidth.setValue(clampedProgress);
    }
  }, [clampedProgress, animated, animatedWidth]);

  const widthInterpolated = animatedWidth.interpolate({
    inputRange: [0, 100],
    outputRange: ['0%', '100%'],
    extrapolate: 'clamp',
  });

  return (
    <View style={[styles.container, style]}>
      {showLabel && (
        <View style={styles.labelRow}>
          <Text style={styles.labelText}>{labelText ?? `${Math.round(clampedProgress)}%`}</Text>
        </View>
      )}
      <View
        style={[
          styles.track,
          {
            height,
            borderRadius: radius,
            backgroundColor: trackColor,
          },
        ]}
      >
        <Animated.View
          style={{
            width: widthInterpolated,
            height,
            borderRadius: radius,
            overflow: 'hidden',
          }}
        >
          <LinearGradient
            colors={[colorStart, colorEnd]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={StyleSheet.absoluteFill}
          />
        </Animated.View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
  },
  track: {
    overflow: 'hidden',
  },
  labelRow: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginBottom: 4,
  },
  labelText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
    opacity: 0.8,
  },
});

export default ProgressBar;
