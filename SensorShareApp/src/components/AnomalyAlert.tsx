import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { AnomalyEvent } from '../types';
import { COLORS } from '../constants';

interface Props {
  anomaly: AnomalyEvent;
}

const SEVERITY_COLOR: Record<AnomalyEvent['severity'], string> = {
  low:      COLORS.primary,
  medium:   COLORS.warning,
  high:     '#E67E22',
  critical: COLORS.danger,
};

export default function AnomalyAlert({ anomaly }: Props) {
  const opacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.timing(opacity, { toValue: 1, duration: 150, useNativeDriver: true }),
      Animated.delay(3000),
      Animated.timing(opacity, { toValue: 0, duration: 400, useNativeDriver: true }),
    ]).start();
  }, [anomaly.timestamp]);

  const color = SEVERITY_COLOR[anomaly.severity];

  return (
    <Animated.View style={[styles.container, { borderColor: color, opacity }]}>
      <View style={[styles.badge, { backgroundColor: color }]}>
        <Text style={styles.badgeText}>{anomaly.severity.toUpperCase()}</Text>
      </View>
      <View style={styles.body}>
        <Text style={styles.sensor}>{anomaly.sensor} › {anomaly.field}</Text>
        <Text style={styles.desc}>{anomaly.description}</Text>
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderRadius: 8,
    borderWidth: 1.5,
    padding: 10,
    marginBottom: 6,
  },
  badge: {
    borderRadius: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
    marginRight: 10,
  },
  badgeText: { color: '#fff', fontSize: 10, fontWeight: '800' },
  body: { flex: 1 },
  sensor: { color: COLORS.text, fontSize: 13, fontWeight: '600' },
  desc: { color: COLORS.textMuted, fontSize: 11, marginTop: 2 },
});
