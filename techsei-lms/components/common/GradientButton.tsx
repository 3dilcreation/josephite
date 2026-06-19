import React from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  ViewStyle,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Colors } from '../../constants/colors';

interface GradientButtonProps {
  label: string;
  onPress: () => void;
  loading?: boolean;
  disabled?: boolean;
  gradient?: string[];
  style?: ViewStyle;
  size?: 'sm' | 'md' | 'lg';
}

export function GradientButton({
  label,
  onPress,
  loading = false,
  disabled = false,
  gradient = Colors.gradients.primary,
  style,
  size = 'md',
}: GradientButtonProps) {
  const height = size === 'sm' ? 40 : size === 'lg' ? 60 : 52;
  const fontSize = size === 'sm' ? 13 : size === 'lg' ? 18 : 16;

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.85}
      style={[{ opacity: disabled ? 0.5 : 1 }, style]}
    >
      <LinearGradient
        colors={gradient as any}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
        style={[styles.gradient, { height, borderRadius: height / 2 }]}
      >
        {loading ? (
          <ActivityIndicator color="#fff" size="small" />
        ) : (
          <Text style={[styles.label, { fontSize }]}>{label}</Text>
        )}
      </LinearGradient>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  gradient: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  label: {
    color: '#fff',
    fontWeight: '700',
    letterSpacing: 0.3,
  },
});
