import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ActivityIndicator, ViewStyle, TextStyle } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Colors, BorderRadius, FontSize, Spacing } from '../theme';

interface CustomButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  disabled?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
  icon?: React.ReactNode;
}

const CustomButton: React.FC<CustomButtonProps> = ({
  title,
  onPress,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  style,
  textStyle,
  icon,
}) => {
  const sizeStyles = {
    sm: { paddingVertical: 8, paddingHorizontal: 16, borderRadius: BorderRadius.sm },
    md: { paddingVertical: 14, paddingHorizontal: 24, borderRadius: BorderRadius.md },
    lg: { paddingVertical: 18, paddingHorizontal: 32, borderRadius: BorderRadius.lg },
  };

  const textSizes = {
    sm: FontSize.sm,
    md: FontSize.md,
    lg: FontSize.lg,
  };

  if (variant === 'primary') {
    return (
      <TouchableOpacity
        onPress={onPress}
        disabled={disabled || loading}
        style={[styles.buttonBase, sizeStyles[size], (disabled || loading) && styles.disabled, style]}
        activeOpacity={0.8}
      >
        <LinearGradient
          colors={[Colors.primary, '#FF8C42']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={[styles.gradient, { borderRadius: sizeStyles[size].borderRadius }]}
        >
          {loading ? (
            <ActivityIndicator color={Colors.white} size="small" />
          ) : (
            <>
              {icon}
              <Text style={[styles.primaryText, { fontSize: textSizes[size] }, textStyle]}>{title}</Text>
            </>
          )}
        </LinearGradient>
      </TouchableOpacity>
    );
  }

  if (variant === 'secondary') {
    return (
      <TouchableOpacity
        onPress={onPress}
        disabled={disabled || loading}
        style={[styles.buttonBase, styles.secondaryButton, sizeStyles[size], (disabled || loading) && styles.disabled, style]}
        activeOpacity={0.8}
      >
        {loading ? (
          <ActivityIndicator color={Colors.white} size="small" />
        ) : (
          <>
            {icon}
            <Text style={[styles.primaryText, { fontSize: textSizes[size] }, textStyle]}>{title}</Text>
          </>
        )}
      </TouchableOpacity>
    );
  }

  if (variant === 'outline') {
    return (
      <TouchableOpacity
        onPress={onPress}
        disabled={disabled || loading}
        style={[styles.buttonBase, styles.outlineButton, sizeStyles[size], (disabled || loading) && styles.disabled, style]}
        activeOpacity={0.8}
      >
        {loading ? (
          <ActivityIndicator color={Colors.primary} size="small" />
        ) : (
          <>
            {icon}
            <Text style={[styles.outlineText, { fontSize: textSizes[size] }, textStyle]}>{title}</Text>
          </>
        )}
      </TouchableOpacity>
    );
  }

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled || loading}
      style={[styles.buttonBase, sizeStyles[size], (disabled || loading) && styles.disabled, style]}
      activeOpacity={0.6}
    >
      {loading ? (
        <ActivityIndicator color={Colors.primary} size="small" />
      ) : (
        <>
          {icon}
          <Text style={[styles.ghostText, { fontSize: textSizes[size] }, textStyle]}>{title}</Text>
        </>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  buttonBase: {
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: Spacing.xs,
  },
  gradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: 24,
    gap: Spacing.xs,
  },
  secondaryButton: {
    backgroundColor: Colors.secondary,
  },
  outlineButton: {
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: Colors.primary,
  },
  primaryText: {
    color: Colors.white,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  outlineText: {
    color: Colors.primary,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  ghostText: {
    color: Colors.primary,
    fontWeight: '600',
  },
  disabled: {
    opacity: 0.5,
  },
});

export default CustomButton;
