import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Colors, Spacing, FontSize, BorderRadius, Shadows } from '../theme';
import { Service } from '../types';
import { LinearGradient } from 'expo-linear-gradient';

interface ServiceCardProps {
  service: Service;
  onPress: () => void;
  compact?: boolean;
}

const ServiceCard: React.FC<ServiceCardProps> = ({ service, onPress, compact = false }) => {
  if (compact) {
    return (
      <TouchableOpacity style={styles.compactCard} onPress={onPress} activeOpacity={0.85}>
        <LinearGradient
          colors={['#FF6B35', '#FF8C42']}
          style={styles.compactIconBg}
        >
          <Text style={styles.compactIcon}>{service.icon}</Text>
        </LinearGradient>
        <Text style={styles.compactName} numberOfLines={2}>{service.name}</Text>
        <Text style={styles.compactPrice}>From &#8377;{service.startingPrice}</Text>
      </TouchableOpacity>
    );
  }

  return (
    <TouchableOpacity style={[styles.card, Shadows.medium]} onPress={onPress} activeOpacity={0.85}>
      <View style={styles.header}>
        <View style={styles.iconContainer}>
          <Text style={styles.icon}>{service.icon}</Text>
        </View>
        <View style={styles.headerText}>
          <Text style={styles.name}>{service.name}</Text>
          <Text style={styles.turnaround}>&#9201; {service.turnaround}</Text>
        </View>
        <View style={styles.priceBadge}>
          <Text style={styles.priceLabel}>From</Text>
          <Text style={styles.price}>&#8377;{service.startingPrice}</Text>
        </View>
      </View>
      <Text style={styles.description} numberOfLines={2}>{service.description}</Text>
      <View style={styles.features}>
        {service.features.slice(0, 3).map((feature, index) => (
          <View key={index} style={styles.featureRow}>
            <Text style={styles.featureCheck}>&#10003;</Text>
            <Text style={styles.featureText}>{feature}</Text>
          </View>
        ))}
      </View>
      <TouchableOpacity style={styles.bookBtn} onPress={onPress}>
        <LinearGradient
          colors={[Colors.primary, '#FF8C42']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={styles.bookBtnGradient}
        >
          <Text style={styles.bookBtnText}>Book Now &#8594;</Text>
        </LinearGradient>
      </TouchableOpacity>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: Colors.card,
    borderRadius: BorderRadius.lg,
    padding: Spacing.md,
    marginBottom: Spacing.md,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  iconContainer: {
    width: 52,
    height: 52,
    borderRadius: BorderRadius.md,
    backgroundColor: '#FFF3EE',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: Spacing.sm,
  },
  icon: {
    fontSize: 26,
  },
  headerText: {
    flex: 1,
  },
  name: {
    fontSize: FontSize.lg,
    fontWeight: '700',
    color: Colors.textPrimary,
    marginBottom: 2,
  },
  turnaround: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
  },
  priceBadge: {
    alignItems: 'center',
    backgroundColor: '#FFF3EE',
    borderRadius: BorderRadius.sm,
    paddingHorizontal: Spacing.sm,
    paddingVertical: 4,
  },
  priceLabel: {
    fontSize: 10,
    color: Colors.textSecondary,
  },
  price: {
    fontSize: FontSize.lg,
    fontWeight: '700',
    color: Colors.primary,
  },
  description: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
    lineHeight: 20,
    marginBottom: Spacing.sm,
  },
  features: {
    marginBottom: Spacing.md,
  },
  featureRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  featureCheck: {
    color: Colors.success,
    fontSize: FontSize.sm,
    marginRight: Spacing.xs,
    fontWeight: '700',
  },
  featureText: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
  },
  bookBtn: {
    borderRadius: BorderRadius.md,
    overflow: 'hidden',
  },
  bookBtnGradient: {
    paddingVertical: 12,
    alignItems: 'center',
  },
  bookBtnText: {
    color: Colors.white,
    fontWeight: '700',
    fontSize: FontSize.md,
  },
  // Compact styles
  compactCard: {
    width: 110,
    backgroundColor: Colors.card,
    borderRadius: BorderRadius.lg,
    padding: Spacing.sm,
    alignItems: 'center',
    marginRight: Spacing.sm,
    ...Shadows.small,
  },
  compactIconBg: {
    width: 52,
    height: 52,
    borderRadius: BorderRadius.md,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.xs,
  },
  compactIcon: {
    fontSize: 26,
  },
  compactName: {
    fontSize: FontSize.xs,
    fontWeight: '600',
    color: Colors.textPrimary,
    textAlign: 'center',
    marginBottom: 4,
  },
  compactPrice: {
    fontSize: FontSize.xs,
    color: Colors.primary,
    fontWeight: '700',
  },
});

export default ServiceCard;
