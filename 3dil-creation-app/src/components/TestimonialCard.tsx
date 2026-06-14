import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors, Spacing, FontSize, BorderRadius, Shadows } from '../theme';
import { Testimonial } from '../types';

interface TestimonialCardProps {
  testimonial: Testimonial;
}

const TestimonialCard: React.FC<TestimonialCardProps> = ({ testimonial }) => {
  return (
    <View style={[styles.card, Shadows.small]}>
      <View style={styles.quoteIcon}>
        <Text style={styles.quoteText}>"</Text>
      </View>
      <Text style={styles.review} numberOfLines={4}>{testimonial.review}</Text>
      <View style={styles.stars}>
        {Array.from({ length: 5 }).map((_, i) => (
          <Ionicons
            key={i}
            name={i < testimonial.rating ? 'star' : 'star-outline'}
            size={14}
            color={Colors.accent}
          />
        ))}
      </View>
      <View style={styles.footer}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>{testimonial.avatar}</Text>
        </View>
        <View>
          <Text style={styles.name}>{testimonial.name}</Text>
          <Text style={styles.location}>{testimonial.location}</Text>
          <Text style={styles.service}>{testimonial.service}</Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: Colors.card,
    borderRadius: BorderRadius.lg,
    padding: Spacing.md,
    width: 280,
    marginRight: Spacing.md,
  },
  quoteIcon: {
    marginBottom: Spacing.xs,
  },
  quoteText: {
    fontSize: 48,
    color: Colors.primary,
    lineHeight: 40,
    fontWeight: '900',
  },
  review: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
    lineHeight: 22,
    marginBottom: Spacing.sm,
    fontStyle: 'italic',
  },
  stars: {
    flexDirection: 'row',
    gap: 2,
    marginBottom: Spacing.sm,
  },
  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
    paddingTop: Spacing.sm,
  },
  avatar: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: '#FFF3EE',
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    fontSize: 22,
  },
  name: {
    fontSize: FontSize.sm,
    fontWeight: '700',
    color: Colors.textPrimary,
  },
  location: {
    fontSize: FontSize.xs,
    color: Colors.textSecondary,
  },
  service: {
    fontSize: FontSize.xs,
    color: Colors.primary,
    fontWeight: '600',
  },
});

export default TestimonialCard;
