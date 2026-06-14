import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors, FontSize, Spacing } from '../theme';

interface RatingStarsProps {
  rating: number;
  reviewCount?: number;
  size?: number;
  showCount?: boolean;
}

const RatingStars: React.FC<RatingStarsProps> = ({
  rating,
  reviewCount,
  size = 14,
  showCount = true,
}) => {
  return (
    <View style={styles.container}>
      {Array.from({ length: 5 }).map((_, i) => {
        const filled = i < Math.floor(rating);
        const halfFilled = !filled && i < rating;
        return (
          <Ionicons
            key={i}
            name={filled ? 'star' : halfFilled ? 'star-half' : 'star-outline'}
            size={size}
            color={Colors.accent}
          />
        );
      })}
      <Text style={[styles.rating, { fontSize: size - 2 }]}>{rating.toFixed(1)}</Text>
      {showCount && reviewCount !== undefined && (
        <Text style={[styles.count, { fontSize: size - 2 }]}>({reviewCount} reviews)</Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
  },
  rating: {
    fontWeight: '700',
    color: Colors.textPrimary,
    marginLeft: 4,
  },
  count: {
    color: Colors.textSecondary,
  },
});

export default RatingStars;
