import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Colors, FontSize, BorderRadius, Spacing } from '../theme';

interface PriceTagProps {
  price: number;
  originalPrice?: number;
  size?: 'sm' | 'md' | 'lg';
  showDiscount?: boolean;
}

const PriceTag: React.FC<PriceTagProps> = ({ price, originalPrice, size = 'md', showDiscount = true }) => {
  const discount = originalPrice ? Math.round(((originalPrice - price) / originalPrice) * 100) : 0;

  const priceSizes = { sm: FontSize.md, md: FontSize.xl, lg: FontSize.xxxl };
  const originalSizes = { sm: FontSize.xs, md: FontSize.sm, lg: FontSize.md };

  return (
    <View style={styles.container}>
      <Text style={[styles.price, { fontSize: priceSizes[size] }]}>
        &#8377;{price.toLocaleString('en-IN')}
      </Text>
      {originalPrice && originalPrice > price && (
        <Text style={[styles.originalPrice, { fontSize: originalSizes[size] }]}>
          &#8377;{originalPrice.toLocaleString('en-IN')}
        </Text>
      )}
      {showDiscount && discount > 0 && (
        <View style={styles.discountBadge}>
          <Text style={styles.discountText}>{discount}% OFF</Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    flexWrap: 'wrap',
  },
  price: {
    fontWeight: '800',
    color: Colors.primary,
  },
  originalPrice: {
    color: Colors.textSecondary,
    textDecorationLine: 'line-through',
    fontWeight: '400',
  },
  discountBadge: {
    backgroundColor: '#DCFCE7',
    borderRadius: BorderRadius.sm,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  discountText: {
    fontSize: 11,
    fontWeight: '700',
    color: Colors.success,
  },
});

export default PriceTag;
