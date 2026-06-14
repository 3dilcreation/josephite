import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors, Spacing, FontSize, BorderRadius, Shadows } from '../theme';
import { Product } from '../types';
import { LinearGradient } from 'expo-linear-gradient';

interface ProductCardProps {
  product: Product;
  onPress: () => void;
  onAddToCart: () => void;
}

const ProductCard: React.FC<ProductCardProps> = ({ product, onPress, onAddToCart }) => {
  const discount = Math.round(((product.originalPrice - product.price) / product.originalPrice) * 100);

  return (
    <TouchableOpacity style={[styles.card, Shadows.small]} onPress={onPress} activeOpacity={0.9}>
      {product.isBestseller && (
        <View style={styles.bestsellerBadge}>
          <Text style={styles.bestsellerText}>BESTSELLER</Text>
        </View>
      )}
      {discount > 0 && (
        <View style={styles.discountBadge}>
          <Text style={styles.discountText}>{discount}% OFF</Text>
        </View>
      )}
      <LinearGradient
        colors={['#FFF3EE', '#FFF8F5']}
        style={styles.imageContainer}
      >
        <Text style={styles.productEmoji}>{product.emoji}</Text>
      </LinearGradient>
      <View style={styles.content}>
        <Text style={styles.category}>{product.category}</Text>
        <Text style={styles.name} numberOfLines={2}>{product.name}</Text>
        <View style={styles.ratingRow}>
          <Ionicons name="star" size={12} color={Colors.accent} />
          <Text style={styles.rating}>{product.rating}</Text>
          <Text style={styles.reviewCount}>({product.reviewCount})</Text>
        </View>
        <View style={styles.priceRow}>
          <Text style={styles.price}>&#8377;{product.price.toLocaleString('en-IN')}</Text>
          {product.originalPrice > product.price && (
            <Text style={styles.originalPrice}>&#8377;{product.originalPrice.toLocaleString('en-IN')}</Text>
          )}
        </View>
        <TouchableOpacity style={styles.addBtn} onPress={onAddToCart} activeOpacity={0.85}>
          <LinearGradient
            colors={[Colors.primary, '#FF8C42']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.addBtnGradient}
          >
            <Ionicons name="cart-outline" size={14} color={Colors.white} />
            <Text style={styles.addBtnText}>Add to Cart</Text>
          </LinearGradient>
        </TouchableOpacity>
      </View>
      {!product.inStock && (
        <View style={styles.outOfStock}>
          <Text style={styles.outOfStockText}>Out of Stock</Text>
        </View>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  card: {
    flex: 1,
    backgroundColor: Colors.card,
    borderRadius: BorderRadius.lg,
    overflow: 'hidden',
    margin: 6,
  },
  imageContainer: {
    height: 140,
    alignItems: 'center',
    justifyContent: 'center',
  },
  productEmoji: {
    fontSize: 64,
  },
  bestsellerBadge: {
    position: 'absolute',
    top: 8,
    left: 8,
    backgroundColor: Colors.accent,
    borderRadius: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
    zIndex: 1,
  },
  bestsellerText: {
    fontSize: 9,
    fontWeight: '800',
    color: Colors.secondary,
  },
  discountBadge: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: Colors.error,
    borderRadius: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
    zIndex: 1,
  },
  discountText: {
    fontSize: 9,
    fontWeight: '800',
    color: Colors.white,
  },
  content: {
    padding: Spacing.sm,
  },
  category: {
    fontSize: FontSize.xs,
    color: Colors.primary,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 2,
  },
  name: {
    fontSize: FontSize.sm,
    fontWeight: '700',
    color: Colors.textPrimary,
    marginBottom: 4,
    lineHeight: 18,
  },
  ratingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
    gap: 2,
  },
  rating: {
    fontSize: FontSize.xs,
    fontWeight: '700',
    color: Colors.textPrimary,
  },
  reviewCount: {
    fontSize: FontSize.xs,
    color: Colors.textSecondary,
  },
  priceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    marginBottom: Spacing.sm,
  },
  price: {
    fontSize: FontSize.lg,
    fontWeight: '800',
    color: Colors.primary,
  },
  originalPrice: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
    textDecorationLine: 'line-through',
  },
  addBtn: {
    borderRadius: BorderRadius.sm,
    overflow: 'hidden',
  },
  addBtnGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    gap: 4,
  },
  addBtnText: {
    color: Colors.white,
    fontSize: FontSize.xs,
    fontWeight: '700',
  },
  outOfStock: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(255,255,255,0.8)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  outOfStockText: {
    fontSize: FontSize.md,
    fontWeight: '700',
    color: Colors.textSecondary,
    backgroundColor: '#E5E7EB',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.sm,
  },
});

export default ProductCard;
