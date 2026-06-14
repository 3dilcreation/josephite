import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';
import { products } from '../data/products';
import { useCart } from '../context/CartContext';

const { width } = Dimensions.get('window');
type Nav = NativeStackNavigationProp<RootStackParamList>;
type Route = RouteProp<RootStackParamList, 'ProductDetail'>;

const ProductDetailScreen: React.FC = () => {
  const navigation = useNavigation<Nav>();
  const route = useRoute<Route>();
  const { addToCart } = useCart();
  const product = products.find(p => p.id === route.params.productId) || products[0];

  const [selectedMaterial, setSelectedMaterial] = useState(product.material);
  const [selectedColor, setSelectedColor] = useState(product.colors[0]);
  const [selectedSize, setSelectedSize] = useState(product.sizes[0]);
  const [quantity, setQuantity] = useState(1);
  const [addedToCart, setAddedToCart] = useState(false);

  const discount = Math.round(((product.originalPrice - product.price) / product.originalPrice) * 100);

  const handleAddToCart = () => {
    addToCart(product, { quantity, selectedMaterial, selectedColor, selectedSize });
    setAddedToCart(true);
    setTimeout(() => setAddedToCart(false), 2000);
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color={Colors.textPrimary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle} numberOfLines={1}>{product.name}</Text>
        <TouchableOpacity style={styles.cartBtn} onPress={() => navigation.navigate('Cart')}>
          <Ionicons name="cart-outline" size={24} color={Colors.primary} />
        </TouchableOpacity>
      </View>

      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Product Image */}
        <LinearGradient
          colors={['#F0F9FF', '#E0F2FE']}
          style={styles.imageBox}
        >
          <Text style={styles.productEmoji}>{product.emoji}</Text>
          {product.isBestseller && (
            <View style={styles.bestsellerBadge}>
              <Text style={styles.bestsellerText}>⭐ Bestseller</Text>
            </View>
          )}
          <TouchableOpacity
            style={styles.arBtn}
            onPress={() => navigation.navigate('ARViewer', { productId: product.id })}
          >
            <Ionicons name="camera-outline" size={18} color={Colors.white} />
            <Text style={styles.arBtnText}>View in AR</Text>
          </TouchableOpacity>
        </LinearGradient>

        <View style={styles.content}>
          {/* Price */}
          <View style={styles.priceRow}>
            <Text style={styles.price}>₹{product.price.toLocaleString()}</Text>
            <Text style={styles.original}>₹{product.originalPrice.toLocaleString()}</Text>
            <View style={styles.discountBadge}>
              <Text style={styles.discountText}>{discount}% OFF</Text>
            </View>
          </View>
          <Text style={styles.name}>{product.name}</Text>

          {/* Rating */}
          <View style={styles.ratingRow}>
            <Ionicons name="star" size={16} color={Colors.accent} />
            <Text style={styles.rating}>{product.rating}</Text>
            <Text style={styles.reviewCount}>({product.reviewCount} reviews)</Text>
            <View style={styles.stockBadge}>
              <Ionicons name={product.inStock ? 'checkmark-circle' : 'close-circle'} size={14} color={product.inStock ? Colors.success : Colors.error} />
              <Text style={[styles.stockText, { color: product.inStock ? Colors.success : Colors.error }]}>
                {product.inStock ? 'In Stock' : 'Out of Stock'}
              </Text>
            </View>
          </View>

          <Text style={styles.description}>{product.description}</Text>

          {/* Material */}
          <View style={styles.optionSection}>
            <Text style={styles.optionLabel}>Material</Text>
            <View style={styles.chipRow}>
              {['PLA', 'ABS', 'PETG', 'Resin'].map(m => (
                <TouchableOpacity
                  key={m}
                  style={[styles.chip, selectedMaterial === m && styles.chipActive]}
                  onPress={() => setSelectedMaterial(m)}
                >
                  <Text style={[styles.chipText, selectedMaterial === m && styles.chipTextActive]}>{m}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Color */}
          <View style={styles.optionSection}>
            <Text style={styles.optionLabel}>Color: <Text style={styles.selectedLabel}>{selectedColor}</Text></Text>
            <View style={styles.colorRow}>
              {product.colors.map(c => (
                <TouchableOpacity
                  key={c}
                  style={[styles.colorChip, selectedColor === c && styles.colorChipActive]}
                  onPress={() => setSelectedColor(c)}
                >
                  <Text style={styles.colorChipText}>{c}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Size */}
          <View style={styles.optionSection}>
            <Text style={styles.optionLabel}>Size</Text>
            <View style={styles.chipRow}>
              {product.sizes.map(s => (
                <TouchableOpacity
                  key={s}
                  style={[styles.chip, selectedSize === s && styles.chipActive]}
                  onPress={() => setSelectedSize(s)}
                >
                  <Text style={[styles.chipText, selectedSize === s && styles.chipTextActive]}>{s}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Quantity */}
          <View style={styles.optionSection}>
            <Text style={styles.optionLabel}>Quantity</Text>
            <View style={styles.quantityRow}>
              <TouchableOpacity style={styles.qtyBtn} onPress={() => setQuantity(q => Math.max(1, q - 1))}>
                <Ionicons name="remove" size={20} color={Colors.primary} />
              </TouchableOpacity>
              <Text style={styles.qty}>{quantity}</Text>
              <TouchableOpacity style={styles.qtyBtn} onPress={() => setQuantity(q => q + 1)}>
                <Ionicons name="add" size={20} color={Colors.primary} />
              </TouchableOpacity>
            </View>
          </View>

          {/* Features */}
          <View style={styles.featuresBox}>
            <Text style={styles.optionLabel}>Features</Text>
            {product.features.map((f, i) => (
              <View key={i} style={styles.featureRow}>
                <Ionicons name="checkmark-circle" size={16} color={Colors.success} />
                <Text style={styles.featureText}>{f}</Text>
              </View>
            ))}
          </View>

          {/* Custom Order */}
          {product.isCustomizable && (
            <TouchableOpacity
              style={styles.customizeBtn}
              onPress={() => navigation.navigate('CustomOrder', {})}
            >
              <Ionicons name="construct-outline" size={18} color={Colors.primary} />
              <Text style={styles.customizeBtnText}>Customize This Product</Text>
            </TouchableOpacity>
          )}
        </View>
      </ScrollView>

      {/* Bottom Buttons */}
      <View style={styles.bottomBar}>
        <TouchableOpacity
          style={[styles.addCartBtn, addedToCart && styles.addedBtn]}
          onPress={handleAddToCart}
        >
          <Ionicons name={addedToCart ? 'checkmark' : 'cart-outline'} size={20} color={Colors.primary} />
          <Text style={styles.addCartText}>{addedToCart ? 'Added!' : 'Add to Cart'}</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.buyNowBtn}
          onPress={() => { handleAddToCart(); navigation.navigate('Checkout'); }}
        >
          <LinearGradient colors={[Colors.primary, '#FF8C42']} style={styles.buyNowGradient}>
            <Text style={styles.buyNowText}>Buy Now</Text>
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { flexDirection: 'row', alignItems: 'center', paddingTop: 52, paddingHorizontal: Spacing.md, paddingBottom: 12, backgroundColor: Colors.card, ...Shadows.small },
  backBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: Colors.background, alignItems: 'center', justifyContent: 'center' },
  headerTitle: { flex: 1, fontSize: FontSize.lg, fontWeight: '800', color: Colors.textPrimary, marginHorizontal: 10 },
  cartBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: Colors.background, alignItems: 'center', justifyContent: 'center' },
  imageBox: { height: 280, alignItems: 'center', justifyContent: 'center', position: 'relative' },
  productEmoji: { fontSize: 120 },
  bestsellerBadge: { position: 'absolute', top: 16, left: 16, backgroundColor: Colors.accent, paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12 },
  bestsellerText: { fontSize: FontSize.sm, fontWeight: '800', color: Colors.secondary },
  arBtn: { position: 'absolute', bottom: 16, right: 16, flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: Colors.secondary, paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20 },
  arBtnText: { color: Colors.white, fontSize: FontSize.sm, fontWeight: '700' },
  content: { padding: Spacing.md },
  priceRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 },
  price: { fontSize: FontSize.xxxl, fontWeight: '900', color: Colors.primary },
  original: { fontSize: FontSize.lg, color: Colors.textLight, textDecorationLine: 'line-through' },
  discountBadge: { backgroundColor: Colors.success + '20', paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8 },
  discountText: { color: Colors.success, fontSize: FontSize.sm, fontWeight: '800' },
  name: { fontSize: FontSize.xxl, fontWeight: '800', color: Colors.textPrimary, marginBottom: 12 },
  ratingRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 12 },
  rating: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary },
  reviewCount: { fontSize: FontSize.md, color: Colors.textSecondary },
  stockBadge: { flexDirection: 'row', alignItems: 'center', gap: 4, marginLeft: 'auto' },
  stockText: { fontSize: FontSize.sm, fontWeight: '600' },
  description: { fontSize: FontSize.md, color: Colors.textSecondary, lineHeight: 22, marginBottom: 16 },
  optionSection: { marginBottom: 16 },
  optionLabel: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary, marginBottom: 8 },
  selectedLabel: { color: Colors.primary, fontWeight: '600' },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: { paddingHorizontal: 16, paddingVertical: 8, borderRadius: BorderRadius.round, borderWidth: 1.5, borderColor: Colors.border, backgroundColor: Colors.card },
  chipActive: { borderColor: Colors.primary, backgroundColor: Colors.primary + '15' },
  chipText: { fontSize: FontSize.md, color: Colors.textSecondary, fontWeight: '600' },
  chipTextActive: { color: Colors.primary },
  colorRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  colorChip: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8, borderWidth: 1.5, borderColor: Colors.border, backgroundColor: Colors.card },
  colorChipActive: { borderColor: Colors.primary, backgroundColor: Colors.primary + '15' },
  colorChipText: { fontSize: FontSize.sm, color: Colors.textPrimary, fontWeight: '600' },
  quantityRow: { flexDirection: 'row', alignItems: 'center', gap: 16 },
  qtyBtn: { width: 40, height: 40, borderRadius: 20, borderWidth: 1.5, borderColor: Colors.primary, alignItems: 'center', justifyContent: 'center' },
  qty: { fontSize: FontSize.xxl, fontWeight: '800', color: Colors.textPrimary, minWidth: 32, textAlign: 'center' },
  featuresBox: { backgroundColor: Colors.background, borderRadius: BorderRadius.md, padding: 14, marginBottom: 16 },
  featureRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 },
  featureText: { fontSize: FontSize.md, color: Colors.textSecondary },
  customizeBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, borderWidth: 2, borderColor: Colors.primary, borderRadius: BorderRadius.lg, paddingVertical: 14, marginBottom: 16, borderStyle: 'dashed' },
  customizeBtnText: { color: Colors.primary, fontWeight: '700', fontSize: FontSize.lg },
  bottomBar: { flexDirection: 'row', gap: 12, padding: 16, backgroundColor: Colors.card, borderTopWidth: 1, borderTopColor: Colors.border, paddingBottom: 28 },
  addCartBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, borderWidth: 2, borderColor: Colors.primary, borderRadius: BorderRadius.lg, paddingVertical: 16 },
  addedBtn: { borderColor: Colors.success, backgroundColor: Colors.success + '10' },
  addCartText: { color: Colors.primary, fontWeight: '800', fontSize: FontSize.lg },
  buyNowBtn: { flex: 1.5, borderRadius: BorderRadius.lg, overflow: 'hidden' },
  buyNowGradient: { paddingVertical: 16, alignItems: 'center', borderRadius: BorderRadius.lg },
  buyNowText: { color: Colors.white, fontWeight: '800', fontSize: FontSize.lg },
});

export default ProductDetailScreen;
