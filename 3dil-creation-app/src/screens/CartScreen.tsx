import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, TextInput,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';
import { useCart } from '../context/CartContext';

type Nav = NativeStackNavigationProp<RootStackParamList>;

const CartScreen: React.FC = () => {
  const navigation = useNavigation<Nav>();
  const { cartItems, removeFromCart, updateQuantity, cartTotal, discount, applyCoupon, couponCode, removeCoupon } = useCart();
  const [couponInput, setCouponInput] = useState('');
  const [couponError, setCouponError] = useState('');

  const delivery = cartTotal > 999 ? 0 : 99;
  const discountAmount = Math.round(cartTotal * discount / 100);
  const total = cartTotal - discountAmount + delivery;

  const handleApplyCoupon = () => {
    const success = applyCoupon(couponInput.trim().toUpperCase());
    if (!success) setCouponError('Invalid coupon code');
    else { setCouponError(''); setCouponInput(''); }
  };

  if (cartItems.length === 0) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
            <Ionicons name="close" size={24} color={Colors.textPrimary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>My Cart</Text>
          <View style={{ width: 36 }} />
        </View>
        <View style={styles.empty}>
          <Text style={styles.emptyEmoji}>🛒</Text>
          <Text style={styles.emptyTitle}>Your cart is empty</Text>
          <Text style={styles.emptyDesc}>Add some amazing 3D printed products to get started!</Text>
          <TouchableOpacity style={styles.shopBtn} onPress={() => navigation.goBack()}>
            <Text style={styles.shopBtnText}>Browse Products</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="close" size={24} color={Colors.textPrimary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>My Cart ({cartItems.length})</Text>
        <View style={{ width: 36 }} />
      </View>

      <ScrollView showsVerticalScrollIndicator={false}>
        {cartItems.map(item => (
          <View key={item.product.id} style={styles.cartItem}>
            <View style={styles.itemImageBox}>
              <Text style={styles.itemEmoji}>{item.product.emoji}</Text>
            </View>
            <View style={styles.itemDetails}>
              <Text style={styles.itemName} numberOfLines={2}>{item.product.name}</Text>
              <Text style={styles.itemMeta}>{item.selectedMaterial} · {item.selectedColor} · {item.selectedSize}</Text>
              <Text style={styles.itemPrice}>₹{(item.product.price * item.quantity).toLocaleString()}</Text>
            </View>
            <View style={styles.itemActions}>
              <TouchableOpacity style={styles.removeBtn} onPress={() => removeFromCart(item.product.id)}>
                <Ionicons name="trash-outline" size={18} color={Colors.error} />
              </TouchableOpacity>
              <View style={styles.qtyControl}>
                <TouchableOpacity style={styles.qtyBtn} onPress={() => updateQuantity(item.product.id, item.quantity - 1)}>
                  <Ionicons name="remove" size={16} color={Colors.primary} />
                </TouchableOpacity>
                <Text style={styles.qty}>{item.quantity}</Text>
                <TouchableOpacity style={styles.qtyBtn} onPress={() => updateQuantity(item.product.id, item.quantity + 1)}>
                  <Ionicons name="add" size={16} color={Colors.primary} />
                </TouchableOpacity>
              </View>
            </View>
          </View>
        ))}

        {/* Coupon */}
        <View style={styles.couponSection}>
          <Text style={styles.couponTitle}>Have a coupon?</Text>
          {couponCode ? (
            <View style={styles.appliedCoupon}>
              <Ionicons name="pricetag" size={18} color={Colors.success} />
              <Text style={styles.appliedCouponText}>{couponCode} — {discount}% OFF applied!</Text>
              <TouchableOpacity onPress={removeCoupon}>
                <Ionicons name="close-circle" size={20} color={Colors.error} />
              </TouchableOpacity>
            </View>
          ) : (
            <View style={styles.couponRow}>
              <TextInput
                style={styles.couponInput}
                placeholder="Enter coupon code"
                value={couponInput}
                onChangeText={t => { setCouponInput(t); setCouponError(''); }}
                placeholderTextColor={Colors.textLight}
                autoCapitalize="characters"
              />
              <TouchableOpacity style={styles.applyBtn} onPress={handleApplyCoupon}>
                <Text style={styles.applyBtnText}>Apply</Text>
              </TouchableOpacity>
            </View>
          )}
          {couponError ? <Text style={styles.couponError}>{couponError}</Text> : null}
          <Text style={styles.couponHint}>Try: FIRST10, DIWALI20, 3DIL15, WELCOME25</Text>
        </View>

        {/* Price Breakdown */}
        <View style={styles.priceCard}>
          <Text style={styles.priceTitle}>Order Summary</Text>
          <View style={styles.priceRow}>
            <Text style={styles.priceLabel}>Subtotal</Text>
            <Text style={styles.priceValue}>₹{cartTotal.toLocaleString()}</Text>
          </View>
          {discountAmount > 0 && (
            <View style={styles.priceRow}>
              <Text style={[styles.priceLabel, { color: Colors.success }]}>Discount ({discount}%)</Text>
              <Text style={[styles.priceValue, { color: Colors.success }]}>-₹{discountAmount.toLocaleString()}</Text>
            </View>
          )}
          <View style={styles.priceRow}>
            <Text style={styles.priceLabel}>Delivery</Text>
            <Text style={[styles.priceValue, delivery === 0 ? { color: Colors.success } : {}]}>
              {delivery === 0 ? 'FREE' : `₹${delivery}`}
            </Text>
          </View>
          {delivery > 0 && (
            <Text style={styles.freeDeliveryHint}>Add ₹{(1000 - cartTotal).toLocaleString()} more for free delivery</Text>
          )}
          <View style={styles.totalRow}>
            <Text style={styles.totalLabel}>Total</Text>
            <Text style={styles.totalValue}>₹{total.toLocaleString()}</Text>
          </View>
        </View>
      </ScrollView>

      <View style={styles.bottomBar}>
        <View>
          <Text style={styles.bottomTotal}>₹{total.toLocaleString()}</Text>
          <Text style={styles.bottomItems}>{cartItems.length} items</Text>
        </View>
        <TouchableOpacity style={styles.checkoutBtn} onPress={() => navigation.navigate('Checkout')}>
          <LinearGradient colors={[Colors.primary, '#FF8C42']} style={styles.checkoutGradient}>
            <Text style={styles.checkoutText}>Proceed to Checkout</Text>
            <Ionicons name="arrow-forward" size={20} color={Colors.white} />
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingTop: 52, paddingHorizontal: Spacing.md, paddingBottom: 12, backgroundColor: Colors.card, borderBottomWidth: 1, borderBottomColor: Colors.border },
  backBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: Colors.background, alignItems: 'center', justifyContent: 'center' },
  headerTitle: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.textPrimary },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 40 },
  emptyEmoji: { fontSize: 72, marginBottom: 16 },
  emptyTitle: { fontSize: FontSize.xxl, fontWeight: '800', color: Colors.textPrimary, marginBottom: 8 },
  emptyDesc: { fontSize: FontSize.md, color: Colors.textSecondary, textAlign: 'center', marginBottom: 24 },
  shopBtn: { backgroundColor: Colors.primary, paddingHorizontal: 32, paddingVertical: 16, borderRadius: BorderRadius.lg },
  shopBtnText: { color: Colors.white, fontWeight: '800', fontSize: FontSize.lg },
  cartItem: { flexDirection: 'row', backgroundColor: Colors.card, margin: 12, marginBottom: 0, borderRadius: BorderRadius.md, padding: 12, gap: 12, ...Shadows.small },
  itemImageBox: { width: 72, height: 72, borderRadius: BorderRadius.md, backgroundColor: Colors.background, alignItems: 'center', justifyContent: 'center' },
  itemEmoji: { fontSize: 36 },
  itemDetails: { flex: 1 },
  itemName: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary, marginBottom: 4 },
  itemMeta: { fontSize: FontSize.xs, color: Colors.textSecondary, marginBottom: 4 },
  itemPrice: { fontSize: FontSize.lg, fontWeight: '800', color: Colors.primary },
  itemActions: { alignItems: 'flex-end', justifyContent: 'space-between' },
  removeBtn: { padding: 4 },
  qtyControl: { flexDirection: 'row', alignItems: 'center', gap: 8, borderWidth: 1.5, borderColor: Colors.border, borderRadius: 20, paddingHorizontal: 8, paddingVertical: 4 },
  qtyBtn: { padding: 2 },
  qty: { fontSize: FontSize.md, fontWeight: '800', color: Colors.textPrimary, minWidth: 20, textAlign: 'center' },
  couponSection: { margin: 12, backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 16, ...Shadows.small },
  couponTitle: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary, marginBottom: 10 },
  couponRow: { flexDirection: 'row', gap: 10 },
  couponInput: { flex: 1, borderWidth: 1.5, borderColor: Colors.border, borderRadius: BorderRadius.md, paddingHorizontal: 14, paddingVertical: 10, fontSize: FontSize.md, color: Colors.textPrimary, fontWeight: '700', letterSpacing: 1 },
  applyBtn: { backgroundColor: Colors.primary, paddingHorizontal: 20, borderRadius: BorderRadius.md, alignItems: 'center', justifyContent: 'center' },
  applyBtnText: { color: Colors.white, fontWeight: '800', fontSize: FontSize.md },
  appliedCoupon: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: Colors.success + '15', borderRadius: BorderRadius.md, padding: 12, borderWidth: 1, borderColor: Colors.success + '40' },
  appliedCouponText: { flex: 1, color: Colors.success, fontWeight: '700', fontSize: FontSize.md },
  couponError: { color: Colors.error, fontSize: FontSize.sm, marginTop: 6 },
  couponHint: { color: Colors.textLight, fontSize: FontSize.xs, marginTop: 6 },
  priceCard: { margin: 12, backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 16, ...Shadows.small },
  priceTitle: { fontSize: FontSize.lg, fontWeight: '800', color: Colors.textPrimary, marginBottom: 12, paddingBottom: 10, borderBottomWidth: 1, borderBottomColor: Colors.border },
  priceRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8 },
  priceLabel: { fontSize: FontSize.md, color: Colors.textSecondary },
  priceValue: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary },
  freeDeliveryHint: { fontSize: FontSize.xs, color: Colors.primary, fontWeight: '600', marginBottom: 4 },
  totalRow: { flexDirection: 'row', justifyContent: 'space-between', paddingTop: 12, borderTopWidth: 1.5, borderTopColor: Colors.border, marginTop: 8 },
  totalLabel: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.textPrimary },
  totalValue: { fontSize: FontSize.xl, fontWeight: '900', color: Colors.primary },
  bottomBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: 16, backgroundColor: Colors.card, borderTopWidth: 1, borderTopColor: Colors.border, paddingBottom: 28 },
  bottomTotal: { fontSize: FontSize.xxl, fontWeight: '900', color: Colors.textPrimary },
  bottomItems: { fontSize: FontSize.sm, color: Colors.textSecondary },
  checkoutBtn: { flex: 1, marginLeft: 16, borderRadius: BorderRadius.lg, overflow: 'hidden' },
  checkoutGradient: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 16, gap: 8, borderRadius: BorderRadius.lg },
  checkoutText: { color: Colors.white, fontSize: FontSize.lg, fontWeight: '800' },
});

export default CartScreen;
