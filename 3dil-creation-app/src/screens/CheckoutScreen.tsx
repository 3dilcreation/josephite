import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  TextInput, ActivityIndicator, Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';
import { useCart } from '../context/CartContext';
import { useAuth } from '../context/AuthContext';

const paymentMethods = [
  { id: 'upi', label: 'UPI (GPay, PhonePe, Paytm)', icon: '💳', popular: true },
  { id: 'card', label: 'Credit / Debit Card', icon: '🏦', popular: false },
  { id: 'netbanking', label: 'Net Banking', icon: '🌐', popular: false },
  { id: 'cod', label: 'Cash on Delivery', icon: '💵', popular: false },
];

const CheckoutScreen: React.FC = () => {
  const navigation = useNavigation();
  const { cartItems, cartTotal, discount, clearCart } = useCart();
  const { user } = useAuth();
  const [name, setName] = useState(user?.name || '');
  const [phone, setPhone] = useState(user?.phone || '');
  const [address, setAddress] = useState('');
  const [city, setCity] = useState('');
  const [pincode, setPincode] = useState('');
  const [paymentMethod, setPaymentMethod] = useState('upi');
  const [upiId, setUpiId] = useState('');
  const [placing, setPlacing] = useState(false);
  const [orderPlaced, setOrderPlaced] = useState(false);

  const discountAmount = Math.round(cartTotal * discount / 100);
  const delivery = cartTotal > 999 ? 0 : 99;
  const total = cartTotal - discountAmount + delivery;

  const handlePlaceOrder = async () => {
    if (!name || !phone || !address || !city || !pincode) {
      Alert.alert('Missing Info', 'Please fill in all delivery details.');
      return;
    }
    setPlacing(true);
    await new Promise(r => setTimeout(r, 2500));
    setPlacing(false);
    setOrderPlaced(true);
    clearCart();
  };

  if (orderPlaced) {
    return (
      <View style={styles.successContainer}>
        <View style={styles.successCard}>
          <Text style={styles.successEmoji}>🎉</Text>
          <Text style={styles.successTitle}>Order Placed!</Text>
          <Text style={styles.successDesc}>
            Your order has been confirmed. We'll start printing right away!
          </Text>
          <Text style={styles.successOrderId}>Order #3DIL{Date.now().toString().slice(-6)}</Text>
          <Text style={styles.successETA}>⏱ Estimated delivery: 3–5 business days</Text>
          <TouchableOpacity
            style={styles.trackBtn}
            onPress={() => navigation.navigate('Orders' as never)}
          >
            <LinearGradient colors={[Colors.primary, '#FF8C42']} style={styles.trackBtnGradient}>
              <Text style={styles.trackBtnText}>Track My Order</Text>
            </LinearGradient>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => navigation.navigate('Home' as never)}>
            <Text style={styles.homeLink}>Back to Home</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color={Colors.textPrimary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Checkout</Text>
        <View style={{ width: 36 }} />
      </View>

      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Delivery Details */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📦 Delivery Details</Text>
          <TextInput style={styles.input} placeholder="Full Name" value={name} onChangeText={setName} placeholderTextColor={Colors.textLight} />
          <TextInput style={styles.input} placeholder="Phone Number" value={phone} onChangeText={setPhone} keyboardType="phone-pad" placeholderTextColor={Colors.textLight} />
          <TextInput style={styles.input} placeholder="Address" value={address} onChangeText={setAddress} placeholderTextColor={Colors.textLight} />
          <View style={styles.row}>
            <TextInput style={[styles.input, { flex: 1 }]} placeholder="City" value={city} onChangeText={setCity} placeholderTextColor={Colors.textLight} />
            <TextInput style={[styles.input, { flex: 1 }]} placeholder="PIN Code" value={pincode} onChangeText={setPincode} keyboardType="numeric" maxLength={6} placeholderTextColor={Colors.textLight} />
          </View>
        </View>

        {/* Order Summary */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🛒 Order Summary</Text>
          {cartItems.map(item => (
            <View key={item.product.id} style={styles.orderItem}>
              <Text style={styles.orderItemEmoji}>{item.product.emoji}</Text>
              <View style={{ flex: 1 }}>
                <Text style={styles.orderItemName} numberOfLines={1}>{item.product.name}</Text>
                <Text style={styles.orderItemMeta}>x{item.quantity} · {item.selectedMaterial}</Text>
              </View>
              <Text style={styles.orderItemPrice}>₹{(item.product.price * item.quantity).toLocaleString()}</Text>
            </View>
          ))}
          <View style={styles.divider} />
          <View style={styles.priceRow}>
            <Text style={styles.priceLabel}>Subtotal</Text>
            <Text style={styles.priceValue}>₹{cartTotal.toLocaleString()}</Text>
          </View>
          {discountAmount > 0 && (
            <View style={styles.priceRow}>
              <Text style={[styles.priceLabel, { color: Colors.success }]}>Discount</Text>
              <Text style={[styles.priceValue, { color: Colors.success }]}>-₹{discountAmount.toLocaleString()}</Text>
            </View>
          )}
          <View style={styles.priceRow}>
            <Text style={styles.priceLabel}>Delivery</Text>
            <Text style={styles.priceValue}>{delivery === 0 ? 'FREE' : `₹${delivery}`}</Text>
          </View>
          <View style={[styles.priceRow, styles.totalRow]}>
            <Text style={styles.totalLabel}>Total</Text>
            <Text style={styles.totalValue}>₹{total.toLocaleString()}</Text>
          </View>
        </View>

        {/* Payment Method */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>💳 Payment Method</Text>
          {paymentMethods.map(m => (
            <TouchableOpacity
              key={m.id}
              style={[styles.paymentOption, paymentMethod === m.id && styles.paymentOptionActive]}
              onPress={() => setPaymentMethod(m.id)}
            >
              <Text style={styles.paymentIcon}>{m.icon}</Text>
              <Text style={[styles.paymentLabel, paymentMethod === m.id && { color: Colors.primary }]}>{m.label}</Text>
              {m.popular && <View style={styles.popularBadge}><Text style={styles.popularText}>Popular</Text></View>}
              <View style={[styles.radio, paymentMethod === m.id && styles.radioActive]}>
                {paymentMethod === m.id && <View style={styles.radioInner} />}
              </View>
            </TouchableOpacity>
          ))}
          {paymentMethod === 'upi' && (
            <TextInput
              style={[styles.input, { marginTop: 8 }]}
              placeholder="Enter UPI ID (e.g. name@upi)"
              value={upiId}
              onChangeText={setUpiId}
              placeholderTextColor={Colors.textLight}
            />
          )}
        </View>
      </ScrollView>

      <View style={styles.bottomBar}>
        <View>
          <Text style={styles.totalLabel}>₹{total.toLocaleString()}</Text>
          <Text style={styles.taxNote}>Incl. all taxes</Text>
        </View>
        <TouchableOpacity
          style={styles.placeOrderBtn}
          onPress={handlePlaceOrder}
          disabled={placing}
        >
          <LinearGradient colors={[Colors.primary, '#FF8C42']} style={styles.placeOrderGradient}>
            {placing ? (
              <ActivityIndicator color={Colors.white} />
            ) : (
              <Text style={styles.placeOrderText}>Place Order 🚀</Text>
            )}
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  successContainer: { flex: 1, backgroundColor: Colors.primary, alignItems: 'center', justifyContent: 'center', padding: 24 },
  successCard: { backgroundColor: Colors.card, borderRadius: 24, padding: 32, alignItems: 'center', width: '100%', ...Shadows.large },
  successEmoji: { fontSize: 80, marginBottom: 16 },
  successTitle: { fontSize: FontSize.xxxl, fontWeight: '900', color: Colors.textPrimary, marginBottom: 8 },
  successDesc: { fontSize: FontSize.md, color: Colors.textSecondary, textAlign: 'center', lineHeight: 22, marginBottom: 16 },
  successOrderId: { fontSize: FontSize.lg, fontWeight: '800', color: Colors.primary, marginBottom: 4 },
  successETA: { fontSize: FontSize.md, color: Colors.textSecondary, marginBottom: 24 },
  trackBtn: { width: '100%', borderRadius: BorderRadius.lg, overflow: 'hidden', marginBottom: 12 },
  trackBtnGradient: { paddingVertical: 16, alignItems: 'center', borderRadius: BorderRadius.lg },
  trackBtnText: { color: Colors.white, fontWeight: '800', fontSize: FontSize.xl },
  homeLink: { color: Colors.primary, fontWeight: '700', fontSize: FontSize.md },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingTop: 52, paddingHorizontal: Spacing.md, paddingBottom: 12, backgroundColor: Colors.card, borderBottomWidth: 1, borderBottomColor: Colors.border },
  backBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: Colors.background, alignItems: 'center', justifyContent: 'center' },
  headerTitle: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.textPrimary },
  section: { margin: 12, backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 16, ...Shadows.small },
  sectionTitle: { fontSize: FontSize.lg, fontWeight: '800', color: Colors.textPrimary, marginBottom: 14 },
  input: { borderWidth: 1.5, borderColor: Colors.border, borderRadius: BorderRadius.md, paddingHorizontal: 14, paddingVertical: 12, fontSize: FontSize.md, color: Colors.textPrimary, marginBottom: 10 },
  row: { flexDirection: 'row', gap: 10 },
  orderItem: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 8 },
  orderItemEmoji: { fontSize: 28 },
  orderItemName: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary, marginBottom: 2 },
  orderItemMeta: { fontSize: FontSize.xs, color: Colors.textSecondary },
  orderItemPrice: { fontSize: FontSize.md, fontWeight: '800', color: Colors.primary },
  divider: { height: 1, backgroundColor: Colors.border, marginVertical: 10 },
  priceRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 6 },
  priceLabel: { fontSize: FontSize.md, color: Colors.textSecondary },
  priceValue: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary },
  totalRow: { borderTopWidth: 1.5, borderTopColor: Colors.border, marginTop: 8, paddingTop: 12 },
  totalLabel: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.textPrimary },
  totalValue: { fontSize: FontSize.xl, fontWeight: '900', color: Colors.primary },
  paymentOption: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 14, paddingHorizontal: 14, borderRadius: BorderRadius.md, borderWidth: 1.5, borderColor: Colors.border, marginBottom: 10 },
  paymentOptionActive: { borderColor: Colors.primary, backgroundColor: Colors.primary + '08' },
  paymentIcon: { fontSize: 24 },
  paymentLabel: { flex: 1, fontSize: FontSize.md, fontWeight: '600', color: Colors.textPrimary },
  popularBadge: { backgroundColor: Colors.accent, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8 },
  popularText: { fontSize: 9, fontWeight: '800', color: Colors.secondary },
  radio: { width: 20, height: 20, borderRadius: 10, borderWidth: 2, borderColor: Colors.border, alignItems: 'center', justifyContent: 'center' },
  radioActive: { borderColor: Colors.primary },
  radioInner: { width: 10, height: 10, borderRadius: 5, backgroundColor: Colors.primary },
  bottomBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: 16, backgroundColor: Colors.card, borderTopWidth: 1, borderTopColor: Colors.border, paddingBottom: 28 },
  taxNote: { fontSize: FontSize.xs, color: Colors.textSecondary },
  placeOrderBtn: { flex: 1, marginLeft: 16, borderRadius: BorderRadius.lg, overflow: 'hidden' },
  placeOrderGradient: { paddingVertical: 16, alignItems: 'center', borderRadius: BorderRadius.lg },
  placeOrderText: { color: Colors.white, fontSize: FontSize.lg, fontWeight: '800' },
});

export default CheckoutScreen;
