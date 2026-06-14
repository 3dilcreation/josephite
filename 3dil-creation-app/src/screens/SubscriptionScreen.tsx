import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';

const plans = [
  {
    id: 'starter',
    name: 'Starter',
    emoji: '🌱',
    monthlyPrice: 499,
    annualPrice: 4990,
    freePrints: 1,
    discount: 10,
    color: '#10B981',
    gradient: ['#10B981', '#059669'] as [string, string],
    features: [
      '1 free print per month (up to 10cm)',
      '10% discount on all orders',
      'Priority customer support',
      'Access to exclusive designs',
      'Monthly newsletter',
    ],
    isPopular: false,
  },
  {
    id: 'pro',
    name: 'Pro',
    emoji: '⭐',
    monthlyPrice: 1299,
    annualPrice: 12990,
    freePrints: 3,
    discount: 20,
    color: Colors.primary,
    gradient: [Colors.primary, '#FF8C42'] as [string, string],
    features: [
      '3 free prints per month',
      '20% discount on all orders',
      'Free design consultation (1/mo)',
      'AR preview for all products',
      'Design file review',
      'Priority printing queue',
    ],
    isPopular: true,
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    emoji: '💎',
    monthlyPrice: 3499,
    annualPrice: 34990,
    freePrints: 10,
    discount: 30,
    color: '#7C3AED',
    gradient: ['#7C3AED', '#6D28D9'] as [string, string],
    features: [
      '10 free prints per month',
      '30% discount on all orders',
      'Dedicated account manager',
      'Monthly delivery to your office',
      'Custom branding on products',
      'Quarterly business review',
      'API access for bulk orders',
    ],
    isPopular: false,
  },
];

const SubscriptionScreen: React.FC = () => {
  const navigation = useNavigation();
  const [isAnnual, setIsAnnual] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState<string | null>(null);

  const handleSubscribe = (planId: string) => {
    setSelectedPlan(planId);
    const plan = plans.find(p => p.id === planId);
    Alert.alert(
      `Subscribe to ${plan?.name}?`,
      `You'll be charged ₹${isAnnual ? plan?.annualPrice.toLocaleString() : plan?.monthlyPrice}/month`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Subscribe Now',
          onPress: () => Alert.alert('🎉 Subscribed!', `Welcome to 3DIL ${plan?.name}! Your benefits are now active.`),
        },
      ]
    );
  };

  return (
    <View style={styles.container}>
      <LinearGradient colors={[Colors.secondary, '#0F3460']} style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="close" size={24} color={Colors.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>3DIL Pro Plans</Text>
        <Text style={styles.headerSubtitle}>Save more, print more, grow more</Text>
        <View style={styles.toggle}>
          <TouchableOpacity
            style={[styles.toggleBtn, !isAnnual && styles.toggleBtnActive]}
            onPress={() => setIsAnnual(false)}
          >
            <Text style={[styles.toggleText, !isAnnual && styles.toggleTextActive]}>Monthly</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.toggleBtn, isAnnual && styles.toggleBtnActive]}
            onPress={() => setIsAnnual(true)}
          >
            <Text style={[styles.toggleText, isAnnual && styles.toggleTextActive]}>Annual</Text>
            <View style={styles.saveBadge}>
              <Text style={styles.saveBadgeText}>2 months FREE</Text>
            </View>
          </TouchableOpacity>
        </View>
      </LinearGradient>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: Spacing.md, paddingBottom: 32 }}>
        {plans.map(plan => (
          <View key={plan.id} style={[styles.planCard, plan.isPopular && styles.planCardPopular]}>
            {plan.isPopular && (
              <View style={styles.popularLabel}>
                <Text style={styles.popularLabelText}>⭐ MOST POPULAR</Text>
              </View>
            )}
            <LinearGradient colors={plan.gradient} style={styles.planHeader}>
              <Text style={styles.planEmoji}>{plan.emoji}</Text>
              <Text style={styles.planName}>{plan.name}</Text>
              <View style={styles.priceRow}>
                <Text style={styles.currency}>₹</Text>
                <Text style={styles.price}>{isAnnual ? Math.round(plan.annualPrice / 12) : plan.monthlyPrice}</Text>
                <Text style={styles.period}>/month</Text>
              </View>
              {isAnnual && (
                <Text style={styles.annualNote}>₹{plan.annualPrice.toLocaleString()} billed annually</Text>
              )}
              <View style={styles.highlights}>
                <View style={styles.highlight}>
                  <Text style={styles.highlightValue}>{plan.freePrints}</Text>
                  <Text style={styles.highlightLabel}>Free prints/mo</Text>
                </View>
                <View style={styles.highlightDivider} />
                <View style={styles.highlight}>
                  <Text style={styles.highlightValue}>{plan.discount}%</Text>
                  <Text style={styles.highlightLabel}>Discount</Text>
                </View>
              </View>
            </LinearGradient>

            <View style={styles.featuresContainer}>
              {plan.features.map((feature, i) => (
                <View key={i} style={styles.featureRow}>
                  <Ionicons name="checkmark-circle" size={18} color={plan.color} />
                  <Text style={styles.featureText}>{feature}</Text>
                </View>
              ))}
            </View>

            <TouchableOpacity
              style={[styles.subscribeBtn, { borderColor: plan.color }]}
              onPress={() => handleSubscribe(plan.id)}
            >
              <LinearGradient colors={plan.gradient} style={styles.subscribeBtnGradient}>
                <Text style={styles.subscribeBtnText}>Get {plan.name} Plan</Text>
                <Ionicons name="arrow-forward" size={18} color={Colors.white} />
              </LinearGradient>
            </TouchableOpacity>
          </View>
        ))}

        {/* Compare plans */}
        <View style={styles.compareCard}>
          <Text style={styles.compareTitle}>🆚 Compare All Plans</Text>
          <View style={styles.compareRow}>
            <Text style={styles.compareFeature}>Feature</Text>
            {plans.map(p => (
              <Text key={p.id} style={[styles.compareCol, { color: p.color }]}>{p.name}</Text>
            ))}
          </View>
          {[
            { feature: 'Free Prints', values: ['1/mo', '3/mo', '10/mo'] },
            { feature: 'Discount', values: ['10%', '20%', '30%'] },
            { feature: 'Support', values: ['Email', 'Priority', 'Dedicated'] },
            { feature: 'Design Review', values: ['✗', '1/mo', 'Unlimited'] },
          ].map((row, i) => (
            <View key={i} style={[styles.compareRow, i % 2 === 0 && styles.compareRowAlt]}>
              <Text style={styles.compareFeature}>{row.feature}</Text>
              {row.values.map((v, j) => (
                <Text key={j} style={styles.compareVal}>{v}</Text>
              ))}
            </View>
          ))}
        </View>

        <Text style={styles.cancelNote}>✓ No contracts · Cancel anytime · Instant activation</Text>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { paddingTop: 52, paddingBottom: 24, paddingHorizontal: Spacing.md, alignItems: 'center' },
  backBtn: { position: 'absolute', top: 52, right: 16, width: 36, height: 36, borderRadius: 18, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  headerTitle: { color: Colors.white, fontSize: FontSize.xxxl, fontWeight: '900', marginBottom: 4 },
  headerSubtitle: { color: 'rgba(255,255,255,0.75)', fontSize: FontSize.md, marginBottom: 16 },
  toggle: { flexDirection: 'row', backgroundColor: 'rgba(255,255,255,0.15)', borderRadius: 20, padding: 4 },
  toggleBtn: { paddingHorizontal: 20, paddingVertical: 8, borderRadius: 16, alignItems: 'center' },
  toggleBtnActive: { backgroundColor: Colors.white },
  toggleText: { color: 'rgba(255,255,255,0.7)', fontWeight: '700', fontSize: FontSize.md },
  toggleTextActive: { color: Colors.secondary },
  saveBadge: { backgroundColor: Colors.accent, borderRadius: 8, paddingHorizontal: 6, paddingVertical: 2, marginTop: 2 },
  saveBadgeText: { color: Colors.secondary, fontSize: 9, fontWeight: '800' },
  planCard: { backgroundColor: Colors.card, borderRadius: BorderRadius.lg, marginBottom: 16, overflow: 'hidden', ...Shadows.medium },
  planCardPopular: { borderWidth: 2, borderColor: Colors.primary },
  popularLabel: { backgroundColor: Colors.primary, paddingVertical: 6, alignItems: 'center' },
  popularLabelText: { color: Colors.white, fontSize: FontSize.sm, fontWeight: '800', letterSpacing: 1 },
  planHeader: { padding: 24, alignItems: 'center' },
  planEmoji: { fontSize: 40, marginBottom: 8 },
  planName: { color: Colors.white, fontSize: FontSize.xxl, fontWeight: '900', marginBottom: 12 },
  priceRow: { flexDirection: 'row', alignItems: 'flex-end', gap: 2, marginBottom: 4 },
  currency: { color: 'rgba(255,255,255,0.8)', fontSize: FontSize.xl, fontWeight: '700', marginBottom: 6 },
  price: { color: Colors.white, fontSize: 48, fontWeight: '900' },
  period: { color: 'rgba(255,255,255,0.8)', fontSize: FontSize.md, marginBottom: 8 },
  annualNote: { color: 'rgba(255,255,255,0.7)', fontSize: FontSize.sm, marginBottom: 12 },
  highlights: { flexDirection: 'row', gap: 24, marginTop: 8 },
  highlight: { alignItems: 'center' },
  highlightValue: { color: Colors.white, fontSize: FontSize.xxl, fontWeight: '900' },
  highlightLabel: { color: 'rgba(255,255,255,0.7)', fontSize: FontSize.xs },
  highlightDivider: { width: 1, backgroundColor: 'rgba(255,255,255,0.3)' },
  featuresContainer: { padding: 20 },
  featureRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 6 },
  featureText: { fontSize: FontSize.md, color: Colors.textPrimary, flex: 1 },
  subscribeBtn: { marginHorizontal: 20, marginBottom: 20, borderRadius: BorderRadius.lg, overflow: 'hidden' },
  subscribeBtnGradient: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 16, gap: 8, borderRadius: BorderRadius.lg },
  subscribeBtnText: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '800' },
  compareCard: { backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 16, ...Shadows.small, marginBottom: 16 },
  compareTitle: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.textPrimary, marginBottom: 12 },
  compareRow: { flexDirection: 'row', paddingVertical: 8 },
  compareRowAlt: { backgroundColor: Colors.background, borderRadius: 8 },
  compareFeature: { flex: 2, fontSize: FontSize.sm, color: Colors.textSecondary, fontWeight: '600' },
  compareCol: { flex: 1, fontSize: FontSize.sm, fontWeight: '800', textAlign: 'center' },
  compareVal: { flex: 1, fontSize: FontSize.sm, color: Colors.textPrimary, textAlign: 'center', fontWeight: '600' },
  cancelNote: { textAlign: 'center', color: Colors.textSecondary, fontSize: FontSize.sm, fontWeight: '600' },
});

export default SubscriptionScreen;
