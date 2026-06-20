import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Colors, Spacing, Radius } from '../theme/colors';
import { SUBSCRIPTIONS } from '../data/mockData';

const { width } = Dimensions.get('window');

export default function SubscriptionScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const [selected, setSelected] = useState('sub2');
  const [billing, setBilling] = useState('monthly');
  const [subscribed, setSubscribed] = useState(false);

  const selectedPlan = SUBSCRIPTIONS.find(s => s.id === selected);

  if (subscribed) {
    return (
      <View style={[styles.container, { paddingTop: insets.top, alignItems: 'center', justifyContent: 'center', padding: Spacing.xl }]}>
        <LinearGradient colors={selectedPlan.color} style={styles.successIcon}>
          <MaterialCommunityIcons name="crown" size={48} color="#FFF" />
        </LinearGradient>
        <Text style={styles.successTitle}>Welcome to {selectedPlan.name}!</Text>
        <Text style={styles.successSub}>
          Your subscription is now active. Start booking your{'\n'}
          {selectedPlan.cuts === 99 ? 'unlimited' : selectedPlan.cuts} monthly cuts now!
        </Text>
        <TouchableOpacity style={styles.successBtn} onPress={() => { setSubscribed(false); navigation.goBack(); }}>
          <LinearGradient colors={selectedPlan.color} style={styles.successBtnGrad} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}>
            <Text style={styles.successBtnText}>Start Booking</Text>
          </LinearGradient>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <MaterialCommunityIcons name="arrow-left" size={22} color={Colors.textPrimary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Membership Plans</Text>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        {/* Hero */}
        <View style={styles.hero}>
          <Text style={styles.heroTitle}>Get More,{'\n'}Pay Less.</Text>
          <Text style={styles.heroSub}>
            Join thousands who save money and always look fresh with a FadeBlades membership.
          </Text>
        </View>

        {/* Billing Toggle */}
        <View style={styles.billingToggle}>
          {['monthly', 'annual'].map((b) => (
            <TouchableOpacity
              key={b}
              style={[styles.billingOption, billing === b && styles.billingOptionActive]}
              onPress={() => setBilling(b)}
              activeOpacity={0.8}
            >
              {billing === b && (
                <LinearGradient colors={Colors.gradientGold} style={StyleSheet.absoluteFill} borderRadius={Radius.full} />
              )}
              <Text style={[styles.billingText, billing === b && { color: '#0A0A0F' }]}>
                {b.charAt(0).toUpperCase() + b.slice(1)}
              </Text>
              {b === 'annual' && (
                <View style={styles.saveBadge}>
                  <Text style={styles.saveBadgeText}>Save 20%</Text>
                </View>
              )}
            </TouchableOpacity>
          ))}
        </View>

        {/* Plans */}
        {SUBSCRIPTIONS.map((plan) => (
          <TouchableOpacity
            key={plan.id}
            style={[styles.planCard, selected === plan.id && styles.planCardSelected]}
            onPress={() => setSelected(plan.id)}
            activeOpacity={0.9}
          >
            {/* Card Background */}
            <LinearGradient
              colors={selected === plan.id ? plan.color.map(c => c + '20') : Colors.gradientCard}
              style={StyleSheet.absoluteFill}
              borderRadius={Radius.xl}
            />

            {selected === plan.id && (
              <View style={[styles.planBorder, { borderColor: plan.color[0] }]} />
            )}

            {plan.popular && (
              <LinearGradient colors={plan.color} style={styles.popularBadge} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}>
                <MaterialCommunityIcons name="crown" size={12} color="#FFF" />
                <Text style={styles.popularBadgeText}>Most Popular</Text>
              </LinearGradient>
            )}

            <View style={styles.planHeader}>
              <View>
                <Text style={styles.planName}>{plan.name}</Text>
                <Text style={styles.planCuts}>
                  {plan.cuts === 99 ? 'Unlimited' : `${plan.cuts} cuts`} per month
                </Text>
              </View>
              <View style={styles.planPricing}>
                <Text style={styles.planPrice}>
                  ${billing === 'annual' ? Math.round(plan.price * 0.8) : plan.price}
                </Text>
                <Text style={styles.planPeriod}>/mo</Text>
              </View>
            </View>

            <View style={styles.planDivider} />

            <View style={styles.planFeatures}>
              {plan.features.map((feature) => (
                <View key={feature} style={styles.featureRow}>
                  <View style={[styles.featureCheck, { backgroundColor: plan.color[0] + '30' }]}>
                    <MaterialCommunityIcons name="check" size={12} color={plan.color[0]} />
                  </View>
                  <Text style={styles.featureText}>{feature}</Text>
                </View>
              ))}
            </View>

            {selected === plan.id && (
              <View style={styles.selectedIndicator}>
                <LinearGradient colors={plan.color} style={styles.selectedIndicatorGrad}>
                  <MaterialCommunityIcons name="check" size={14} color="#FFF" />
                  <Text style={styles.selectedText}>Selected</Text>
                </LinearGradient>
              </View>
            )}
          </TouchableOpacity>
        ))}

        {/* Value Props */}
        <View style={styles.valueSection}>
          <Text style={styles.valueSectionTitle}>Why Subscribe?</Text>
          {[
            { icon: '💰', title: 'Save Up to $90/mo', desc: 'vs. paying per visit every time' },
            { icon: '⚡', title: 'Priority Booking', desc: 'First access to all time slots' },
            { icon: '🎁', title: 'Exclusive Perks', desc: 'Free add-ons and monthly treatments' },
            { icon: '🔒', title: 'Cancel Anytime', desc: 'No long-term commitments required' },
          ].map((v) => (
            <View key={v.title} style={styles.valueProp}>
              <Text style={styles.valuePropEmoji}>{v.icon}</Text>
              <View>
                <Text style={styles.valuePropTitle}>{v.title}</Text>
                <Text style={styles.valuePropDesc}>{v.desc}</Text>
              </View>
            </View>
          ))}
        </View>
      </ScrollView>

      {/* Subscribe CTA */}
      <View style={[styles.footer, { paddingBottom: insets.bottom + Spacing.md }]}>
        <Text style={styles.footerNote}>
          Cancel anytime · No hidden fees · Billed {billing}
        </Text>
        <TouchableOpacity
          style={styles.subscribeBtn}
          onPress={() => setSubscribed(true)}
          activeOpacity={0.85}
        >
          <LinearGradient
            colors={selectedPlan?.color || Colors.gradientGold}
            style={styles.subscribeBtnGrad}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <MaterialCommunityIcons name="crown" size={20} color="#FFF" />
            <Text style={styles.subscribeBtnText}>
              Start {selectedPlan?.name} · ${billing === 'annual'
                ? Math.round((selectedPlan?.price || 89) * 0.8)
                : selectedPlan?.price}/mo
            </Text>
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.md,
    gap: 12,
  },
  backBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.bgGlass,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: { color: Colors.textPrimary, fontSize: 20, fontWeight: '800' },
  scrollContent: { paddingBottom: 140 },
  hero: { paddingHorizontal: Spacing.lg, paddingVertical: Spacing.lg },
  heroTitle: { color: Colors.textPrimary, fontSize: 36, fontWeight: '900', lineHeight: 44, marginBottom: 10 },
  heroSub: { color: Colors.textSecondary, fontSize: 15, lineHeight: 24 },
  billingToggle: {
    flexDirection: 'row',
    marginHorizontal: Spacing.md,
    backgroundColor: Colors.bgCardAlt,
    borderRadius: Radius.full,
    padding: 4,
    marginBottom: Spacing.lg,
  },
  billingOption: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    borderRadius: Radius.full,
    overflow: 'hidden',
    gap: 6,
  },
  billingOptionActive: {},
  billingText: { color: Colors.textSecondary, fontSize: 14, fontWeight: '700' },
  saveBadge: {
    backgroundColor: Colors.success + '30',
    borderRadius: Radius.full,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  saveBadgeText: { color: Colors.success, fontSize: 9, fontWeight: '800' },
  planCard: {
    marginHorizontal: Spacing.md,
    borderRadius: Radius.xl,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
    position: 'relative',
  },
  planCardSelected: {},
  planBorder: {
    position: 'absolute',
    top: 0, left: 0, right: 0, bottom: 0,
    borderRadius: Radius.xl,
    borderWidth: 2,
  },
  popularBadge: {
    position: 'absolute',
    top: 16,
    right: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: Radius.full,
  },
  popularBadgeText: { color: '#FFF', fontSize: 11, fontWeight: '800' },
  planHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: Spacing.md },
  planName: { color: Colors.textPrimary, fontSize: 24, fontWeight: '900', marginBottom: 4 },
  planCuts: { color: Colors.textSecondary, fontSize: 13 },
  planPricing: { flexDirection: 'row', alignItems: 'flex-end', gap: 2 },
  planPrice: { color: Colors.textPrimary, fontSize: 36, fontWeight: '900' },
  planPeriod: { color: Colors.textSecondary, fontSize: 14, marginBottom: 8 },
  planDivider: { height: 1, backgroundColor: Colors.borderSubtle, marginBottom: Spacing.md },
  planFeatures: { gap: 10 },
  featureRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  featureCheck: {
    width: 22,
    height: 22,
    borderRadius: 11,
    alignItems: 'center',
    justifyContent: 'center',
  },
  featureText: { color: Colors.textSecondary, fontSize: 14 },
  selectedIndicator: { marginTop: Spacing.md, borderRadius: Radius.full, overflow: 'hidden' },
  selectedIndicatorGrad: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 10,
  },
  selectedText: { color: '#FFF', fontSize: 13, fontWeight: '700' },
  valueSection: { marginHorizontal: Spacing.md, marginTop: Spacing.lg },
  valueSectionTitle: { color: Colors.textSecondary, fontSize: 12, fontWeight: '700', letterSpacing: 1, textTransform: 'uppercase', marginBottom: Spacing.md },
  valueProp: { flexDirection: 'row', alignItems: 'center', gap: 14, marginBottom: Spacing.md },
  valuePropEmoji: { fontSize: 28, width: 36, textAlign: 'center' },
  valuePropTitle: { color: Colors.textPrimary, fontSize: 15, fontWeight: '700', marginBottom: 2 },
  valuePropDesc: { color: Colors.textSecondary, fontSize: 13 },
  footer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: Colors.bg,
    borderTopWidth: 1,
    borderTopColor: Colors.borderSubtle,
    paddingHorizontal: Spacing.md,
    paddingTop: Spacing.md,
    gap: 8,
  },
  footerNote: { color: Colors.textMuted, fontSize: 12, textAlign: 'center' },
  subscribeBtn: { borderRadius: Radius.full, overflow: 'hidden' },
  subscribeBtnGrad: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 18,
  },
  subscribeBtnText: { color: '#FFF', fontSize: 16, fontWeight: '800' },
  successIcon: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.xl,
  },
  successTitle: { color: Colors.textPrimary, fontSize: 30, fontWeight: '900', marginBottom: 12 },
  successSub: { color: Colors.textSecondary, fontSize: 15, textAlign: 'center', lineHeight: 24, marginBottom: Spacing.xl },
  successBtn: { width: '100%', borderRadius: Radius.full, overflow: 'hidden' },
  successBtnGrad: { paddingVertical: 18, alignItems: 'center' },
  successBtnText: { color: '#FFF', fontSize: 17, fontWeight: '800' },
});
