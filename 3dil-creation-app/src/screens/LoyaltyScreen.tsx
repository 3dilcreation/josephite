import React from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Share,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';
import { useAuth } from '../context/AuthContext';
import { useNavigation } from '@react-navigation/native';
import { LoyaltyTier } from '../types';

const tiers: { name: LoyaltyTier; min: number; max: number; color: string; gradient: [string, string]; perks: string[] }[] = [
  { name: 'Bronze', min: 0, max: 499, color: '#CD7F32', gradient: ['#CD7F32', '#A0522D'], perks: ['5% discount', 'Birthday bonus points', 'Early access to offers'] },
  { name: 'Silver', min: 500, max: 1999, color: '#9E9E9E', gradient: ['#C0C0C0', '#9E9E9E'], perks: ['10% discount', 'Free design consultation', 'Priority delivery', 'Exclusive products'] },
  { name: 'Gold', min: 2000, max: 4999, color: '#FFD700', gradient: ['#FFD700', '#FFA500'], perks: ['20% discount', 'Free painting on orders ₹1000+', 'Dedicated support', 'Monthly free print'] },
  { name: 'Platinum', min: 5000, max: Infinity, color: '#E5E4E2', gradient: ['#E5E4E2', '#B0C4DE'], perks: ['30% discount', 'Free delivery always', 'VIP events', '2 monthly free prints', 'Personal account manager'] },
];

const redeemOptions = [
  { points: 100, reward: '₹50 off next order', icon: '🎫' },
  { points: 200, reward: '₹100 off next order', icon: '💰' },
  { points: 150, reward: 'Free delivery', icon: '🚚' },
  { points: 300, reward: 'Priority printing', icon: '⚡' },
  { points: 500, reward: '₹300 off custom order', icon: '🎁' },
];

const transactions = [
  { date: 'Jan 10', desc: 'Custom Medal Order #ORD001', points: +150, type: 'earn' },
  { date: 'Jan 08', desc: 'Architectural Model #ORD002', points: +400, type: 'earn' },
  { date: 'Jan 05', desc: 'Redeemed — ₹50 off', points: -100, type: 'redeem' },
  { date: 'Jan 02', desc: 'Ganesh Idol #ORD003', points: +90, type: 'earn' },
  { date: 'Dec 25', desc: 'Referral Bonus', points: +200, type: 'bonus' },
];

const LoyaltyScreen: React.FC = () => {
  const { user, loyaltyPoints, loyaltyTier, redeemPoints } = useAuth();
  const navigation = useNavigation();
  const currentTier = tiers.find(t => t.name === loyaltyTier) || tiers[0];
  const nextTier = tiers[tiers.indexOf(currentTier) + 1];
  const progressToNext = nextTier
    ? ((loyaltyPoints - currentTier.min) / (nextTier.min - currentTier.min)) * 100
    : 100;

  const handleShare = async () => {
    await Share.share({
      message: `Join 3DIL Creation and get amazing 3D printed products! Use my referral code ${user?.referralCode || '3DIL-XXXX'} and get ₹100 off your first order! 🎉`,
    });
  };

  return (
    <View style={styles.container}>
      <LinearGradient colors={currentTier.gradient} style={styles.header}>
        <Text style={styles.tierEmoji}>
          {loyaltyTier === 'Bronze' ? '🥉' : loyaltyTier === 'Silver' ? '🥈' : loyaltyTier === 'Gold' ? '🥇' : '💎'}
        </Text>
        <Text style={styles.tierName}>{loyaltyTier} Member</Text>
        <Text style={styles.points}>{loyaltyPoints.toLocaleString()}</Text>
        <Text style={styles.pointsLabel}>Loyalty Points</Text>
        {nextTier && (
          <View style={styles.progressSection}>
            <Text style={styles.progressText}>
              {nextTier.min - loyaltyPoints} pts to {nextTier.name}
            </Text>
            <View style={styles.progressBar}>
              <View style={[styles.progressFill, { width: `${Math.min(100, progressToNext)}%` }]} />
            </View>
          </View>
        )}
      </LinearGradient>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 32 }}>
        {/* Earn Info */}
        <View style={styles.earnCard}>
          <Text style={styles.sectionTitle}>💡 How to Earn Points</Text>
          {[
            { icon: '🛍️', text: '1 point per ₹10 spent on orders' },
            { icon: '👥', text: '200 points per successful referral' },
            { icon: '⭐', text: '50 points for leaving a review' },
            { icon: '🎂', text: '100 bonus points on your birthday' },
          ].map((item, i) => (
            <View key={i} style={styles.earnRow}>
              <Text style={styles.earnIcon}>{item.icon}</Text>
              <Text style={styles.earnText}>{item.text}</Text>
            </View>
          ))}
        </View>

        {/* Redeem */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🎁 Redeem Points</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            {redeemOptions.map((opt, i) => (
              <TouchableOpacity
                key={i}
                style={[styles.redeemCard, loyaltyPoints < opt.points && styles.redeemCardDisabled]}
                onPress={() => loyaltyPoints >= opt.points && redeemPoints(opt.points)}
              >
                <Text style={styles.redeemIcon}>{opt.icon}</Text>
                <Text style={styles.redeemPoints}>{opt.points} pts</Text>
                <Text style={styles.redeemReward}>{opt.reward}</Text>
                <View style={[styles.redeemBtn, loyaltyPoints < opt.points && styles.redeemBtnDisabled]}>
                  <Text style={styles.redeemBtnText}>{loyaltyPoints >= opt.points ? 'Redeem' : 'Need more'}</Text>
                </View>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        {/* Referral */}
        <View style={styles.referralCard}>
          <LinearGradient colors={['#1A1A2E', '#16213E']} style={styles.referralGradient}>
            <Text style={styles.referralTitle}>👥 Refer & Earn</Text>
            <Text style={styles.referralDesc}>
              Invite friends to 3DIL Creation and earn 200 points for each successful referral!
            </Text>
            <View style={styles.codeBox}>
              <Text style={styles.codeLabel}>Your Referral Code</Text>
              <Text style={styles.code}>{user?.referralCode || '3DIL-XXXX'}</Text>
            </View>
            <TouchableOpacity style={styles.shareBtn} onPress={handleShare}>
              <Ionicons name="share-social" size={18} color={Colors.secondary} />
              <Text style={styles.shareBtnText}>Share & Earn ₹200</Text>
            </TouchableOpacity>
          </LinearGradient>
        </View>

        {/* Tiers */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🏆 Membership Tiers</Text>
          {tiers.map(tier => (
            <View key={tier.name} style={[styles.tierCard, tier.name === loyaltyTier && styles.tierCardActive]}>
              <LinearGradient colors={tier.gradient} style={styles.tierBadge}>
                <Text style={styles.tierBadgeText}>{tier.name}</Text>
              </LinearGradient>
              <View style={{ flex: 1 }}>
                <Text style={styles.tierRange}>{tier.min.toLocaleString()} – {tier.max === Infinity ? '∞' : tier.max.toLocaleString()} pts</Text>
                {tier.perks.slice(0, 2).map((p, i) => (
                  <Text key={i} style={styles.tierPerk}>✓ {p}</Text>
                ))}
              </View>
              {tier.name === loyaltyTier && (
                <View style={styles.currentBadge}>
                  <Text style={styles.currentBadgeText}>Current</Text>
                </View>
              )}
            </View>
          ))}
        </View>

        {/* Transaction History */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📊 Points History</Text>
          {transactions.map((t, i) => (
            <View key={i} style={styles.transactionRow}>
              <View>
                <Text style={styles.transactionDesc}>{t.desc}</Text>
                <Text style={styles.transactionDate}>{t.date}</Text>
              </View>
              <Text style={[styles.transactionPoints, { color: t.type === 'earn' || t.type === 'bonus' ? Colors.success : Colors.error }]}>
                {t.points > 0 ? '+' : ''}{t.points} pts
              </Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { paddingTop: 52, paddingBottom: 28, paddingHorizontal: Spacing.md, alignItems: 'center' },
  tierEmoji: { fontSize: 48, marginBottom: 8 },
  tierName: { color: 'rgba(255,255,255,0.85)', fontSize: FontSize.lg, fontWeight: '700', marginBottom: 4 },
  points: { color: Colors.white, fontSize: 52, fontWeight: '900' },
  pointsLabel: { color: 'rgba(255,255,255,0.75)', fontSize: FontSize.md, marginBottom: 16 },
  progressSection: { width: '100%' },
  progressText: { color: 'rgba(255,255,255,0.8)', fontSize: FontSize.sm, marginBottom: 6, textAlign: 'center' },
  progressBar: { height: 8, backgroundColor: 'rgba(255,255,255,0.25)', borderRadius: 4, overflow: 'hidden' },
  progressFill: { height: 8, backgroundColor: Colors.white, borderRadius: 4 },
  earnCard: { margin: 12, backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 16, ...Shadows.small },
  sectionTitle: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.textPrimary, marginBottom: 12 },
  earnRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 8 },
  earnIcon: { fontSize: 22 },
  earnText: { fontSize: FontSize.md, color: Colors.textSecondary },
  section: { margin: 12, backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 16, ...Shadows.small },
  redeemCard: { width: 140, backgroundColor: Colors.background, borderRadius: BorderRadius.md, padding: 14, marginRight: 10, alignItems: 'center', borderWidth: 1.5, borderColor: Colors.border },
  redeemCardDisabled: { opacity: 0.5 },
  redeemIcon: { fontSize: 32, marginBottom: 8 },
  redeemPoints: { fontSize: FontSize.lg, fontWeight: '900', color: Colors.primary, marginBottom: 4 },
  redeemReward: { fontSize: FontSize.xs, color: Colors.textSecondary, textAlign: 'center', marginBottom: 10, lineHeight: 16 },
  redeemBtn: { backgroundColor: Colors.primary, paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20 },
  redeemBtnDisabled: { backgroundColor: Colors.border },
  redeemBtnText: { color: Colors.white, fontSize: FontSize.xs, fontWeight: '800' },
  referralCard: { margin: 12, borderRadius: BorderRadius.lg, overflow: 'hidden', ...Shadows.medium },
  referralGradient: { borderRadius: BorderRadius.lg },
  referralTitle: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '900', padding: 20, paddingBottom: 8 },
  referralDesc: { color: 'rgba(255,255,255,0.75)', fontSize: FontSize.md, paddingHorizontal: 20, lineHeight: 22, marginBottom: 16 },
  codeBox: { marginHorizontal: 20, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: BorderRadius.md, padding: 14, alignItems: 'center', marginBottom: 16 },
  codeLabel: { color: 'rgba(255,255,255,0.6)', fontSize: FontSize.xs, marginBottom: 4 },
  code: { color: Colors.accent, fontSize: FontSize.xxl, fontWeight: '900', letterSpacing: 2 },
  shareBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: Colors.accent, margin: 20, marginTop: 0, borderRadius: BorderRadius.lg, paddingVertical: 16 },
  shareBtnText: { color: Colors.secondary, fontWeight: '900', fontSize: FontSize.lg },
  tierCard: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 12, borderRadius: BorderRadius.md, marginBottom: 8, borderWidth: 1.5, borderColor: Colors.border },
  tierCardActive: { borderColor: Colors.primary, backgroundColor: Colors.primary + '08' },
  tierBadge: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12 },
  tierBadgeText: { color: Colors.white, fontWeight: '800', fontSize: FontSize.sm },
  tierRange: { fontSize: FontSize.xs, color: Colors.textSecondary, marginBottom: 4 },
  tierPerk: { fontSize: FontSize.xs, color: Colors.textSecondary },
  currentBadge: { backgroundColor: Colors.primary, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8 },
  currentBadgeText: { color: Colors.white, fontSize: 9, fontWeight: '800' },
  transactionRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: Colors.border },
  transactionDesc: { fontSize: FontSize.sm, fontWeight: '600', color: Colors.textPrimary, marginBottom: 2 },
  transactionDate: { fontSize: FontSize.xs, color: Colors.textLight },
  transactionPoints: { fontSize: FontSize.md, fontWeight: '800' },
});

export default LoyaltyScreen;
