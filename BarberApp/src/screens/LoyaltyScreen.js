import React, { useState, useRef, useEffect } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  Animated, Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, Spacing, Radius } from '../theme/colors';
import { BADGES } from '../data/mockData';

const { width } = Dimensions.get('window');

const TIERS = [
  { name: 'Bronze', icon: '🥉', minXP: 0, color: '#CD7F32', nextAt: 500 },
  { name: 'Silver', icon: '🥈', minXP: 500, color: '#C0C0C0', nextAt: 1250 },
  { name: 'Gold', icon: '🥇', minXP: 1250, color: '#FFD700', nextAt: 2500 },
  { name: 'Platinum', icon: '💎', minXP: 2500, color: '#E5E4E2', nextAt: 5000 },
  { name: 'Diamond', icon: '👑', minXP: 5000, color: '#B9F2FF', nextAt: null },
];

const REWARDS = [
  { id: 'r1', name: 'Free Line-Up', cost: 200, icon: '✂️', available: true },
  { id: 'r2', name: '20% Off Next Cut', cost: 350, icon: '💰', available: true },
  { id: 'r3', name: 'Free Beard Trim', cost: 450, icon: '🧔', available: false },
  { id: 'r4', name: 'Free Premium Cut', cost: 800, icon: '👑', available: false },
  { id: 'r5', name: 'Hot Towel Shave', cost: 300, icon: '🔥', available: true },
  { id: 'r6', name: 'Scalp Treatment', cost: 600, icon: '💆', available: false },
];

const USER_XP = 1250;
const USER_TIER_INDEX = 2; // Gold
const STREAK = 4;

export default function LoyaltyScreen() {
  const insets = useSafeAreaInsets();
  const xpAnim = useRef(new Animated.Value(0)).current;
  const tierAnim = useRef(new Animated.Value(0)).current;
  const [activeTab, setActiveTab] = useState('rewards');

  useEffect(() => {
    Animated.parallel([
      Animated.timing(xpAnim, { toValue: 1, duration: 1200, useNativeDriver: false }),
      Animated.spring(tierAnim, { toValue: 1, tension: 60, friction: 8, useNativeDriver: true }),
    ]).start();
  }, []);

  const currentTier = TIERS[USER_TIER_INDEX];
  const nextTier = TIERS[USER_TIER_INDEX + 1];
  const xpToNext = nextTier ? nextTier.minXP - USER_XP : 0;
  const xpProgress = nextTier ? (USER_XP - currentTier.minXP) / (nextTier.minXP - currentTier.minXP) : 1;

  const xpWidth = xpAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', `${Math.round(xpProgress * 100)}%`],
  });

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Hero Card */}
        <LinearGradient colors={['#1A1200', '#0A0A0F']} style={styles.hero}>
          <View style={styles.heroHeader}>
            <View>
              <Text style={styles.heroGreet}>Your Loyalty Status</Text>
              <View style={styles.tierRow}>
                <Text style={styles.tierEmoji}>{currentTier.icon}</Text>
                <Text style={[styles.tierName, { color: currentTier.color }]}>{currentTier.name} Member</Text>
              </View>
            </View>
            <Animated.View style={[styles.streakBubble, { transform: [{ scale: tierAnim }] }]}>
              <Text style={styles.streakNum}>{STREAK}</Text>
              <Text style={styles.streakLabel}>🔥 Streak</Text>
            </Animated.View>
          </View>

          {/* XP Display */}
          <View style={styles.xpCard}>
            <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.md} />
            <View style={styles.xpHeader}>
              <View>
                <Text style={styles.xpValue}>{USER_XP.toLocaleString()} XP</Text>
                <Text style={styles.xpSub}>
                  {nextTier ? `${xpToNext} XP to ${nextTier.name}` : 'Max Tier Reached!'}
                </Text>
              </View>
              <View style={styles.xpEarned}>
                <MaterialCommunityIcons name="trending-up" size={16} color={Colors.success} />
                <Text style={styles.xpEarnedText}>+250 this month</Text>
              </View>
            </View>
            <View style={styles.xpBarBg}>
              <Animated.View style={[styles.xpBarFill, { width: xpWidth, backgroundColor: currentTier.color }]} />
            </View>
            <View style={styles.xpTierLabels}>
              <Text style={[styles.xpTierLabel, { color: currentTier.color }]}>{currentTier.name}</Text>
              {nextTier && <Text style={styles.xpTierLabelNext}>{nextTier.name}</Text>}
            </View>
          </View>

          {/* Tier Progress */}
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.tiersScroll}>
            {TIERS.map((tier, i) => (
              <View
                key={tier.name}
                style={[styles.tierChip, i === USER_TIER_INDEX && styles.tierChipActive]}
              >
                {i === USER_TIER_INDEX && (
                  <LinearGradient
                    colors={[tier.color + '30', tier.color + '10']}
                    style={StyleSheet.absoluteFill}
                    borderRadius={Radius.md}
                  />
                )}
                <Text style={styles.tierChipEmoji}>{tier.icon}</Text>
                <Text style={[styles.tierChipName, i <= USER_TIER_INDEX && { color: tier.color }]}>
                  {tier.name}
                </Text>
                {i <= USER_TIER_INDEX && (
                  <MaterialCommunityIcons name="check-circle" size={14} color={tier.color} />
                )}
              </View>
            ))}
          </ScrollView>
        </LinearGradient>

        {/* Stats Row */}
        <View style={styles.statsRow}>
          {[
            { label: 'Total Visits', value: '14', icon: 'scissors-cutting' },
            { label: 'Money Saved', value: '$48', icon: 'cash' },
            { label: 'Free Cuts', value: '2', icon: 'gift' },
          ].map((stat) => (
            <View key={stat.label} style={styles.statCard}>
              <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.md} />
              <MaterialCommunityIcons name={stat.icon} size={22} color={Colors.primary} />
              <Text style={styles.statValue}>{stat.value}</Text>
              <Text style={styles.statLabel}>{stat.label}</Text>
            </View>
          ))}
        </View>

        {/* Tabs */}
        <View style={styles.tabs}>
          {['rewards', 'badges', 'history'].map((tab) => (
            <TouchableOpacity
              key={tab}
              style={[styles.tab, activeTab === tab && styles.tabActive]}
              onPress={() => setActiveTab(tab)}
            >
              {activeTab === tab && (
                <LinearGradient colors={Colors.gradientGold} style={StyleSheet.absoluteFill} borderRadius={Radius.full} />
              )}
              <Text style={[styles.tabText, activeTab === tab && { color: '#0A0A0F' }]}>
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Rewards Tab */}
        {activeTab === 'rewards' && (
          <View style={styles.section}>
            <View style={styles.xpBalance}>
              <MaterialCommunityIcons name="star-circle" size={20} color={Colors.primary} />
              <Text style={styles.xpBalanceText}>You have {USER_XP} XP to spend</Text>
            </View>
            <View style={styles.rewardsGrid}>
              {REWARDS.map((reward) => (
                <TouchableOpacity
                  key={reward.id}
                  style={[styles.rewardCard, !reward.available && styles.rewardCardLocked]}
                  activeOpacity={0.85}
                >
                  <LinearGradient
                    colors={reward.available ? Colors.gradientCard : ['#111', '#0A0A0F']}
                    style={StyleSheet.absoluteFill}
                    borderRadius={Radius.md}
                  />
                  <Text style={styles.rewardEmoji}>{reward.icon}</Text>
                  <Text style={styles.rewardName}>{reward.name}</Text>
                  <View style={styles.rewardCost}>
                    <MaterialCommunityIcons name="star" size={12} color={reward.available ? Colors.primary : Colors.textMuted} />
                    <Text style={[styles.rewardCostText, !reward.available && { color: Colors.textMuted }]}>
                      {reward.cost} XP
                    </Text>
                  </View>
                  {!reward.available && (
                    <MaterialCommunityIcons name="lock" size={16} color={Colors.textMuted} style={styles.lockIcon} />
                  )}
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}

        {/* Badges Tab */}
        {activeTab === 'badges' && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>
              {BADGES.filter(b => b.earned).length} / {BADGES.length} Earned
            </Text>
            <View style={styles.badgesGrid}>
              {BADGES.map((badge) => (
                <View
                  key={badge.id}
                  style={[styles.badgeCard, !badge.earned && styles.badgeCardLocked]}
                >
                  <LinearGradient
                    colors={badge.earned ? ['#1A1200', '#0F0D00'] : ['#111', '#0A0A0F']}
                    style={StyleSheet.absoluteFill}
                    borderRadius={Radius.md}
                  />
                  <Text style={[styles.badgeEmoji, !badge.earned && { opacity: 0.3 }]}>
                    {badge.icon}
                  </Text>
                  <Text style={[styles.badgeName, !badge.earned && { color: Colors.textMuted }]}>
                    {badge.name}
                  </Text>
                  <Text style={styles.badgeDesc} numberOfLines={2}>{badge.description}</Text>
                  <View style={styles.badgeXP}>
                    <MaterialCommunityIcons name="star" size={10} color={badge.earned ? Colors.primary : Colors.textMuted} />
                    <Text style={[styles.badgeXPText, !badge.earned && { color: Colors.textMuted }]}>
                      +{badge.xp} XP
                    </Text>
                  </View>
                  {badge.earned && (
                    <View style={styles.earnedCheck}>
                      <MaterialCommunityIcons name="check" size={12} color="#0A0A0F" />
                    </View>
                  )}
                </View>
              ))}
            </View>
          </View>
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
          <View style={styles.section}>
            {[
              { date: 'Jun 15', action: 'Booked Appointment', xp: '+75', icon: 'calendar-check', color: Colors.primary },
              { date: 'Jun 15', action: 'Visit Completed', xp: '+150', icon: 'check-circle', color: Colors.success },
              { date: 'Jun 8', action: '4-Week Streak Bonus', xp: '+200', icon: 'fire', color: '#E05C5C' },
              { date: 'Jun 1', action: 'Referred a Friend', xp: '+100', icon: 'account-plus', color: '#5C9EE0' },
              { date: 'May 28', action: 'Visit Completed', xp: '+150', icon: 'check-circle', color: Colors.success },
              { date: 'May 20', action: 'Badge Earned: Sharp Dresser', xp: '+300', icon: 'medal', color: Colors.primary },
            ].map((h, i) => (
              <View key={i} style={styles.historyItem}>
                <View style={[styles.historyIcon, { backgroundColor: h.color + '20' }]}>
                  <MaterialCommunityIcons name={h.icon} size={20} color={h.color} />
                </View>
                <View style={styles.historyInfo}>
                  <Text style={styles.historyAction}>{h.action}</Text>
                  <Text style={styles.historyDate}>{h.date}</Text>
                </View>
                <Text style={styles.historyXP}>{h.xp}</Text>
              </View>
            ))}
          </View>
        )}

        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  hero: { padding: Spacing.lg, paddingTop: Spacing.md, marginBottom: Spacing.md },
  heroHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: Spacing.lg },
  heroGreet: { color: Colors.textSecondary, fontSize: 13, marginBottom: 6 },
  tierRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  tierEmoji: { fontSize: 28 },
  tierName: { fontSize: 24, fontWeight: '800' },
  streakBubble: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: '#E05C5C20',
    borderWidth: 2,
    borderColor: '#E05C5C40',
    alignItems: 'center',
    justifyContent: 'center',
  },
  streakNum: { color: Colors.textPrimary, fontSize: 24, fontWeight: '900' },
  streakLabel: { fontSize: 11, color: Colors.textSecondary },
  xpCard: {
    borderRadius: Radius.md,
    padding: Spacing.md,
    marginBottom: Spacing.lg,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
  },
  xpHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: Spacing.md },
  xpValue: { color: Colors.textPrimary, fontSize: 28, fontWeight: '900' },
  xpSub: { color: Colors.textSecondary, fontSize: 12, marginTop: 2 },
  xpEarned: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  xpEarnedText: { color: Colors.success, fontSize: 12, fontWeight: '700' },
  xpBarBg: { height: 8, backgroundColor: Colors.bgGlass, borderRadius: 4, overflow: 'hidden', marginBottom: 8 },
  xpBarFill: { height: '100%', borderRadius: 4 },
  xpTierLabels: { flexDirection: 'row', justifyContent: 'space-between' },
  xpTierLabel: { fontSize: 11, fontWeight: '700' },
  xpTierLabelNext: { color: Colors.textMuted, fontSize: 11 },
  tiersScroll: { marginBottom: 4 },
  tierChip: {
    alignItems: 'center',
    padding: 10,
    borderRadius: Radius.md,
    marginRight: 8,
    minWidth: 72,
    gap: 4,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
  },
  tierChipActive: { borderColor: Colors.primary },
  tierChipEmoji: { fontSize: 22 },
  tierChipName: { color: Colors.textMuted, fontSize: 11, fontWeight: '700' },
  statsRow: {
    flexDirection: 'row',
    paddingHorizontal: Spacing.md,
    gap: Spacing.sm,
    marginBottom: Spacing.md,
  },
  statCard: {
    flex: 1,
    borderRadius: Radius.md,
    padding: 12,
    alignItems: 'center',
    gap: 4,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
  },
  statValue: { color: Colors.textPrimary, fontSize: 22, fontWeight: '900' },
  statLabel: { color: Colors.textMuted, fontSize: 10, fontWeight: '600', textAlign: 'center' },
  tabs: {
    flexDirection: 'row',
    marginHorizontal: Spacing.md,
    backgroundColor: Colors.bgCardAlt,
    borderRadius: Radius.full,
    padding: 4,
    marginBottom: Spacing.md,
    gap: 4,
  },
  tab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: Radius.full,
    overflow: 'hidden',
  },
  tabActive: {},
  tabText: { color: Colors.textSecondary, fontSize: 13, fontWeight: '700' },
  section: { paddingHorizontal: Spacing.md },
  sectionTitle: { color: Colors.textSecondary, fontSize: 13, fontWeight: '700', marginBottom: 12 },
  xpBalance: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: Spacing.md,
    backgroundColor: Colors.primary + '15',
    padding: 10,
    borderRadius: Radius.md,
  },
  xpBalanceText: { color: Colors.primary, fontSize: 13, fontWeight: '700' },
  rewardsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginBottom: 20 },
  rewardCard: {
    width: (width - Spacing.md * 2 - 10) / 2,
    borderRadius: Radius.md,
    padding: Spacing.md,
    alignItems: 'center',
    gap: 6,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
    position: 'relative',
  },
  rewardCardLocked: { opacity: 0.6 },
  rewardEmoji: { fontSize: 32, marginBottom: 4 },
  rewardName: { color: Colors.textPrimary, fontSize: 13, fontWeight: '700', textAlign: 'center' },
  rewardCost: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  rewardCostText: { color: Colors.primary, fontSize: 12, fontWeight: '700' },
  lockIcon: { position: 'absolute', top: 8, right: 8 },
  badgesGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginBottom: 20 },
  badgeCard: {
    width: (width - Spacing.md * 2 - 10) / 2,
    borderRadius: Radius.md,
    padding: Spacing.md,
    alignItems: 'center',
    gap: 6,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
    position: 'relative',
  },
  badgeCardLocked: { opacity: 0.5 },
  badgeEmoji: { fontSize: 32, marginBottom: 4 },
  badgeName: { color: Colors.textPrimary, fontSize: 13, fontWeight: '700', textAlign: 'center' },
  badgeDesc: { color: Colors.textSecondary, fontSize: 10, textAlign: 'center', lineHeight: 14 },
  badgeXP: { flexDirection: 'row', alignItems: 'center', gap: 3 },
  badgeXPText: { color: Colors.primary, fontSize: 11, fontWeight: '700' },
  earnedCheck: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: Colors.success,
    alignItems: 'center',
    justifyContent: 'center',
  },
  historyItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderSubtle,
  },
  historyIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  historyInfo: { flex: 1 },
  historyAction: { color: Colors.textPrimary, fontSize: 13, fontWeight: '600', marginBottom: 2 },
  historyDate: { color: Colors.textMuted, fontSize: 11 },
  historyXP: { color: Colors.success, fontSize: 15, fontWeight: '800' },
});
