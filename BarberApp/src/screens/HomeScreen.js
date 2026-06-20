import React, { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  Dimensions, Animated, Image,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Colors, Spacing, Radius } from '../theme/colors';
import { BARBERS, SERVICES, HAIRSTYLES } from '../data/mockData';

const { width } = Dimensions.get('window');

const QUICK_ACTIONS = [
  { icon: 'augmented-reality', label: 'AR Try-On', screen: 'ARTryOn', color: '#6C5CE7', gradient: ['#6C5CE7', '#2D1B69'] },
  { icon: 'format-list-numbered', label: 'Queue', screen: 'Queue', color: '#5C9EE0', gradient: ['#5C9EE0', '#1A4A80'] },
  { icon: 'face-recognition', label: 'Hair AI', screen: 'HairAnalysis', color: '#E05C5C', gradient: ['#E05C5C', '#A02020'] },
  { icon: 'camera-enhance', label: 'Journey', screen: 'HairJourney', color: '#4CAF82', gradient: ['#4CAF82', '#1A5A3A'] },
  { icon: 'crown', label: 'Subscribe', screen: 'Subscription', color: '#F0C849', gradient: ['#F0C849', '#A07830'] },
  { icon: 'chat', label: 'Chat', screen: 'Chat', color: '#C8A96E', gradient: ['#C8A96E', '#A07840'] },
];

function BarberCard({ barber, onPress }) {
  return (
    <TouchableOpacity style={styles.barberCard} onPress={onPress} activeOpacity={0.85}>
      <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.lg} />

      <View style={styles.barberTop}>
        <View style={styles.barberAvatarWrap}>
          <Image source={{ uri: barber.avatar }} style={styles.barberAvatar} />
          <View style={[styles.availDot, { backgroundColor: barber.available ? Colors.success : Colors.textMuted }]} />
        </View>
        <View style={styles.barberInfo}>
          <Text style={styles.barberName}>{barber.name}</Text>
          <Text style={styles.barberSpecialty}>{barber.specialty}</Text>
          <View style={styles.barberMeta}>
            <MaterialCommunityIcons name="star" size={14} color={Colors.primary} />
            <Text style={styles.barberRating}>{barber.rating}</Text>
            <Text style={styles.barberReviews}>({barber.reviews})</Text>
          </View>
        </View>
        <View style={styles.barberPrice}>
          <Text style={styles.priceLabel}>from</Text>
          <Text style={styles.priceValue}>${barber.price}</Text>
        </View>
      </View>

      <View style={styles.barberBadges}>
        {barber.badges.slice(0, 2).map((badge) => (
          <View key={badge} style={styles.badge}>
            <Text style={styles.badgeText}>{badge}</Text>
          </View>
        ))}
      </View>

      <View style={styles.barberBottom}>
        <View style={styles.barberStat}>
          <MaterialCommunityIcons name="clock-outline" size={14} color={Colors.textSecondary} />
          <Text style={styles.statText}>
            {barber.available ? `Next: ${barber.nextSlot}` : `Wait: ${barber.waitTime}min`}
          </Text>
        </View>
        <TouchableOpacity style={styles.bookNowBtn} activeOpacity={0.8}>
          <LinearGradient
            colors={Colors.gradientGold}
            style={styles.bookNowGrad}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <Text style={styles.bookNowText}>Book</Text>
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );
}

function TrendingStyle({ style }) {
  return (
    <TouchableOpacity style={styles.styleChip} activeOpacity={0.8}>
      <LinearGradient
        colors={[style.color + '25', style.color + '10']}
        style={styles.styleChipGrad}
      >
        <Text style={styles.styleEmoji}>{style.emoji}</Text>
        <Text style={styles.styleName}>{style.name}</Text>
        {style.trending && (
          <View style={[styles.trendDot, { backgroundColor: style.color }]} />
        )}
      </LinearGradient>
    </TouchableOpacity>
  );
}

export default function HomeScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const scrollY = useRef(new Animated.Value(0)).current;

  const headerOpacity = scrollY.interpolate({
    inputRange: [0, 80],
    outputRange: [0, 1],
    extrapolate: 'clamp',
  });

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Sticky Compact Header */}
      <Animated.View style={[styles.stickyHeader, { opacity: headerOpacity }]}>
        <LinearGradient colors={['#0A0A0F', '#0A0A0FEE']} style={StyleSheet.absoluteFill} />
        <Text style={styles.stickyTitle}>FadeBlades 💈</Text>
      </Animated.View>

      <Animated.ScrollView
        contentContainerStyle={styles.scrollContent}
        onScroll={Animated.event([{ nativeEvent: { contentOffset: { y: scrollY } } }], {
          useNativeDriver: true,
        })}
        scrollEventThrottle={16}
        showsVerticalScrollIndicator={false}
      >
        {/* Hero */}
        <LinearGradient
          colors={['#1A1200', '#0A0A0F']}
          style={styles.hero}
        >
          <View style={styles.heroTop}>
            <View>
              <Text style={styles.greeting}>Good morning, Jordan 👋</Text>
              <Text style={styles.heroTitle}>Time to get{'\n'}fresh again.</Text>
            </View>
            <TouchableOpacity style={styles.notifBtn}>
              <MaterialCommunityIcons name="bell-outline" size={24} color={Colors.textPrimary} />
              <View style={styles.notifDot} />
            </TouchableOpacity>
          </View>

          {/* Next Appointment Card */}
          <TouchableOpacity style={styles.apptCard} activeOpacity={0.9}>
            <LinearGradient colors={Colors.gradientGold} style={styles.apptGrad} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}>
              <View style={styles.apptLeft}>
                <MaterialCommunityIcons name="calendar-check" size={28} color="#0A0A0F" />
                <View>
                  <Text style={styles.apptLabel}>Next Appointment</Text>
                  <Text style={styles.apptTime}>Tomorrow · 10:30 AM</Text>
                  <Text style={styles.apptBarber}>with Marcus • Mid Taper Fade</Text>
                </View>
              </View>
              <MaterialCommunityIcons name="chevron-right" size={22} color="#0A0A0F80" />
            </LinearGradient>
          </TouchableOpacity>
        </LinearGradient>

        {/* Quick Actions Grid */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Quick Access</Text>
          <View style={styles.quickGrid}>
            {QUICK_ACTIONS.map((action) => (
              <TouchableOpacity
                key={action.label}
                style={styles.quickAction}
                onPress={() => navigation.navigate(action.screen)}
                activeOpacity={0.8}
              >
                <LinearGradient colors={action.gradient} style={styles.quickIconBg}>
                  <MaterialCommunityIcons name={action.icon} size={26} color="#FFF" />
                </LinearGradient>
                <Text style={styles.quickLabel}>{action.label}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Trending Styles */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Trending Styles</Text>
            <TouchableOpacity>
              <Text style={styles.seeAll}>See All</Text>
            </TouchableOpacity>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.hScroll}>
            {HAIRSTYLES.filter(s => s.trending).map((style) => (
              <TrendingStyle key={style.id} style={style} />
            ))}
          </ScrollView>
        </View>

        {/* AI Recommendation Banner */}
        <View style={styles.section}>
          <TouchableOpacity
            activeOpacity={0.9}
            onPress={() => navigation.navigate('HairAnalysis')}
          >
            <LinearGradient
              colors={['#1A0A2E', '#0A0A1A']}
              style={styles.aiBanner}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
            >
              <View style={styles.aiBannerLeft}>
                <View style={styles.aiBadge}>
                  <MaterialCommunityIcons name="star-four-points" size={12} color="#6C5CE7" />
                  <Text style={styles.aiBadgeText}>AI-Powered</Text>
                </View>
                <Text style={styles.aiBannerTitle}>Get Your{'\n'}Perfect Style</Text>
                <Text style={styles.aiBannerSub}>
                  Our AI analyzes your face shape and recommends cuts that suit you best
                </Text>
                <View style={styles.aiBannerBtn}>
                  <Text style={styles.aiBannerBtnText}>Analyze Now</Text>
                  <MaterialCommunityIcons name="arrow-right" size={16} color="#6C5CE7" />
                </View>
              </View>
              <MaterialCommunityIcons name="face-recognition" size={80} color="#6C5CE720" style={styles.aiIcon} />
            </LinearGradient>
          </TouchableOpacity>
        </View>

        {/* Barbers */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Our Barbers</Text>
            <TouchableOpacity>
              <Text style={styles.seeAll}>View All</Text>
            </TouchableOpacity>
          </View>
          {BARBERS.map((barber) => (
            <BarberCard
              key={barber.id}
              barber={barber}
              onPress={() => navigation.navigate('Book')}
            />
          ))}
        </View>

        {/* Loyalty Teaser */}
        <View style={[styles.section, { marginBottom: 24 }]}>
          <TouchableOpacity
            activeOpacity={0.9}
            onPress={() => navigation.navigate('Loyalty')}
          >
            <LinearGradient
              colors={['#1A1000', '#120D00']}
              style={styles.loyaltyTeaser}
            >
              <View style={styles.loyaltyTeaserLeft}>
                <Text style={styles.loyaltyTeaserTitle}>Your Streak 🔥</Text>
                <Text style={styles.loyaltyTeaserSub}>4 weeks strong • 1,250 XP</Text>
                <View style={styles.xpBar}>
                  <View style={[styles.xpFill, { width: '65%' }]} />
                </View>
                <Text style={styles.xpLabel}>650 XP to Gold Tier</Text>
              </View>
              <MaterialCommunityIcons name="crown" size={52} color={Colors.primary + '40'} />
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </Animated.ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  stickyHeader: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 99,
    height: 60,
    justifyContent: 'flex-end',
    paddingBottom: 12,
    alignItems: 'center',
  },
  stickyTitle: {
    color: Colors.textPrimary,
    fontSize: 18,
    fontWeight: '700',
  },
  scrollContent: { paddingBottom: 40 },
  hero: {
    padding: Spacing.lg,
    paddingTop: Spacing.xl,
    paddingBottom: Spacing.xl,
  },
  heroTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: Spacing.xl,
  },
  greeting: {
    color: Colors.textSecondary,
    fontSize: 14,
    marginBottom: 6,
  },
  heroTitle: {
    color: Colors.textPrimary,
    fontSize: 34,
    fontWeight: '800',
    lineHeight: 42,
    letterSpacing: -0.5,
  },
  notifBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: Colors.bgGlass,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    alignItems: 'center',
    justifyContent: 'center',
  },
  notifDot: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.accent,
    borderWidth: 2,
    borderColor: Colors.bgCard,
  },
  apptCard: { borderRadius: Radius.lg, overflow: 'hidden' },
  apptGrad: {
    padding: Spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  apptLeft: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  apptLabel: { fontSize: 11, color: '#0A0A0F80', fontWeight: '700', letterSpacing: 1, textTransform: 'uppercase' },
  apptTime: { fontSize: 18, fontWeight: '800', color: '#0A0A0F', marginTop: 2 },
  apptBarber: { fontSize: 12, color: '#0A0A0F60', marginTop: 2 },
  section: { paddingHorizontal: Spacing.md, marginTop: Spacing.xl },
  sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 },
  sectionTitle: { color: Colors.textPrimary, fontSize: 20, fontWeight: '700' },
  seeAll: { color: Colors.primary, fontSize: 14, fontWeight: '600' },
  quickGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  quickAction: {
    width: (width - Spacing.md * 2 - 12 * 2) / 3,
    alignItems: 'center',
    gap: 8,
  },
  quickIconBg: {
    width: 60,
    height: 60,
    borderRadius: Radius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  quickLabel: { color: Colors.textSecondary, fontSize: 12, fontWeight: '600' },
  hScroll: { marginHorizontal: -Spacing.md, paddingHorizontal: Spacing.md },
  styleChip: { marginRight: 10 },
  styleChipGrad: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: Radius.full,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
  },
  styleEmoji: { fontSize: 16 },
  styleName: { color: Colors.textPrimary, fontSize: 13, fontWeight: '600' },
  trendDot: { width: 6, height: 6, borderRadius: 3 },
  aiBanner: {
    borderRadius: Radius.lg,
    padding: Spacing.lg,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#6C5CE730',
    flexDirection: 'row',
    alignItems: 'center',
  },
  aiBannerLeft: { flex: 1 },
  aiBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 10,
  },
  aiBadgeText: { color: '#6C5CE7', fontSize: 11, fontWeight: '700', letterSpacing: 1 },
  aiBannerTitle: { color: Colors.textPrimary, fontSize: 24, fontWeight: '800', lineHeight: 30, marginBottom: 8 },
  aiBannerSub: { color: Colors.textSecondary, fontSize: 13, lineHeight: 20, marginBottom: 16 },
  aiBannerBtn: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  aiBannerBtnText: { color: '#6C5CE7', fontSize: 14, fontWeight: '700' },
  aiIcon: { position: 'absolute', right: 16, bottom: 16 },
  barberCard: {
    borderRadius: Radius.lg,
    padding: Spacing.md,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
  },
  barberTop: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 12 },
  barberAvatarWrap: { position: 'relative' },
  barberAvatar: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.bgCardAlt,
    borderWidth: 2,
    borderColor: Colors.border,
  },
  availDot: {
    position: 'absolute',
    bottom: 2,
    right: 2,
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: Colors.bgCard,
  },
  barberInfo: { flex: 1 },
  barberName: { color: Colors.textPrimary, fontSize: 15, fontWeight: '700', marginBottom: 2 },
  barberSpecialty: { color: Colors.textSecondary, fontSize: 12, marginBottom: 4 },
  barberMeta: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  barberRating: { color: Colors.primary, fontSize: 13, fontWeight: '700' },
  barberReviews: { color: Colors.textMuted, fontSize: 12 },
  barberPrice: { alignItems: 'flex-end' },
  priceLabel: { color: Colors.textMuted, fontSize: 10, marginBottom: 2 },
  priceValue: { color: Colors.primary, fontSize: 20, fontWeight: '800' },
  barberBadges: { flexDirection: 'row', gap: 6, marginBottom: 12 },
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    backgroundColor: Colors.primary + '15',
    borderRadius: Radius.full,
    borderWidth: 1,
    borderColor: Colors.primary + '30',
  },
  badgeText: { color: Colors.primary, fontSize: 10, fontWeight: '700' },
  barberBottom: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  barberStat: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  statText: { color: Colors.textSecondary, fontSize: 12 },
  bookNowBtn: { borderRadius: Radius.full, overflow: 'hidden' },
  bookNowGrad: { paddingHorizontal: 24, paddingVertical: 10 },
  bookNowText: { color: '#0A0A0F', fontSize: 13, fontWeight: '800' },
  loyaltyTeaser: {
    borderRadius: Radius.lg,
    padding: Spacing.lg,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  loyaltyTeaserLeft: { flex: 1 },
  loyaltyTeaserTitle: { color: Colors.textPrimary, fontSize: 20, fontWeight: '800', marginBottom: 4 },
  loyaltyTeaserSub: { color: Colors.textSecondary, fontSize: 13, marginBottom: 12 },
  xpBar: {
    height: 6,
    backgroundColor: Colors.bgGlass,
    borderRadius: 3,
    marginBottom: 6,
    overflow: 'hidden',
  },
  xpFill: {
    height: '100%',
    backgroundColor: Colors.primary,
    borderRadius: 3,
  },
  xpLabel: { color: Colors.textMuted, fontSize: 11 },
});
