import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Switch,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Colors, Spacing, Radius } from '../theme/colors';

const MENU_SECTIONS = [
  {
    title: 'My Barber Life',
    items: [
      { icon: 'calendar-clock', label: 'Appointment History', screen: 'Book', color: '#C8A96E' },
      { icon: 'camera-enhance', label: 'Hair Journey', screen: 'HairJourney', color: '#4CAF82' },
      { icon: 'crown', label: 'Loyalty & Rewards', screen: 'Loyalty', color: '#F0C849' },
      { icon: 'star-circle', label: 'My Subscription', screen: 'Subscription', color: '#6C5CE7' },
    ],
  },
  {
    title: 'Discover',
    items: [
      { icon: 'augmented-reality', label: 'AR Hairstyle Try-On', screen: 'ARTryOn', color: '#6C5CE7' },
      { icon: 'face-recognition', label: 'AI Hair Analysis', screen: 'HairAnalysis', color: '#E05C5C' },
      { icon: 'format-list-numbered', label: 'Live Queue', screen: 'Queue', color: '#5C9EE0' },
    ],
  },
  {
    title: 'Connect',
    items: [
      { icon: 'chat', label: 'Chat with Barber', screen: 'Chat', color: '#C8A96E' },
      { icon: 'account-group', label: 'Refer Friends', screen: null, color: '#4CAF82' },
      { icon: 'gift', label: 'Send Gift Card', screen: null, color: '#E05C5C' },
    ],
  },
  {
    title: 'Preferences',
    items: [
      { icon: 'bell-outline', label: 'Notifications', screen: null, toggle: true, key: 'notifications' },
      { icon: 'map-marker-outline', label: 'Location Services', screen: null, toggle: true, key: 'location' },
      { icon: 'moon-waning-crescent', label: 'Dark Mode', screen: null, toggle: true, key: 'dark', value: true },
    ],
  },
];

export default function ProfileScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const [toggles, setToggles] = useState({
    notifications: true,
    location: true,
    dark: true,
  });

  const setToggle = (key, val) => setToggles(t => ({ ...t, [key]: val }));

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Profile Hero */}
        <LinearGradient colors={['#1A1200', '#0A0A0F']} style={styles.profileHero}>
          {/* Avatar */}
          <View style={styles.avatarContainer}>
            <LinearGradient colors={Colors.gradientGold} style={styles.avatarBg}>
              <Text style={styles.avatarText}>J</Text>
            </LinearGradient>
            <TouchableOpacity style={styles.avatarEdit}>
              <MaterialCommunityIcons name="camera" size={14} color={Colors.textPrimary} />
            </TouchableOpacity>
          </View>

          <Text style={styles.profileName}>Jordan Mitchell</Text>
          <Text style={styles.profileEmail}>jordan@example.com</Text>

          {/* Tier Badge */}
          <View style={styles.tierBadge}>
            <Text style={styles.tierBadgeEmoji}>🥇</Text>
            <Text style={styles.tierBadgeText}>Gold Member</Text>
            <MaterialCommunityIcons name="chevron-right" size={16} color={Colors.primary} />
          </View>

          {/* Quick Stats */}
          <View style={styles.quickStats}>
            {[
              { label: 'Visits', value: '14' },
              { label: 'XP', value: '1.25k' },
              { label: 'Saved', value: '$48' },
              { label: 'Streak', value: '4🔥' },
            ].map((stat) => (
              <View key={stat.label} style={styles.quickStat}>
                <Text style={styles.quickStatValue}>{stat.value}</Text>
                <Text style={styles.quickStatLabel}>{stat.label}</Text>
              </View>
            ))}
          </View>
        </LinearGradient>

        {/* Subscription CTA */}
        <TouchableOpacity
          style={styles.subCta}
          onPress={() => navigation.navigate('Subscription')}
          activeOpacity={0.9}
        >
          <LinearGradient
            colors={Colors.gradientGold}
            style={StyleSheet.absoluteFill}
            borderRadius={Radius.md}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          />
          <View style={styles.subCtaLeft}>
            <MaterialCommunityIcons name="crown" size={24} color="#0A0A0F" />
            <View>
              <Text style={styles.subCtaTitle}>Upgrade to Sharp</Text>
              <Text style={styles.subCtaSub}>4 cuts/month · Save up to $90</Text>
            </View>
          </View>
          <MaterialCommunityIcons name="arrow-right" size={20} color="#0A0A0F" />
        </TouchableOpacity>

        {/* Preferred Barber */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Favorite Barber</Text>
          <TouchableOpacity style={styles.prefBarber} activeOpacity={0.85}>
            <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.md} />
            <View style={styles.prefBarberAvatar}>
              <Text style={styles.prefBarberAvatarText}>M</Text>
            </View>
            <View style={styles.prefBarberInfo}>
              <Text style={styles.prefBarberName}>Marcus "The Blade" Johnson</Text>
              <Text style={styles.prefBarberSub}>Next available: Tomorrow 10:30 AM</Text>
              <View style={styles.prefBarberMeta}>
                <MaterialCommunityIcons name="star" size={12} color={Colors.primary} />
                <Text style={styles.prefBarberRating}>4.9 · 8 visits together</Text>
              </View>
            </View>
            <TouchableOpacity style={styles.bookBtn} onPress={() => navigation.navigate('Book')}>
              <LinearGradient colors={Colors.gradientGold} style={styles.bookBtnGrad}>
                <Text style={styles.bookBtnText}>Book</Text>
              </LinearGradient>
            </TouchableOpacity>
          </TouchableOpacity>
        </View>

        {/* Menu Sections */}
        {MENU_SECTIONS.map((section) => (
          <View key={section.title} style={styles.menuSection}>
            <Text style={styles.menuSectionTitle}>{section.title}</Text>
            <View style={styles.menuCard}>
              <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.lg} />
              {section.items.map((item, i) => (
                <TouchableOpacity
                  key={item.label}
                  style={[styles.menuItem, i < section.items.length - 1 && styles.menuItemBorder]}
                  onPress={() => item.screen && navigation.navigate(item.screen)}
                  activeOpacity={0.7}
                >
                  <View style={[styles.menuIcon, { backgroundColor: item.color + '20' }]}>
                    <MaterialCommunityIcons name={item.icon} size={18} color={item.color} />
                  </View>
                  <Text style={styles.menuLabel}>{item.label}</Text>
                  {item.toggle ? (
                    <Switch
                      value={toggles[item.key]}
                      onValueChange={(v) => setToggle(item.key, v)}
                      trackColor={{ false: Colors.bgCardAlt, true: Colors.primary + '60' }}
                      thumbColor={toggles[item.key] ? Colors.primary : Colors.textMuted}
                    />
                  ) : (
                    <MaterialCommunityIcons name="chevron-right" size={18} color={Colors.textMuted} />
                  )}
                </TouchableOpacity>
              ))}
            </View>
          </View>
        ))}

        {/* Sign Out */}
        <TouchableOpacity style={styles.signOut}>
          <MaterialCommunityIcons name="logout" size={18} color={Colors.error} />
          <Text style={styles.signOutText}>Sign Out</Text>
        </TouchableOpacity>

        <Text style={styles.version}>FadeBlades v1.0.0 · Made with ❤️ for fresh cuts</Text>

        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  profileHero: {
    alignItems: 'center',
    padding: Spacing.xl,
    paddingTop: Spacing.lg,
    marginBottom: Spacing.md,
  },
  avatarContainer: { position: 'relative', marginBottom: Spacing.md },
  avatarBg: {
    width: 90,
    height: 90,
    borderRadius: 45,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: { color: '#0A0A0F', fontSize: 36, fontWeight: '900' },
  avatarEdit: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: Colors.bgCard,
    borderWidth: 2,
    borderColor: Colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  profileName: { color: Colors.textPrimary, fontSize: 24, fontWeight: '800', marginBottom: 4 },
  profileEmail: { color: Colors.textSecondary, fontSize: 14, marginBottom: Spacing.md },
  tierBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: Colors.primary + '15',
    borderRadius: Radius.full,
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginBottom: Spacing.lg,
    borderWidth: 1,
    borderColor: Colors.primary + '30',
  },
  tierBadgeEmoji: { fontSize: 18 },
  tierBadgeText: { color: Colors.primary, fontSize: 14, fontWeight: '700' },
  quickStats: {
    flexDirection: 'row',
    width: '100%',
    borderTopWidth: 1,
    borderTopColor: Colors.borderSubtle,
    paddingTop: Spacing.lg,
  },
  quickStat: { flex: 1, alignItems: 'center' },
  quickStatValue: { color: Colors.textPrimary, fontSize: 20, fontWeight: '900', marginBottom: 4 },
  quickStatLabel: { color: Colors.textMuted, fontSize: 11 },
  subCta: {
    marginHorizontal: Spacing.md,
    borderRadius: Radius.md,
    padding: Spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: Spacing.lg,
    overflow: 'hidden',
  },
  subCtaLeft: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  subCtaTitle: { color: '#0A0A0F', fontSize: 15, fontWeight: '800' },
  subCtaSub: { color: '#0A0A0F80', fontSize: 12 },
  section: { paddingHorizontal: Spacing.md, marginBottom: Spacing.lg },
  sectionTitle: { color: Colors.textSecondary, fontSize: 12, fontWeight: '700', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 10 },
  prefBarber: {
    borderRadius: Radius.md,
    padding: Spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
  },
  prefBarberAvatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.primary + '30',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: Colors.border,
  },
  prefBarberAvatarText: { color: Colors.primary, fontSize: 20, fontWeight: '900' },
  prefBarberInfo: { flex: 1 },
  prefBarberName: { color: Colors.textPrimary, fontSize: 13, fontWeight: '700', marginBottom: 2 },
  prefBarberSub: { color: Colors.textSecondary, fontSize: 11, marginBottom: 4 },
  prefBarberMeta: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  prefBarberRating: { color: Colors.textMuted, fontSize: 11 },
  bookBtn: { borderRadius: Radius.full, overflow: 'hidden' },
  bookBtnGrad: { paddingHorizontal: 16, paddingVertical: 8 },
  bookBtnText: { color: '#0A0A0F', fontSize: 12, fontWeight: '800' },
  menuSection: { paddingHorizontal: Spacing.md, marginBottom: Spacing.md },
  menuSectionTitle: { color: Colors.textMuted, fontSize: 11, fontWeight: '700', letterSpacing: 1.2, textTransform: 'uppercase', marginBottom: 8 },
  menuCard: {
    borderRadius: Radius.lg,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: Spacing.md,
  },
  menuItemBorder: { borderBottomWidth: 1, borderBottomColor: Colors.borderSubtle },
  menuIcon: {
    width: 36,
    height: 36,
    borderRadius: Radius.sm,
    alignItems: 'center',
    justifyContent: 'center',
  },
  menuLabel: { flex: 1, color: Colors.textPrimary, fontSize: 15 },
  signOut: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.md,
    marginBottom: 8,
  },
  signOutText: { color: Colors.error, fontSize: 15, fontWeight: '600' },
  version: { color: Colors.textMuted, fontSize: 11, textAlign: 'center', paddingBottom: 8 },
});
