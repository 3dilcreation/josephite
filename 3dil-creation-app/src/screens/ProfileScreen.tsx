import React from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';
import { useAuth } from '../context/AuthContext';

type Nav = NativeStackNavigationProp<RootStackParamList>;

const tierColors: Record<string, string> = {
  Bronze: '#CD7F32',
  Silver: '#C0C0C0',
  Gold: '#FFD700',
  Platinum: '#E5E4E2',
};

const ProfileScreen: React.FC = () => {
  const navigation = useNavigation<Nav>();
  const { user, isLoggedIn, logout, loyaltyPoints, loyaltyTier } = useAuth();

  const menuSections = [
    {
      title: 'My Account',
      items: [
        { icon: 'receipt-outline', label: 'My Orders', badge: '3', action: () => {} },
        { icon: 'location-outline', label: 'Saved Addresses', action: () => {} },
        { icon: 'card-outline', label: 'Payment Methods', action: () => {} },
        { icon: 'star-outline', label: 'Loyalty & Rewards', action: () => {} },
        { icon: 'gift-outline', label: 'Refer a Friend', action: () => {} },
      ],
    },
    {
      title: 'Preferences',
      items: [
        { icon: 'notifications-outline', label: 'Notifications', action: () => {} },
        { icon: 'language-outline', label: 'Language', badge: 'EN', action: () => {} },
      ],
    },
    {
      title: 'Help & Support',
      items: [
        { icon: 'chatbubble-outline', label: 'Chat Support', action: () => navigation.navigate('Contact') },
        { icon: 'help-circle-outline', label: 'FAQs', action: () => {} },
        { icon: 'document-text-outline', label: 'Terms & Privacy', action: () => {} },
        { icon: 'star-outline', label: 'Rate the App', action: () => {} },
        { icon: 'information-circle-outline', label: 'About 3DIL Creation', action: () => {} },
      ],
    },
  ];

  if (!isLoggedIn) {
    return (
      <View style={styles.container}>
        <LinearGradient colors={[Colors.primary, Colors.secondary]} style={styles.guestHeader}>
          <Text style={styles.guestEmoji}>👤</Text>
          <Text style={styles.guestTitle}>Hello, Guest!</Text>
          <Text style={styles.guestDesc}>Sign in to access orders, loyalty points, and personalized experience</Text>
        </LinearGradient>
        <View style={styles.guestActions}>
          <TouchableOpacity style={styles.signInBtn} onPress={() => navigation.navigate('Login')}>
            <LinearGradient colors={[Colors.primary, '#FF8C42']} style={styles.signInGradient}>
              <Text style={styles.signInText}>Sign In</Text>
            </LinearGradient>
          </TouchableOpacity>
          <TouchableOpacity style={styles.registerBtn} onPress={() => navigation.navigate('Register')}>
            <Text style={styles.registerText}>Create Account →</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.guestMenu}>
          {[
            { icon: 'cube-outline', label: 'Browse Products' },
            { icon: 'calculator-outline', label: 'Get Quote' },
            { icon: 'call-outline', label: 'Contact Us' },
            { icon: 'information-circle-outline', label: 'About Us' },
          ].map((item, i) => (
            <TouchableOpacity key={i} style={styles.guestMenuItem}>
              <Ionicons name={item.icon as any} size={22} color={Colors.primary} />
              <Text style={styles.guestMenuText}>{item.label}</Text>
              <Ionicons name="chevron-forward" size={16} color={Colors.textLight} style={{ marginLeft: 'auto' }} />
            </TouchableOpacity>
          ))}
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <LinearGradient colors={[Colors.secondary, Colors.primary]} style={styles.header}>
        <View style={styles.avatarContainer}>
          <View style={styles.avatar}>
            <Text style={styles.avatarText}>{user?.name.charAt(0) || '?'}</Text>
          </View>
          <View style={[styles.tierBadge, { backgroundColor: tierColors[loyaltyTier] }]}>
            <Text style={styles.tierBadgeText}>{loyaltyTier}</Text>
          </View>
        </View>
        <Text style={styles.userName}>{user?.name}</Text>
        <Text style={styles.userEmail}>{user?.email}</Text>
        <Text style={styles.userPhone}>{user?.phone}</Text>
        <Text style={styles.memberSince}>Member since {user?.memberSince}</Text>
        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{user?.totalOrders || 0}</Text>
            <Text style={styles.statLabel}>Orders</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{loyaltyPoints.toLocaleString()}</Text>
            <Text style={styles.statLabel}>Points</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{loyaltyTier}</Text>
            <Text style={styles.statLabel}>Tier</Text>
          </View>
        </View>
      </LinearGradient>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 32 }}>
        {/* Subscription Banner */}
        <TouchableOpacity style={styles.subBanner} onPress={() => navigation.navigate('Subscription')}>
          <LinearGradient colors={['#FFD700', '#FFA500']} style={styles.subBannerGradient}>
            <Text style={styles.subBannerEmoji}>⭐</Text>
            <View>
              <Text style={styles.subBannerTitle}>Upgrade to Pro</Text>
              <Text style={styles.subBannerDesc}>Save 20% + 3 free prints/month</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={Colors.secondary} />
          </LinearGradient>
        </TouchableOpacity>

        {menuSections.map((section, si) => (
          <View key={si} style={styles.menuSection}>
            <Text style={styles.menuSectionTitle}>{section.title}</Text>
            <View style={styles.menuCard}>
              {section.items.map((item, ii) => (
                <TouchableOpacity
                  key={ii}
                  style={[styles.menuItem, ii < section.items.length - 1 && styles.menuItemBorder]}
                  onPress={item.action}
                >
                  <View style={styles.menuIconBox}>
                    <Ionicons name={item.icon as any} size={22} color={Colors.primary} />
                  </View>
                  <Text style={styles.menuLabel}>{item.label}</Text>
                  <View style={styles.menuRight}>
                    {item.badge && (
                      <View style={styles.menuBadge}>
                        <Text style={styles.menuBadgeText}>{item.badge}</Text>
                      </View>
                    )}
                    <Ionicons name="chevron-forward" size={16} color={Colors.textLight} />
                  </View>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        ))}

        <TouchableOpacity
          style={styles.logoutBtn}
          onPress={() =>
            Alert.alert('Sign Out', 'Are you sure you want to sign out?', [
              { text: 'Cancel', style: 'cancel' },
              { text: 'Sign Out', style: 'destructive', onPress: logout },
            ])
          }
        >
          <Ionicons name="log-out-outline" size={20} color={Colors.error} />
          <Text style={styles.logoutText}>Sign Out</Text>
        </TouchableOpacity>

        <Text style={styles.appVersion}>3DIL Creation v1.0.0</Text>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { paddingTop: 52, paddingBottom: 24, paddingHorizontal: Spacing.md, alignItems: 'center' },
  avatarContainer: { position: 'relative', marginBottom: 12 },
  avatar: { width: 80, height: 80, borderRadius: 40, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center', borderWidth: 3, borderColor: Colors.white },
  avatarText: { color: Colors.white, fontSize: 36, fontWeight: '900' },
  tierBadge: { position: 'absolute', bottom: -4, right: -8, paddingHorizontal: 10, paddingVertical: 3, borderRadius: 10 },
  tierBadgeText: { fontSize: 9, fontWeight: '900', color: Colors.secondary },
  userName: { color: Colors.white, fontSize: FontSize.xxl, fontWeight: '900', marginBottom: 2 },
  userEmail: { color: 'rgba(255,255,255,0.75)', fontSize: FontSize.md, marginBottom: 2 },
  userPhone: { color: 'rgba(255,255,255,0.7)', fontSize: FontSize.sm, marginBottom: 4 },
  memberSince: { color: 'rgba(255,255,255,0.55)', fontSize: FontSize.xs, marginBottom: 16 },
  statsRow: { flexDirection: 'row', backgroundColor: 'rgba(255,255,255,0.15)', borderRadius: BorderRadius.md, paddingVertical: 12, paddingHorizontal: 20, gap: 24 },
  statItem: { alignItems: 'center' },
  statValue: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '900' },
  statLabel: { color: 'rgba(255,255,255,0.7)', fontSize: FontSize.xs },
  statDivider: { width: 1, backgroundColor: 'rgba(255,255,255,0.3)' },
  subBanner: { margin: 12, borderRadius: BorderRadius.md, overflow: 'hidden' },
  subBannerGradient: { flexDirection: 'row', alignItems: 'center', padding: 14, gap: 10, borderRadius: BorderRadius.md },
  subBannerEmoji: { fontSize: 28 },
  subBannerTitle: { fontSize: FontSize.md, fontWeight: '800', color: Colors.secondary, marginBottom: 2 },
  subBannerDesc: { fontSize: FontSize.xs, color: Colors.secondary, opacity: 0.8 },
  menuSection: { paddingHorizontal: 12, marginBottom: 4 },
  menuSectionTitle: { fontSize: FontSize.sm, fontWeight: '800', color: Colors.textSecondary, marginBottom: 8, marginTop: 12, paddingLeft: 4, letterSpacing: 0.5, textTransform: 'uppercase' },
  menuCard: { backgroundColor: Colors.card, borderRadius: BorderRadius.md, ...Shadows.small },
  menuItem: { flexDirection: 'row', alignItems: 'center', paddingVertical: 14, paddingHorizontal: 14, gap: 12 },
  menuItemBorder: { borderBottomWidth: 1, borderBottomColor: Colors.border },
  menuIconBox: { width: 38, height: 38, borderRadius: 10, backgroundColor: Colors.primary + '15', alignItems: 'center', justifyContent: 'center' },
  menuLabel: { flex: 1, fontSize: FontSize.md, fontWeight: '600', color: Colors.textPrimary },
  menuRight: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  menuBadge: { backgroundColor: Colors.primary, borderRadius: 10, paddingHorizontal: 8, paddingVertical: 2, minWidth: 24, alignItems: 'center' },
  menuBadgeText: { color: Colors.white, fontSize: 10, fontWeight: '800' },
  logoutBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, margin: 12, backgroundColor: Colors.error + '15', borderRadius: BorderRadius.md, paddingVertical: 16, borderWidth: 1.5, borderColor: Colors.error + '30' },
  logoutText: { color: Colors.error, fontWeight: '800', fontSize: FontSize.lg },
  appVersion: { textAlign: 'center', color: Colors.textLight, fontSize: FontSize.xs, marginTop: 4 },
  guestHeader: { paddingTop: 80, paddingBottom: 40, paddingHorizontal: Spacing.lg, alignItems: 'center' },
  guestEmoji: { fontSize: 64, marginBottom: 12 },
  guestTitle: { color: Colors.white, fontSize: FontSize.xxxl, fontWeight: '900', marginBottom: 8 },
  guestDesc: { color: 'rgba(255,255,255,0.8)', fontSize: FontSize.md, textAlign: 'center', lineHeight: 22 },
  guestActions: { padding: 16, gap: 10 },
  signInBtn: { borderRadius: BorderRadius.lg, overflow: 'hidden' },
  signInGradient: { paddingVertical: 18, alignItems: 'center', borderRadius: BorderRadius.lg },
  signInText: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '800' },
  registerBtn: { paddingVertical: 16, alignItems: 'center' },
  registerText: { color: Colors.primary, fontWeight: '800', fontSize: FontSize.lg },
  guestMenu: { margin: 12, backgroundColor: Colors.card, borderRadius: BorderRadius.md, ...Shadows.small },
  guestMenuItem: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 14, paddingHorizontal: 14, borderBottomWidth: 1, borderBottomColor: Colors.border },
  guestMenuText: { flex: 1, fontSize: FontSize.md, fontWeight: '600', color: Colors.textPrimary },
});

export default ProfileScreen;
