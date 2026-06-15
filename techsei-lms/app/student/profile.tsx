// ============================================================
// TechSei LMS — Student Profile Screen
// ============================================================
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Switch,
  Alert,
  Modal,
  FlatList,
  TextInput,
  Dimensions,
  Animated,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { useAuthStore } from '../../stores/authStore';
import { useGamificationStore } from '../../stores/gamificationStore';
import { SUPPORTED_LANGUAGES } from '../../constants/i18n';
import type { LanguageCode, SubscriptionTier } from '../../types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// ─── Colors ───────────────────────────────────────────────────────────────────
const C = {
  bg: '#0A0A1A',
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  primary: '#6C63FF',
  accent: '#43E97B',
  gold: '#FFD700',
  warning: '#FFB84C',
  text: '#FFFFFF',
  textSec: '#9494B8',
  textMuted: '#5A5A7A',
  border: '#2A2A4A',
  error: '#FF4B4B',
  overlay: 'rgba(0,0,0,0.8)',
};

// ─── Subscription badge colors ────────────────────────────────────────────────
const TIER_CONFIG: Record<SubscriptionTier, { label: string; colors: [string, string]; icon: string }> = {
  free: { label: 'Free', colors: [C.surfaceLight, C.border], icon: '🎓' },
  pro: { label: 'Pro', colors: [C.primary, '#8B5CF6'], icon: '⚡' },
  premium: { label: 'Premium', colors: [C.gold, '#FFA500'], icon: '👑' },
};

// ─── Avatar component ─────────────────────────────────────────────────────────
function ProfileAvatar({
  name,
  onEdit,
}: {
  name: string;
  onEdit: () => void;
}) {
  const initials = name
    .split(' ')
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() ?? '')
    .join('');

  return (
    <View style={styles.avatarWrapper}>
      <LinearGradient
        colors={[C.primary, '#8B5CF6']}
        style={styles.avatar}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        <Text style={styles.avatarText}>{initials || '?'}</Text>
      </LinearGradient>
      <TouchableOpacity style={styles.editAvatarBtn} onPress={onEdit}>
        <Ionicons name="camera" size={14} color={C.text} />
      </TouchableOpacity>
    </View>
  );
}

// ─── Section header ───────────────────────────────────────────────────────────
function SectionHeader({ title }: { title: string }) {
  return <Text style={styles.sectionHeader}>{title}</Text>;
}

// ─── Settings row ─────────────────────────────────────────────────────────────
function SettingsRow({
  icon,
  label,
  value,
  onPress,
  toggle,
  toggleValue,
  onToggle,
  danger,
}: {
  icon: React.ComponentProps<typeof Ionicons>['name'];
  label: string;
  value?: string;
  onPress?: () => void;
  toggle?: boolean;
  toggleValue?: boolean;
  onToggle?: (v: boolean) => void;
  danger?: boolean;
}) {
  return (
    <TouchableOpacity
      style={styles.settingsRow}
      onPress={onPress}
      disabled={toggle}
      activeOpacity={toggle ? 1 : 0.7}
    >
      <View style={[styles.settingsIcon, { backgroundColor: danger ? 'rgba(255,75,75,0.12)' : C.surfaceLight }]}>
        <Ionicons name={icon} size={18} color={danger ? C.error : C.textSec} />
      </View>
      <Text style={[styles.settingsLabel, danger && { color: C.error }]}>{label}</Text>
      <View style={styles.settingsRight}>
        {value ? <Text style={styles.settingsValue}>{value}</Text> : null}
        {toggle ? (
          <Switch
            value={toggleValue ?? false}
            onValueChange={onToggle}
            trackColor={{ false: C.border, true: C.primary }}
            thumbColor={C.text}
          />
        ) : (
          <Ionicons
            name={danger ? 'chevron-forward' : 'chevron-forward'}
            size={16}
            color={danger ? C.error : C.textMuted}
          />
        )}
      </View>
    </TouchableOpacity>
  );
}

// ─── Language Picker Modal ────────────────────────────────────────────────────
function LanguageModal({
  visible,
  current,
  onSelect,
  onClose,
}: {
  visible: boolean;
  current: LanguageCode;
  onSelect: (code: LanguageCode) => void;
  onClose: () => void;
}) {
  const [search, setSearch] = useState('');
  const [pending, setPending] = useState<LanguageCode>(current);

  const filtered = SUPPORTED_LANGUAGES.filter(
    (l) =>
      l.name.toLowerCase().includes(search.toLowerCase()) ||
      l.nativeName.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <View style={styles.modalBg}>
        <SafeAreaView style={styles.modalCard} edges={['bottom']}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Select Language</Text>
            <TouchableOpacity onPress={onClose}>
              <Ionicons name="close" size={24} color={C.textSec} />
            </TouchableOpacity>
          </View>

          {/* Search */}
          <View style={styles.searchBox}>
            <Ionicons name="search-outline" size={16} color={C.textMuted} />
            <TextInput
              style={styles.searchInput}
              placeholder="Search..."
              placeholderTextColor={C.textMuted}
              value={search}
              onChangeText={setSearch}
            />
          </View>

          {/* Language grid */}
          <FlatList
            data={filtered}
            keyExtractor={(l) => l.code}
            numColumns={2}
            renderItem={({ item }) => {
              const active = item.code === pending;
              return (
                <TouchableOpacity
                  style={[styles.langItem, active && styles.langItemActive]}
                  onPress={() => setPending(item.code as LanguageCode)}
                >
                  <Text style={styles.langFlag}>{item.flag}</Text>
                  <View style={{ flex: 1 }}>
                    <Text style={[styles.langName, active && { color: C.primary }]}>
                      {item.name}
                    </Text>
                    <Text style={styles.langNative}>{item.nativeName}</Text>
                  </View>
                  {active && <Ionicons name="checkmark-circle" size={16} color={C.primary} />}
                </TouchableOpacity>
              );
            }}
            contentContainerStyle={{ paddingBottom: 16 }}
            showsVerticalScrollIndicator={false}
          />

          {/* Save button */}
          <TouchableOpacity
            style={styles.saveBtn}
            onPress={() => { onSelect(pending); onClose(); }}
          >
            <LinearGradient
              colors={[C.primary, '#8B5CF6']}
              style={styles.saveBtnGrad}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
            >
              <Text style={styles.saveBtnText}>Save Language</Text>
            </LinearGradient>
          </TouchableOpacity>
        </SafeAreaView>
      </View>
    </Modal>
  );
}

// ─── Quick Action ─────────────────────────────────────────────────────────────
function QuickAction({
  icon,
  label,
  color,
  onPress,
}: {
  icon: React.ComponentProps<typeof Ionicons>['name'];
  label: string;
  color: string;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity style={styles.quickAction} onPress={onPress}>
      <View style={[styles.quickActionIcon, { backgroundColor: `${color}22` }]}>
        <Ionicons name={icon} size={22} color={color} />
      </View>
      <Text style={styles.quickActionLabel}>{label}</Text>
    </TouchableOpacity>
  );
}

// ─── Main Screen ──────────────────────────────────────────────────────────────
export default function ProfileScreen() {
  const { user, profile, signOut, setLanguage, updateProfile } = useAuthStore();
  const { xp, level, streak } = useGamificationStore();

  const [langModalVisible, setLangModalVisible] = useState(false);

  // Notification toggles
  const [notifCourseUpdates, setNotifCourseUpdates] = useState(true);
  const [notifDailyReminder, setNotifDailyReminder] = useState(true);
  const [notifStreakAlerts, setNotifStreakAlerts] = useState(true);
  const [notifWeeklyReport, setNotifWeeklyReport] = useState(false);

  // Appearance / Privacy
  const [darkMode, setDarkMode] = useState(true);
  const [accountPublic, setAccountPublic] = useState(true);

  const displayName = user?.name ?? 'Student';
  const displayEmail = user?.email ?? '';
  const memberSince = user?.created_at
    ? new Date(user.created_at).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })
    : '—';
  const tier: SubscriptionTier = profile?.subscription_tier ?? 'free';
  const tierCfg = TIER_CONFIG[tier];
  const currentLang = SUPPORTED_LANGUAGES.find((l) => l.code === (user?.language_pref ?? 'en'));

  const subscriptionExpiry = '2026-09-01'; // placeholder

  const handlePickAvatar = useCallback(async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });
    if (!result.canceled && result.assets[0]) {
      // In production: upload to Supabase Storage and update avatar_url
      Alert.alert('Avatar', 'Avatar upload coming soon!');
    }
  }, []);

  const handleSignOut = useCallback(() => {
    Alert.alert('Sign Out', 'Are you sure you want to sign out?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Sign Out', style: 'destructive', onPress: () => signOut() },
    ]);
  }, [signOut]);

  const handleDeleteAccount = useCallback(() => {
    Alert.alert(
      'Delete Account',
      'This will permanently delete your account and all your progress. This cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => Alert.alert('Contact Support', 'Please contact support@techsei.app to delete your account.'),
        },
      ]
    );
  }, []);

  const handleLanguageSelect = useCallback(
    async (code: LanguageCode) => {
      try {
        await setLanguage(code);
      } catch {
        Alert.alert('Error', 'Failed to update language preference.');
      }
    },
    [setLanguage]
  );

  const miniStats = [
    { label: 'Level', value: level, icon: '⚡' },
    { label: 'XP', value: xp.toLocaleString(), icon: '🌟' },
    { label: 'Streak', value: `${streak}d`, icon: '🔥' },
    { label: 'Courses', value: profile?.total_courses_completed ?? 0, icon: '📚' },
  ];

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        {/* ── Profile Header ──────────────────────────────────────────────── */}
        <LinearGradient
          colors={['#141428', '#1E1E3A']}
          style={styles.profileHeader}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        >
          <ProfileAvatar name={displayName} onEdit={handlePickAvatar} />

          <Text style={styles.profileName}>{displayName}</Text>
          <Text style={styles.profileEmail}>{displayEmail}</Text>
          <Text style={styles.profileSince}>Member since {memberSince}</Text>

          {/* Tier badge */}
          <LinearGradient
            colors={tierCfg.colors}
            style={styles.tierBadge}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <Text style={styles.tierBadgeText}>{tierCfg.icon} {tierCfg.label}</Text>
          </LinearGradient>
        </LinearGradient>

        {/* ── Mini Stats ──────────────────────────────────────────────────── */}
        <View style={styles.miniStats}>
          {miniStats.map((s) => (
            <View key={s.label} style={styles.miniStatCard}>
              <Text style={styles.miniStatIcon}>{s.icon}</Text>
              <Text style={styles.miniStatValue}>{s.value}</Text>
              <Text style={styles.miniStatLabel}>{s.label}</Text>
            </View>
          ))}
        </View>

        {/* ── Quick Actions ───────────────────────────────────────────────── */}
        <SectionHeader title="Quick Actions" />
        <View style={styles.quickActionsGrid}>
          <QuickAction icon="create-outline" label="Edit Profile" color={C.primary} onPress={() => Alert.alert('Edit Profile', 'Profile editor coming soon!')} />
          <QuickAction icon="card-outline" label="Subscription" color={C.gold} onPress={() => {}} />
          <QuickAction icon="ribbon-outline" label="Certificates" color={C.accent} onPress={() => Alert.alert('Certificates', 'View in Progress tab!')} />
          <QuickAction icon="share-social-outline" label="Share Profile" color={C.warning} onPress={() => Alert.alert('Share', 'Share link: techsei.app/u/' + (user?.id ?? ''))} />
        </View>

        {/* ── Settings ────────────────────────────────────────────────────── */}
        <SectionHeader title="Settings" />
        <View style={styles.settingsCard}>
          {/* Language */}
          <SettingsRow
            icon="language-outline"
            label="Language"
            value={`${currentLang?.flag ?? '🌐'} ${currentLang?.name ?? 'English'}`}
            onPress={() => setLangModalVisible(true)}
          />
          <View style={styles.divider} />

          {/* Notifications */}
          <Text style={styles.settingsGroupLabel}>Notifications</Text>
          <SettingsRow icon="school-outline" label="Course Updates" toggle toggleValue={notifCourseUpdates} onToggle={setNotifCourseUpdates} />
          <View style={styles.divider} />
          <SettingsRow icon="alarm-outline" label="Daily Reminders" toggle toggleValue={notifDailyReminder} onToggle={setNotifDailyReminder} />
          <View style={styles.divider} />
          <SettingsRow icon="flame-outline" label="Streak Alerts" toggle toggleValue={notifStreakAlerts} onToggle={setNotifStreakAlerts} />
          <View style={styles.divider} />
          <SettingsRow icon="bar-chart-outline" label="Weekly Report" toggle toggleValue={notifWeeklyReport} onToggle={setNotifWeeklyReport} />
          <View style={styles.divider} />

          {/* Appearance */}
          <Text style={styles.settingsGroupLabel}>Appearance</Text>
          <SettingsRow icon="moon-outline" label="Dark Mode" toggle toggleValue={darkMode} onToggle={setDarkMode} />
          <View style={styles.divider} />

          {/* Privacy */}
          <Text style={styles.settingsGroupLabel}>Privacy</Text>
          <SettingsRow icon="eye-outline" label="Public Profile" toggle toggleValue={accountPublic} onToggle={setAccountPublic} />
        </View>

        {/* ── Subscription Card ────────────────────────────────────────────── */}
        <SectionHeader title="Subscription" />
        <LinearGradient
          colors={tier === 'free' ? ['#141428', '#1E1E3A'] : [C.primary, '#8B5CF6']}
          style={styles.subscriptionCard}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        >
          <View style={styles.subCardTop}>
            <View>
              <Text style={styles.subTierLabel}>{tierCfg.icon} {tierCfg.label} Plan</Text>
              {tier !== 'free' && (
                <Text style={styles.subExpiry}>
                  Expires: {new Date(subscriptionExpiry).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                </Text>
              )}
            </View>
            <TouchableOpacity style={styles.subActionBtn}>
              <Text style={styles.subActionText}>
                {tier === 'free' ? 'Upgrade to Pro' : 'Manage Plan'}
              </Text>
            </TouchableOpacity>
          </View>
          {tier === 'free' && (
            <Text style={styles.subPromo}>
              Unlock unlimited courses, AI tutoring, and certificates with Pro.
            </Text>
          )}
        </LinearGradient>

        {/* ── Danger Zone ─────────────────────────────────────────────────── */}
        <SectionHeader title="Danger Zone" />
        <View style={styles.settingsCard}>
          <SettingsRow
            icon="trash-outline"
            label="Delete Account"
            onPress={handleDeleteAccount}
            danger
          />
        </View>

        {/* ── Sign Out ────────────────────────────────────────────────────── */}
        <TouchableOpacity style={styles.signOutBtn} onPress={handleSignOut}>
          <Ionicons name="log-out-outline" size={20} color={C.error} />
          <Text style={styles.signOutText}>Sign Out</Text>
        </TouchableOpacity>

        <View style={{ height: 40 }} />
      </ScrollView>

      {/* Language Picker Modal */}
      <LanguageModal
        visible={langModalVisible}
        current={(user?.language_pref ?? 'en') as LanguageCode}
        onSelect={handleLanguageSelect}
        onClose={() => setLangModalVisible(false)}
      />
    </SafeAreaView>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: C.bg },
  scroll: { flex: 1 },
  content: { paddingHorizontal: 16, paddingTop: 8, paddingBottom: 16 },

  // Profile header
  profileHeader: {
    borderRadius: 24,
    padding: 24,
    alignItems: 'center',
    marginBottom: 16,
    borderWidth: 1,
    borderColor: C.border,
  },
  avatarWrapper: { position: 'relative', marginBottom: 12 },
  avatar: {
    width: 90, height: 90, borderRadius: 45,
    alignItems: 'center', justifyContent: 'center',
  },
  avatarText: { fontSize: 34, fontWeight: '800', color: C.text },
  editAvatarBtn: {
    position: 'absolute', bottom: 0, right: 0,
    width: 28, height: 28, borderRadius: 14,
    backgroundColor: C.primary,
    alignItems: 'center', justifyContent: 'center',
    borderWidth: 2, borderColor: C.bg,
  },
  profileName: { fontSize: 22, fontWeight: '800', color: C.text, marginBottom: 4 },
  profileEmail: { fontSize: 13, color: C.textSec, marginBottom: 2 },
  profileSince: { fontSize: 12, color: C.textMuted, marginBottom: 12 },
  tierBadge: {
    paddingHorizontal: 16, paddingVertical: 6,
    borderRadius: 20,
  },
  tierBadgeText: { fontSize: 13, fontWeight: '700', color: C.text },

  // Mini stats
  miniStats: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 24,
  },
  miniStatCard: {
    flex: 1,
    backgroundColor: C.surface,
    borderRadius: 16,
    paddingVertical: 14,
    alignItems: 'center',
    gap: 2,
    borderWidth: 1,
    borderColor: C.border,
  },
  miniStatIcon: { fontSize: 18 },
  miniStatValue: { fontSize: 16, fontWeight: '800', color: C.text },
  miniStatLabel: { fontSize: 10, color: C.textSec, fontWeight: '600' },

  // Section header
  sectionHeader: {
    fontSize: 16, fontWeight: '800', color: C.text,
    marginBottom: 12,
    marginTop: 4,
  },

  // Quick actions
  quickActionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 24,
  },
  quickAction: {
    width: (SCREEN_WIDTH - 32 - 10) / 2,
    backgroundColor: C.surface,
    borderRadius: 16,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    borderWidth: 1,
    borderColor: C.border,
  },
  quickActionIcon: {
    width: 40, height: 40, borderRadius: 12,
    alignItems: 'center', justifyContent: 'center',
  },
  quickActionLabel: { fontSize: 13, fontWeight: '700', color: C.text, flex: 1 },

  // Settings card
  settingsCard: {
    backgroundColor: C.surface,
    borderRadius: 18,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: C.border,
    overflow: 'hidden',
  },
  settingsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  settingsIcon: {
    width: 34, height: 34, borderRadius: 10,
    alignItems: 'center', justifyContent: 'center',
  },
  settingsLabel: { flex: 1, fontSize: 14, fontWeight: '600', color: C.text },
  settingsRight: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  settingsValue: { fontSize: 13, color: C.textSec },
  settingsGroupLabel: {
    fontSize: 11, fontWeight: '700', color: C.textMuted,
    letterSpacing: 1, textTransform: 'uppercase',
    paddingHorizontal: 16, paddingTop: 12, paddingBottom: 4,
  },
  divider: { height: 1, backgroundColor: C.border, marginLeft: 62 },

  // Subscription
  subscriptionCard: {
    borderRadius: 20,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: C.border,
  },
  subCardTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  subTierLabel: { fontSize: 18, fontWeight: '800', color: C.text },
  subExpiry: { fontSize: 12, color: 'rgba(255,255,255,0.7)', marginTop: 2 },
  subActionBtn: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingHorizontal: 14, paddingVertical: 8,
    borderRadius: 12,
  },
  subActionText: { fontSize: 13, fontWeight: '700', color: C.text },
  subPromo: { fontSize: 13, color: C.textSec, lineHeight: 19 },

  // Sign out
  signOutBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: 'rgba(255,75,75,0.1)',
    borderRadius: 18,
    paddingVertical: 16,
    marginTop: 4,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: 'rgba(255,75,75,0.25)',
  },
  signOutText: { fontSize: 15, fontWeight: '700', color: C.error },

  // Language modal
  modalBg: {
    flex: 1,
    backgroundColor: C.overlay,
    justifyContent: 'flex-end',
  },
  modalCard: {
    backgroundColor: C.surface,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    maxHeight: '85%',
    paddingTop: 20,
    paddingHorizontal: 16,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  modalTitle: { fontSize: 20, fontWeight: '800', color: C.text },
  searchBox: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    backgroundColor: C.surfaceLight,
    borderRadius: 14,
    paddingHorizontal: 14, paddingVertical: 10,
    marginBottom: 12,
    borderWidth: 1, borderColor: C.border,
  },
  searchInput: { flex: 1, fontSize: 14, color: C.text },
  langItem: {
    flex: 1,
    flexDirection: 'row', alignItems: 'center', gap: 10,
    backgroundColor: C.surfaceLight,
    borderRadius: 14,
    padding: 12, margin: 4,
    borderWidth: 1, borderColor: C.border,
  },
  langItemActive: {
    borderColor: C.primary,
    backgroundColor: 'rgba(108,99,255,0.12)',
  },
  langFlag: { fontSize: 22 },
  langName: { fontSize: 13, fontWeight: '700', color: C.text },
  langNative: { fontSize: 11, color: C.textSec },
  saveBtn: { marginTop: 12, marginBottom: 8, borderRadius: 16, overflow: 'hidden' },
  saveBtnGrad: { paddingVertical: 16, alignItems: 'center' },
  saveBtnText: { fontSize: 16, fontWeight: '800', color: C.text },
});
