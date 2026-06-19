// ============================================================
// TechSei LMS — Admin Dashboard Screen
// ============================================================
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  RefreshControl,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { StatCard } from '../../components/admin/StatCard';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// ─── Design tokens ───────────────────────────────────────────
const COLORS = {
  background: '#0A0A1A',
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  primary: '#6C63FF',
  accent: '#43E97B',
  text: '#FFFFFF',
  textMuted: '#8A8AAA',
  border: '#2A2A4A',
  warning: '#FFB84C',
  error: '#FF6B6B',
  blue: '#4FC3F7',
  orange: '#FF9800',
};

// ─── Mock data ────────────────────────────────────────────────
interface ActivityItem {
  id: string;
  type: 'enrolled' | 'completed' | 'signup' | 'payment' | 'review';
  message: string;
  timestamp: string;
  icon: React.ComponentProps<typeof Ionicons>['name'];
  iconColor: string;
}

const MOCK_ACTIVITY: ActivityItem[] = [
  { id: '1', type: 'payment', message: 'Payment received from Marcus Chen — $29.99', timestamp: '2 min ago', icon: 'card', iconColor: COLORS.accent },
  { id: '2', type: 'signup', message: 'New student signed up: Priya Sharma', timestamp: '8 min ago', icon: 'person-add', iconColor: COLORS.blue },
  { id: '3', type: 'completed', message: 'Alex Thompson completed "React Native Mastery"', timestamp: '15 min ago', icon: 'trophy', iconColor: COLORS.warning },
  { id: '4', type: 'enrolled', message: 'Jordan Lee enrolled in "Python for AI"', timestamp: '23 min ago', icon: 'book', iconColor: COLORS.primary },
  { id: '5', type: 'payment', message: 'Subscription renewed: Sarah Kim — Pro Plan', timestamp: '41 min ago', icon: 'card', iconColor: COLORS.accent },
  { id: '6', type: 'completed', message: 'David Park completed "TypeScript Deep Dive"', timestamp: '1h ago', icon: 'trophy', iconColor: COLORS.warning },
  { id: '7', type: 'signup', message: 'New student signed up: Elena Rodriguez', timestamp: '1h 15m ago', icon: 'person-add', iconColor: COLORS.blue },
  { id: '8', type: 'review', message: 'New 5★ review on "Full-Stack Web Dev"', timestamp: '2h ago', icon: 'star', iconColor: '#FFD700' },
  { id: '9', type: 'enrolled', message: 'Mohammed Al-Hassan enrolled in "Cloud Architecture"', timestamp: '2h 30m ago', icon: 'book', iconColor: COLORS.primary },
  { id: '10', type: 'payment', message: 'Payment received from Yuki Tanaka — $49.99', timestamp: '3h ago', icon: 'card', iconColor: COLORS.accent },
];

interface PendingAction {
  id: string;
  label: string;
  count: number;
  icon: React.ComponentProps<typeof Ionicons>['name'];
  color: string;
}

const PENDING_ACTIONS: PendingAction[] = [
  { id: '1', label: 'Courses awaiting review', count: 3, icon: 'hourglass-outline', color: COLORS.warning },
  { id: '2', label: 'Support tickets open', count: 7, icon: 'chatbubble-ellipses-outline', color: COLORS.error },
  { id: '3', label: 'Pending refund requests', count: 2, icon: 'return-down-back-outline', color: COLORS.orange },
  { id: '4', label: 'Content flagged for review', count: 1, icon: 'flag-outline', color: '#A855F7' },
];

// ─── Sub-components ───────────────────────────────────────────
function AdminAvatar() {
  return (
    <LinearGradient
      colors={[COLORS.primary, '#A855F7']}
      style={styles.adminAvatar}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      <Ionicons name="shield-checkmark" size={20} color="#FFFFFF" />
    </LinearGradient>
  );
}

function SectionHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <View style={styles.sectionHeader}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {subtitle && <Text style={styles.sectionSubtitle}>{subtitle}</Text>}
    </View>
  );
}

function ActivityCard({ item }: { item: ActivityItem }) {
  return (
    <View style={styles.activityItem}>
      <View style={[styles.activityIcon, { backgroundColor: `${item.iconColor}20` }]}>
        <Ionicons name={item.icon} size={16} color={item.iconColor} />
      </View>
      <View style={styles.activityInfo}>
        <Text style={styles.activityMessage} numberOfLines={2}>{item.message}</Text>
        <Text style={styles.activityTime}>{item.timestamp}</Text>
      </View>
    </View>
  );
}

function QuickActionButton({
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
    <TouchableOpacity style={styles.quickActionBtn} onPress={onPress} activeOpacity={0.8}>
      <LinearGradient
        colors={[`${color}30`, `${color}15`]}
        style={styles.quickActionGradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        <View style={[styles.quickActionIcon, { backgroundColor: `${color}25` }]}>
          <Ionicons name={icon} size={22} color={color} />
        </View>
        <Text style={styles.quickActionLabel}>{label}</Text>
      </LinearGradient>
    </TouchableOpacity>
  );
}

function SnapshotItem({
  label,
  value,
  icon,
  color,
}: {
  label: string;
  value: string;
  icon: React.ComponentProps<typeof Ionicons>['name'];
  color: string;
}) {
  return (
    <View style={styles.snapshotItem}>
      <View style={[styles.snapshotIcon, { backgroundColor: `${color}20` }]}>
        <Ionicons name={icon} size={18} color={color} />
      </View>
      <Text style={[styles.snapshotValue, { color }]}>{value}</Text>
      <Text style={styles.snapshotLabel}>{label}</Text>
    </View>
  );
}

function PendingActionCard({ item }: { item: PendingAction }) {
  return (
    <TouchableOpacity
      style={styles.pendingCard}
      activeOpacity={0.8}
      onPress={() => Alert.alert(item.label, `You have ${item.count} item(s) requiring attention.`)}
    >
      <View style={[styles.pendingIconWrap, { backgroundColor: `${item.color}20` }]}>
        <Ionicons name={item.icon} size={20} color={item.color} />
      </View>
      <View style={styles.pendingInfo}>
        <Text style={styles.pendingLabel}>{item.label}</Text>
      </View>
      <View style={[styles.pendingBadge, { backgroundColor: `${item.color}25`, borderColor: `${item.color}60` }]}>
        <Text style={[styles.pendingCount, { color: item.color }]}>{item.count}</Text>
      </View>
      <Ionicons name="chevron-forward" size={16} color={COLORS.textMuted} />
    </TouchableOpacity>
  );
}

// ─── Main Screen ──────────────────────────────────────────────
export default function AdminDashboard() {
  const [refreshing, setRefreshing] = useState(false);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1500);
  }, []);

  const handleQuickAction = (action: string) => {
    Alert.alert('Quick Action', `${action} — feature coming soon!`);
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={['bottom']}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={COLORS.primary}
            colors={[COLORS.primary]}
          />
        }
      >
        {/* ── Header ── */}
        <LinearGradient
          colors={['#141428', '#0A0A1A']}
          style={styles.header}
          start={{ x: 0, y: 0 }}
          end={{ x: 0, y: 1 }}
        >
          <View style={styles.headerTop}>
            <View style={styles.headerLeft}>
              <LinearGradient
                colors={[COLORS.primary, '#A855F7']}
                style={styles.logoBox}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
              >
                <Text style={styles.logoText}>T</Text>
              </LinearGradient>
              <View>
                <Text style={styles.logoLabel}>TechSei</Text>
                <Text style={styles.logoSub}>Admin Portal</Text>
              </View>
            </View>
            <AdminAvatar />
          </View>
          <View style={styles.welcomeRow}>
            <Text style={styles.welcomeText}>Welcome back,</Text>
            <Text style={styles.welcomeName}>Administrator</Text>
          </View>
          <View style={styles.headerDateRow}>
            <Ionicons name="calendar-outline" size={13} color={COLORS.textMuted} />
            <Text style={styles.headerDate}>
              {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
            </Text>
          </View>
        </LinearGradient>

        {/* ── Stats Grid ── */}
        <View style={styles.section}>
          <SectionHeader title="Platform Overview" />
          <View style={styles.statsGrid}>
            <View style={styles.statsRow}>
              <StatCard
                title="Total Students"
                value={1284}
                icon="people"
                color={COLORS.blue}
                trend={{ value: 12, direction: 'up' }}
                style={styles.statCardHalf}
              />
              <StatCard
                title="Active Courses"
                value={47}
                icon="book"
                color={COLORS.primary}
                trend={{ value: 3, direction: 'up' }}
                style={styles.statCardHalf}
              />
            </View>
            <View style={styles.statsRow}>
              <StatCard
                title="Monthly Revenue"
                value="$18,420"
                icon="card"
                color={COLORS.accent}
                trend={{ value: 8, direction: 'up' }}
                style={styles.statCardHalf}
              />
              <StatCard
                title="Completion Rate"
                value="73%"
                icon="checkmark-circle"
                color={COLORS.orange}
                trend={{ value: 2, direction: 'down' }}
                style={styles.statCardHalf}
              />
            </View>
          </View>
        </View>

        {/* ── Today's Snapshot ── */}
        <View style={styles.section}>
          <SectionHeader title="Today's Snapshot" subtitle="Live platform activity" />
          <View style={[styles.card, styles.snapshotCard]}>
            <View style={styles.snapshotRow}>
              <SnapshotItem label="Students Online" value="142" icon="wifi" color={COLORS.accent} />
              <View style={styles.snapshotDivider} />
              <SnapshotItem label="Lessons Done" value="831" icon="play-circle" color={COLORS.primary} />
              <View style={styles.snapshotDivider} />
              <SnapshotItem label="Revenue Today" value="$620" icon="trending-up" color={COLORS.warning} />
            </View>
          </View>
        </View>

        {/* ── Quick Actions ── */}
        <View style={styles.section}>
          <SectionHeader title="Quick Actions" />
          <View style={styles.quickActionsGrid}>
            <QuickActionButton
              icon="person-add-outline"
              label="Add Student"
              color={COLORS.blue}
              onPress={() => handleQuickAction('Add Student')}
            />
            <QuickActionButton
              icon="add-circle-outline"
              label="Create Course"
              color={COLORS.primary}
              onPress={() => handleQuickAction('Create Course')}
            />
            <QuickActionButton
              icon="notifications-outline"
              label="Send Notification"
              color={COLORS.warning}
              onPress={() => handleQuickAction('Send Notification')}
            />
            <QuickActionButton
              icon="download-outline"
              label="Export Data"
              color={COLORS.accent}
              onPress={() => handleQuickAction('Export Data')}
            />
          </View>
        </View>

        {/* ── Pending Actions ── */}
        <View style={styles.section}>
          <SectionHeader title="Pending Actions" subtitle="Requires your attention" />
          <View style={styles.card}>
            {PENDING_ACTIONS.map((item, index) => (
              <View key={item.id}>
                <PendingActionCard item={item} />
                {index < PENDING_ACTIONS.length - 1 && (
                  <View style={styles.divider} />
                )}
              </View>
            ))}
          </View>
        </View>

        {/* ── Recent Activity ── */}
        <View style={styles.section}>
          <SectionHeader title="Recent Activity" subtitle="Last 10 platform events" />
          <View style={styles.card}>
            {MOCK_ACTIVITY.map((item, index) => (
              <View key={item.id}>
                <ActivityCard item={item} />
                {index < MOCK_ACTIVITY.length - 1 && (
                  <View style={styles.divider} />
                )}
              </View>
            ))}
          </View>
        </View>

        <View style={styles.bottomPad} />
      </ScrollView>
    </SafeAreaView>
  );
}

// ─── Styles ──────────────────────────────────────────────────
const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  scroll: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  scrollContent: {
    paddingBottom: 20,
  },

  // Header
  header: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 24,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  logoBox: {
    width: 40,
    height: 40,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoText: {
    color: '#FFFFFF',
    fontSize: 22,
    fontWeight: '900',
  },
  logoLabel: {
    color: COLORS.text,
    fontSize: 17,
    fontWeight: '800',
    letterSpacing: -0.3,
  },
  logoSub: {
    color: COLORS.textMuted,
    fontSize: 11,
    fontWeight: '500',
  },
  adminAvatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: `${COLORS.primary}60`,
  },
  welcomeRow: {
    marginBottom: 6,
  },
  welcomeText: {
    color: COLORS.textMuted,
    fontSize: 13,
    fontWeight: '500',
  },
  welcomeName: {
    color: COLORS.text,
    fontSize: 22,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  headerDateRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  headerDate: {
    color: COLORS.textMuted,
    fontSize: 12,
  },

  // Sections
  section: {
    paddingHorizontal: 16,
    paddingTop: 20,
  },
  sectionHeader: {
    marginBottom: 12,
  },
  sectionTitle: {
    color: COLORS.text,
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: -0.2,
  },
  sectionSubtitle: {
    color: COLORS.textMuted,
    fontSize: 12,
    marginTop: 2,
  },

  // Stats grid
  statsGrid: {
    gap: 10,
  },
  statsRow: {
    flexDirection: 'row',
    gap: 10,
  },
  statCardHalf: {
    flex: 1,
  },

  // Card
  card: {
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: COLORS.border,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },

  // Snapshot
  snapshotCard: {
    paddingVertical: 20,
    paddingHorizontal: 12,
  },
  snapshotRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
  },
  snapshotItem: {
    alignItems: 'center',
    flex: 1,
    gap: 6,
  },
  snapshotIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  snapshotValue: {
    fontSize: 22,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  snapshotLabel: {
    color: COLORS.textMuted,
    fontSize: 11,
    fontWeight: '500',
    textAlign: 'center',
  },
  snapshotDivider: {
    width: 1,
    height: 60,
    backgroundColor: COLORS.border,
  },

  // Quick actions
  quickActionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  quickActionBtn: {
    width: (SCREEN_WIDTH - 52) / 2,
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  quickActionGradient: {
    padding: 16,
    alignItems: 'center',
    gap: 10,
  },
  quickActionIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  quickActionLabel: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: '600',
    textAlign: 'center',
  },

  // Pending
  pendingCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    gap: 12,
  },
  pendingIconWrap: {
    width: 40,
    height: 40,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pendingInfo: {
    flex: 1,
  },
  pendingLabel: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: '600',
  },
  pendingBadge: {
    borderRadius: 10,
    borderWidth: 1,
    paddingHorizontal: 10,
    paddingVertical: 3,
    marginRight: 4,
  },
  pendingCount: {
    fontSize: 13,
    fontWeight: '800',
  },

  // Activity
  activityItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    padding: 14,
    gap: 12,
  },
  activityIcon: {
    width: 36,
    height: 36,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 1,
  },
  activityInfo: {
    flex: 1,
  },
  activityMessage: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: '500',
    lineHeight: 18,
  },
  activityTime: {
    color: COLORS.textMuted,
    fontSize: 11,
    marginTop: 4,
  },

  divider: {
    height: 1,
    backgroundColor: COLORS.border,
    marginHorizontal: 14,
  },

  bottomPad: {
    height: 20,
  },
});
