// ============================================================
// TechSei LMS — Leaderboard Screen
// ============================================================
import React, { useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  Dimensions,
  Image,
  Animated,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '../../stores/authStore';
import type { LeaderboardEntry, LeaderboardPeriod } from '../../types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// ─── Colors ───────────────────────────────────────────────────────────────────
const C = {
  bg: '#0A0A1A',
  surface: '#141428',
  surfaceLight: '#1E1E3A',
  primary: '#6C63FF',
  accent: '#43E97B',
  gold: '#FFD700',
  silver: '#C0C0C0',
  bronze: '#CD7F32',
  warning: '#FFB84C',
  text: '#FFFFFF',
  textSec: '#9494B8',
  textMuted: '#5A5A7A',
  border: '#2A2A4A',
  error: '#FF4B4B',
};

// ─── Mock leaderboard data ────────────────────────────────────────────────────
function generateMockEntries(count: number): (LeaderboardEntry & { rankChange: number; isNew: boolean })[] {
  const names = [
    'Arjun Sharma', 'Priya Patel', 'Rohit Kumar', 'Sneha Gupta', 'Vikram Singh',
    'Ananya Reddy', 'Kiran Rao', 'Divya Nair', 'Aditya Joshi', 'Meera Iyer',
    'Rahul Verma', 'Pooja Mehta', 'Suresh Pillai', 'Anjali Mishra', 'Deepak Tiwari',
    'Kavita Saxena', 'Manish Agarwal', 'Ritu Bhatt', 'Sanjay Chauhan', 'Neha Desai',
    'Anonymous Learner', 'Anonymous Learner', 'You', 'Anonymous Learner', 'Anonymous Learner',
  ];

  return Array.from({ length: count }, (_, i) => ({
    rank: i + 1,
    user_id: i === 22 ? 'current-user' : `user-${i}`,
    name: names[i % names.length] || `Learner ${i + 1}`,
    avatar_url: null,
    xp: Math.max(50, 12000 - i * 450 + Math.floor(Math.random() * 200)),
    level: Math.max(1, 10 - Math.floor(i / 3)),
    streak: Math.max(0, 30 - i * 1 + Math.floor(Math.random() * 5)),
    rankChange: Math.floor(Math.random() * 7) - 3,
    isNew: i > 18 && Math.random() < 0.4,
  }));
}

const WEEKLY_ENTRIES = generateMockEntries(50);
const MONTHLY_ENTRIES = generateMockEntries(50).map((e, i) => ({
  ...e,
  xp: e.xp + i * 100,
  rankChange: Math.floor(Math.random() * 10) - 5,
}));
const ALLTIME_ENTRIES = generateMockEntries(50).map((e) => ({
  ...e,
  xp: e.xp * 3,
  rankChange: 0,
}));
const FRIENDS_ENTRIES = generateMockEntries(10);

type TabKey = 'weekly' | 'monthly' | 'alltime' | 'friends';

const TABS: { key: TabKey; label: string }[] = [
  { key: 'weekly', label: 'This Week' },
  { key: 'monthly', label: 'This Month' },
  { key: 'alltime', label: 'All Time' },
  { key: 'friends', label: 'Friends' },
];

function entriesForTab(tab: TabKey) {
  switch (tab) {
    case 'weekly': return WEEKLY_ENTRIES;
    case 'monthly': return MONTHLY_ENTRIES;
    case 'alltime': return ALLTIME_ENTRIES;
    case 'friends': return FRIENDS_ENTRIES;
  }
}

// ─── Avatar placeholder ───────────────────────────────────────────────────────
function Avatar({ name, size = 40, highlight = false }: { name: string; size?: number; highlight?: boolean }) {
  const initials = name
    .split(' ')
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() ?? '')
    .join('');

  const isAnon = name === 'Anonymous Learner';

  return (
    <LinearGradient
      colors={highlight ? [C.primary, '#8B5CF6'] : isAnon ? ['#2A2A4A', '#1E1E3A'] : ['#3A3A6A', '#2A2A4A']}
      style={[
        styles.avatar,
        { width: size, height: size, borderRadius: size / 2 },
        highlight && styles.avatarHighlight,
      ]}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      {isAnon ? (
        <Ionicons name="person" size={size * 0.45} color={C.textMuted} />
      ) : (
        <Text style={[styles.avatarText, { fontSize: size * 0.32 }]}>{initials}</Text>
      )}
    </LinearGradient>
  );
}

// ─── Rank Change badge ────────────────────────────────────────────────────────
function RankChange({ change, isNew }: { change: number; isNew: boolean }) {
  if (isNew) {
    return (
      <View style={[styles.rankChip, { backgroundColor: 'rgba(67,233,123,0.15)' }]}>
        <Text style={[styles.rankChipText, { color: C.accent }]}>NEW</Text>
      </View>
    );
  }
  if (change === 0) {
    return (
      <View style={[styles.rankChip, { backgroundColor: C.surfaceLight }]}>
        <Text style={[styles.rankChipText, { color: C.textMuted }]}>—</Text>
      </View>
    );
  }
  const up = change > 0;
  return (
    <View style={[styles.rankChip, { backgroundColor: up ? 'rgba(67,233,123,0.12)' : 'rgba(255,75,75,0.12)' }]}>
      <Ionicons name={up ? 'arrow-up' : 'arrow-down'} size={10} color={up ? C.accent : C.error} />
      <Text style={[styles.rankChipText, { color: up ? C.accent : C.error }]}>
        {Math.abs(change)}
      </Text>
    </View>
  );
}

// ─── Podium ───────────────────────────────────────────────────────────────────
function Podium({ entries }: { entries: (LeaderboardEntry & { rankChange: number; isNew: boolean })[] }) {
  const top3 = entries.slice(0, 3);
  const first = top3[0];
  const second = top3[1];
  const third = top3[2];

  if (!first) return null;

  return (
    <View style={styles.podiumContainer}>
      {/* 2nd place */}
      {second && (
        <View style={[styles.podiumSlot, styles.podiumSecond]}>
          <Avatar name={second.name} size={54} />
          <View style={styles.podiumCrown}>
            <Text style={{ fontSize: 16 }}>🥈</Text>
          </View>
          <Text style={styles.podiumName} numberOfLines={1}>{second.name.split(' ')[0]}</Text>
          <Text style={[styles.podiumXP, { color: C.silver }]}>{second.xp.toLocaleString()}</Text>
          <LinearGradient
            colors={[C.silver, '#A0A0A0']}
            style={[styles.podiumBase, { height: 60 }]}
          >
            <Text style={styles.podiumRank}>2</Text>
          </LinearGradient>
        </View>
      )}

      {/* 1st place */}
      <View style={[styles.podiumSlot, styles.podiumFirst]}>
        <View style={styles.crownWrapper}>
          <Text style={{ fontSize: 28 }}>👑</Text>
        </View>
        <Avatar name={first.name} size={72} highlight />
        <Text style={styles.podiumName} numberOfLines={1}>{first.name.split(' ')[0]}</Text>
        <Text style={[styles.podiumXP, { color: C.gold }]}>{first.xp.toLocaleString()}</Text>
        <LinearGradient
          colors={[C.gold, '#FFA500']}
          style={[styles.podiumBase, { height: 90 }]}
        >
          <Text style={styles.podiumRank}>1</Text>
        </LinearGradient>
      </View>

      {/* 3rd place */}
      {third && (
        <View style={[styles.podiumSlot, styles.podiumThird]}>
          <Avatar name={third.name} size={54} />
          <View style={styles.podiumCrown}>
            <Text style={{ fontSize: 16 }}>🥉</Text>
          </View>
          <Text style={styles.podiumName} numberOfLines={1}>{third.name.split(' ')[0]}</Text>
          <Text style={[styles.podiumXP, { color: C.bronze }]}>{third.xp.toLocaleString()}</Text>
          <LinearGradient
            colors={[C.bronze, '#A0522D']}
            style={[styles.podiumBase, { height: 44 }]}
          >
            <Text style={styles.podiumRank}>3</Text>
          </LinearGradient>
        </View>
      )}
    </View>
  );
}

// ─── List row ─────────────────────────────────────────────────────────────────
function LeaderRow({
  entry,
  isCurrentUser,
}: {
  entry: LeaderboardEntry & { rankChange: number; isNew: boolean };
  isCurrentUser: boolean;
}) {
  return (
    <View style={[styles.leaderRow, isCurrentUser && styles.leaderRowHighlight]}>
      <Text style={[styles.rowRank, isCurrentUser && { color: C.primary }]}>
        {entry.rank <= 3
          ? ['🥇', '🥈', '🥉'][entry.rank - 1]
          : `#${entry.rank}`}
      </Text>
      <Avatar name={entry.name} size={36} highlight={isCurrentUser} />
      <View style={styles.rowInfo}>
        <Text style={[styles.rowName, isCurrentUser && { color: C.primary }]} numberOfLines={1}>
          {isCurrentUser ? 'You' : entry.name}
        </Text>
        <View style={styles.rowMeta}>
          <View style={styles.levelChip}>
            <Text style={styles.levelChipText}>Lv {entry.level}</Text>
          </View>
          <Text style={styles.rowStreak}>🔥 {entry.streak}</Text>
        </View>
      </View>
      <View style={styles.rowRight}>
        <Text style={[styles.rowXP, isCurrentUser && { color: C.primary }]}>
          {entry.xp.toLocaleString()}
        </Text>
        <Text style={styles.rowXPLabel}>XP</Text>
        <RankChange change={entry.rankChange} isNew={entry.isNew} />
      </View>
    </View>
  );
}

// ─── Main Screen ──────────────────────────────────────────────────────────────
export default function LeaderboardScreen() {
  const [activeTab, setActiveTab] = useState<TabKey>('weekly');
  const [refreshing, setRefreshing] = useState(false);
  const { user } = useAuthStore();

  const entries = entriesForTab(activeTab);
  const listEntries = entries.slice(3); // skip podium top-3

  const currentUserEntry = entries.find((e) => e.user_id === 'current-user');
  const currentUserRank = currentUserEntry?.rank ?? 0;

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1200);
  }, []);

  const renderItem = useCallback(
    ({ item }: { item: typeof entries[0] }) => (
      <LeaderRow entry={item} isCurrentUser={item.user_id === 'current-user'} />
    ),
    []
  );

  const keyExtractor = useCallback((item: typeof entries[0]) => item.user_id, []);

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Leaderboard</Text>
        <View style={styles.headerIcon}>
          <Ionicons name="trophy" size={22} color={C.gold} />
        </View>
      </View>

      {/* Tab bar */}
      <View style={styles.tabBar}>
        <ScrollViewTabs activeTab={activeTab} onSelect={setActiveTab} />
      </View>

      {/* Your rank banner */}
      {currentUserRank > 0 && (
        <LinearGradient
          colors={['rgba(108,99,255,0.2)', 'rgba(108,99,255,0.05)']}
          style={styles.rankBanner}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
        >
          <Text style={styles.rankBannerText}>
            You're #{currentUserRank} {currentUserRank <= 10 ? '🏆' : currentUserRank <= 25 ? '💪' : '🎯'} Keep going!
          </Text>
        </LinearGradient>
      )}

      <FlatList
        data={listEntries}
        renderItem={renderItem}
        keyExtractor={keyExtractor}
        ListHeaderComponent={
          <>
            <Podium entries={entries} />
            {activeTab === 'friends' && entries.length === 0 && (
              <View style={styles.emptyFriends}>
                <Ionicons name="people-outline" size={48} color={C.textMuted} />
                <Text style={styles.emptyFriendsTitle}>No friends yet</Text>
                <Text style={styles.emptyFriendsText}>Invite friends to compete on the leaderboard!</Text>
              </View>
            )}
            <Text style={styles.listHeader}>All Rankings</Text>
          </>
        }
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={C.primary}
            colors={[C.primary]}
          />
        }
      />
    </SafeAreaView>
  );
}

// ─── Tab scrollview helper ────────────────────────────────────────────────────
function ScrollViewTabs({
  activeTab,
  onSelect,
}: {
  activeTab: TabKey;
  onSelect: (k: TabKey) => void;
}) {
  return (
    <View style={styles.tabsRow}>
      {TABS.map((tab) => {
        const active = activeTab === tab.key;
        return (
          <TouchableOpacity
            key={tab.key}
            onPress={() => onSelect(tab.key)}
            style={[styles.tab, active && styles.tabActive]}
          >
            <Text style={[styles.tabText, active && styles.tabTextActive]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: C.bg },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 8,
    paddingBottom: 12,
  },
  headerTitle: { fontSize: 26, fontWeight: '800', color: C.text },
  headerIcon: {
    width: 40, height: 40, borderRadius: 20,
    backgroundColor: 'rgba(255,215,0,0.12)',
    alignItems: 'center', justifyContent: 'center',
  },

  tabBar: { paddingHorizontal: 16, marginBottom: 8 },
  tabsRow: { flexDirection: 'row', gap: 8 },
  tab: {
    paddingHorizontal: 14, paddingVertical: 8,
    borderRadius: 20, backgroundColor: C.surfaceLight,
  },
  tabActive: { backgroundColor: C.primary },
  tabText: { fontSize: 13, fontWeight: '600', color: C.textSec },
  tabTextActive: { color: C.text },

  rankBanner: {
    marginHorizontal: 16,
    marginBottom: 12,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: 'rgba(108,99,255,0.3)',
  },
  rankBannerText: { color: C.text, fontWeight: '700', fontSize: 14 },

  listContent: { paddingHorizontal: 16, paddingBottom: 32 },
  listHeader: {
    fontSize: 14, fontWeight: '700', color: C.textSec,
    marginBottom: 8, marginTop: 8,
  },

  // Podium
  podiumContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'center',
    marginBottom: 24,
    gap: 8,
  },
  podiumSlot: { alignItems: 'center', position: 'relative' },
  podiumFirst: { zIndex: 2 },
  podiumSecond: {},
  podiumThird: {},
  crownWrapper: { marginBottom: 4 },
  podiumCrown: { position: 'absolute', top: 0, right: 0 },
  podiumName: {
    color: C.text, fontSize: 12, fontWeight: '700',
    marginTop: 6, marginBottom: 2,
    maxWidth: 80, textAlign: 'center',
  },
  podiumXP: { fontSize: 11, fontWeight: '700', marginBottom: 4 },
  podiumBase: {
    width: 70, borderTopLeftRadius: 8, borderTopRightRadius: 8,
    alignItems: 'center', justifyContent: 'center',
  },
  podiumRank: { color: '#0A0A1A', fontSize: 18, fontWeight: '900' },

  // Avatar
  avatar: { alignItems: 'center', justifyContent: 'center' },
  avatarHighlight: {
    borderWidth: 2,
    borderColor: C.primary,
    shadowColor: C.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.7,
    shadowRadius: 8,
    elevation: 8,
  },
  avatarText: { color: C.text, fontWeight: '800' },

  // Row
  leaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.surface,
    borderRadius: 14,
    padding: 12,
    marginBottom: 8,
    gap: 12,
    borderWidth: 1,
    borderColor: C.border,
  },
  leaderRowHighlight: {
    borderColor: C.primary,
    backgroundColor: 'rgba(108,99,255,0.1)',
  },
  rowRank: { width: 32, fontSize: 14, fontWeight: '800', color: C.textSec, textAlign: 'center' },
  rowInfo: { flex: 1 },
  rowName: { color: C.text, fontSize: 14, fontWeight: '700', marginBottom: 2 },
  rowMeta: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  levelChip: {
    backgroundColor: C.surfaceLight,
    paddingHorizontal: 6, paddingVertical: 2,
    borderRadius: 6,
  },
  levelChipText: { color: C.textSec, fontSize: 10, fontWeight: '700' },
  rowStreak: { fontSize: 11, color: C.textMuted },
  rowRight: { alignItems: 'flex-end', gap: 2 },
  rowXP: { fontSize: 15, fontWeight: '800', color: C.text },
  rowXPLabel: { fontSize: 9, color: C.textMuted, fontWeight: '600' },

  // Rank chip
  rankChip: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: 5, paddingVertical: 2,
    borderRadius: 6, gap: 2,
  },
  rankChipText: { fontSize: 10, fontWeight: '700' },

  // Friends empty
  emptyFriends: {
    alignItems: 'center',
    paddingVertical: 40,
    gap: 10,
  },
  emptyFriendsTitle: { fontSize: 18, fontWeight: '700', color: C.text },
  emptyFriendsText: { fontSize: 13, color: C.textMuted, textAlign: 'center' },
});
