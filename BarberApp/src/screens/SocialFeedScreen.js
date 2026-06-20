import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  Dimensions, Image,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, Spacing, Radius } from '../theme/colors';
import { FEED_POSTS, HAIRSTYLES } from '../data/mockData';

const { width } = Dimensions.get('window');

const TRENDING = HAIRSTYLES.filter(s => s.trending).slice(0, 4);

const POST_COLORS = [
  ['#1A1200', '#0F0A00'],
  ['#0A001A', '#05001A'],
  ['#001A0A', '#000F05'],
  ['#1A000A', '#0F0005'],
];

const POST_BG_EMOJIS = ['✂️', '💈', '🔥', '⚡'];
const POST_SIZES = ['Low', 'Mid', 'High', 'Skin'];

function FeedPost({ post, index, onLike }) {
  const colors = POST_COLORS[index % POST_COLORS.length];
  const bgEmoji = POST_BG_EMOJIS[index % POST_BG_EMOJIS.length];

  return (
    <View style={styles.post}>
      {/* Post Image Area */}
      <LinearGradient colors={colors} style={styles.postImage}>
        <View style={styles.postImageContent}>
          <Text style={styles.postBgEmoji}>{bgEmoji}</Text>
          <Text style={styles.postStyleDisplay}>{post.style}</Text>
          <Text style={styles.postStyleSub}>by {post.barber.split(' ')[0]}</Text>
        </View>

        {/* Trending Tag */}
        {index < 2 && (
          <View style={styles.trendingTag}>
            <MaterialCommunityIcons name="fire" size={12} color="#E05C5C" />
            <Text style={styles.trendingTagText}>Trending</Text>
          </View>
        )}

        {/* AR Try-On Button */}
        <TouchableOpacity style={styles.tryOnBtn}>
          <MaterialCommunityIcons name="augmented-reality" size={14} color="#6C5CE7" />
          <Text style={styles.tryOnText}>Try On</Text>
        </TouchableOpacity>
      </LinearGradient>

      {/* Post Content */}
      <View style={styles.postContent}>
        {/* Barber Info */}
        <View style={styles.postHeader}>
          <Image source={{ uri: post.avatar }} style={styles.postAvatar} />
          <View style={styles.postMeta}>
            <Text style={styles.postBarberName}>{post.barber}</Text>
            <Text style={styles.postTime}>{post.timeAgo}</Text>
          </View>
          <TouchableOpacity style={styles.followBtn}>
            <Text style={styles.followText}>Follow</Text>
          </TouchableOpacity>
        </View>

        {/* Tags */}
        <View style={styles.postTags}>
          {post.tags.map((tag) => (
            <Text key={tag} style={styles.postTag}>{tag}</Text>
          ))}
        </View>

        {/* Actions */}
        <View style={styles.postActions}>
          <TouchableOpacity style={styles.actionBtn} onPress={() => onLike(post.id)}>
            <MaterialCommunityIcons
              name={post.liked ? 'heart' : 'heart-outline'}
              size={22}
              color={post.liked ? Colors.accent : Colors.textSecondary}
            />
            <Text style={[styles.actionCount, post.liked && { color: Colors.accent }]}>
              {post.likes}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.actionBtn}>
            <MaterialCommunityIcons name="comment-outline" size={22} color={Colors.textSecondary} />
            <Text style={styles.actionCount}>{post.comments}</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.actionBtn}>
            <MaterialCommunityIcons name="bookmark-outline" size={22} color={Colors.textSecondary} />
            <Text style={styles.actionCount}>{post.saves}</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.actionBtn}>
            <MaterialCommunityIcons name="share-variant-outline" size={22} color={Colors.textSecondary} />
          </TouchableOpacity>

          <TouchableOpacity style={styles.bookItBtn}>
            <LinearGradient colors={Colors.gradientGold} style={styles.bookItGrad} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}>
              <Text style={styles.bookItText}>Book It</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

export default function SocialFeedScreen() {
  const insets = useSafeAreaInsets();
  const [posts, setPosts] = useState(FEED_POSTS);
  const [activeFilter, setActiveFilter] = useState('For You');
  const FILTERS = ['For You', 'Trending', 'Fades', 'Beards', 'Designs'];

  const toggleLike = (id) => {
    setPosts(posts.map(p => p.id === id ? { ...p, liked: !p.liked, likes: p.liked ? p.likes - 1 : p.likes + 1 } : p));
  };

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Style Feed</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity style={styles.headerBtn}>
            <MaterialCommunityIcons name="magnify" size={22} color={Colors.textPrimary} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.headerBtn}>
            <MaterialCommunityIcons name="camera-plus-outline" size={22} color={Colors.textPrimary} />
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Trending Styles Highlight */}
        <View style={styles.trendingSection}>
          <View style={styles.trendingHeader}>
            <MaterialCommunityIcons name="fire" size={18} color="#E05C5C" />
            <Text style={styles.trendingTitle}>Trending This Week</Text>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.trendingScroll}>
            {TRENDING.map((style) => (
              <TouchableOpacity key={style.id} style={styles.trendCard} activeOpacity={0.85}>
                <LinearGradient
                  colors={[style.color + '30', style.color + '10']}
                  style={styles.trendCardGrad}
                >
                  <Text style={styles.trendEmoji}>{style.emoji}</Text>
                  <Text style={styles.trendName}>{style.name}</Text>
                  <Text style={styles.trendCat}>{style.category}</Text>
                </LinearGradient>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        {/* Barber Battle Banner */}
        <TouchableOpacity style={styles.battleBanner} activeOpacity={0.9}>
          <LinearGradient
            colors={['#2D0030', '#1A001F']}
            style={StyleSheet.absoluteFill}
            borderRadius={Radius.lg}
          />
          <View style={styles.battleLeft}>
            <View style={styles.battleBadge}>
              <Text style={styles.battleBadgeText}>⚔️ LIVE</Text>
            </View>
            <Text style={styles.battleTitle}>Barber Battle{'\n'}This Weekend!</Text>
            <Text style={styles.battleSub}>Vote for the best cut & win free sessions</Text>
          </View>
          <MaterialCommunityIcons name="trophy" size={60} color="#FFD70030" style={styles.trophyIcon} />
        </TouchableOpacity>

        {/* Filter Tabs */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filtersScroll}>
          {FILTERS.map((f) => (
            <TouchableOpacity
              key={f}
              style={[styles.filterChip, activeFilter === f && styles.filterChipActive]}
              onPress={() => setActiveFilter(f)}
              activeOpacity={0.8}
            >
              {activeFilter === f && (
                <LinearGradient colors={Colors.gradientGold} style={StyleSheet.absoluteFill} borderRadius={Radius.full} />
              )}
              <Text style={[styles.filterText, activeFilter === f && { color: '#0A0A0F' }]}>{f}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Posts */}
        {posts.map((post, i) => (
          <FeedPost key={post.id} post={post} index={i} onLike={toggleLike} />
        ))}

        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.md,
  },
  headerTitle: { color: Colors.textPrimary, fontSize: 24, fontWeight: '800' },
  headerActions: { flexDirection: 'row', gap: 8 },
  headerBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.bgGlass,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    alignItems: 'center',
    justifyContent: 'center',
  },
  trendingSection: { paddingHorizontal: Spacing.md, marginBottom: Spacing.md },
  trendingHeader: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 12 },
  trendingTitle: { color: Colors.textPrimary, fontSize: 16, fontWeight: '700' },
  trendingScroll: { marginHorizontal: -Spacing.md, paddingHorizontal: Spacing.md },
  trendCard: { marginRight: 10 },
  trendCardGrad: {
    padding: Spacing.md,
    borderRadius: Radius.md,
    alignItems: 'center',
    minWidth: 90,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
  },
  trendEmoji: { fontSize: 28, marginBottom: 6 },
  trendName: { color: Colors.textPrimary, fontSize: 12, fontWeight: '700', textAlign: 'center', marginBottom: 2 },
  trendCat: { color: Colors.textMuted, fontSize: 10 },
  battleBanner: {
    marginHorizontal: Spacing.md,
    marginBottom: Spacing.md,
    borderRadius: Radius.lg,
    padding: Spacing.lg,
    overflow: 'hidden',
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#9B30FF30',
  },
  battleLeft: { flex: 1 },
  battleBadge: {
    backgroundColor: '#E05C5C30',
    borderRadius: Radius.full,
    paddingHorizontal: 10,
    paddingVertical: 4,
    alignSelf: 'flex-start',
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#E05C5C60',
  },
  battleBadgeText: { color: '#E05C5C', fontSize: 11, fontWeight: '800', letterSpacing: 1 },
  battleTitle: { color: Colors.textPrimary, fontSize: 20, fontWeight: '800', lineHeight: 26, marginBottom: 6 },
  battleSub: { color: Colors.textSecondary, fontSize: 12 },
  trophyIcon: { position: 'absolute', right: 16, bottom: 8 },
  filtersScroll: { marginHorizontal: Spacing.md, marginBottom: Spacing.md },
  filterChip: {
    paddingHorizontal: 18,
    paddingVertical: 9,
    borderRadius: Radius.full,
    backgroundColor: Colors.bgCardAlt,
    marginRight: 8,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
  },
  filterChipActive: { borderColor: Colors.primary },
  filterText: { color: Colors.textSecondary, fontSize: 13, fontWeight: '600' },
  post: {
    marginBottom: Spacing.xl,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderSubtle,
    paddingBottom: Spacing.lg,
  },
  postImage: {
    height: 300,
    marginHorizontal: Spacing.md,
    borderRadius: Radius.lg,
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    marginBottom: Spacing.md,
  },
  postImageContent: { alignItems: 'center' },
  postBgEmoji: { fontSize: 72, marginBottom: 12 },
  postStyleDisplay: { color: Colors.textPrimary, fontSize: 24, fontWeight: '800', marginBottom: 4 },
  postStyleSub: { color: Colors.textSecondary, fontSize: 14 },
  trendingTag: {
    position: 'absolute',
    top: 12,
    left: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#E05C5C20',
    borderRadius: Radius.full,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderWidth: 1,
    borderColor: '#E05C5C40',
  },
  trendingTagText: { color: '#E05C5C', fontSize: 11, fontWeight: '700' },
  tryOnBtn: {
    position: 'absolute',
    bottom: 12,
    right: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#6C5CE720',
    borderRadius: Radius.full,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: '#6C5CE740',
  },
  tryOnText: { color: '#6C5CE7', fontSize: 12, fontWeight: '700' },
  postContent: { paddingHorizontal: Spacing.md },
  postHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 10 },
  postAvatar: { width: 40, height: 40, borderRadius: 20, borderWidth: 2, borderColor: Colors.border },
  postMeta: { flex: 1 },
  postBarberName: { color: Colors.textPrimary, fontSize: 14, fontWeight: '700' },
  postTime: { color: Colors.textMuted, fontSize: 12 },
  followBtn: {
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: Radius.full,
    borderWidth: 1,
    borderColor: Colors.primary,
  },
  followText: { color: Colors.primary, fontSize: 12, fontWeight: '700' },
  postTags: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginBottom: 12 },
  postTag: { color: Colors.accentCool, fontSize: 12 },
  postActions: { flexDirection: 'row', alignItems: 'center', gap: 16 },
  actionBtn: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  actionCount: { color: Colors.textSecondary, fontSize: 13, fontWeight: '600' },
  bookItBtn: { marginLeft: 'auto', borderRadius: Radius.full, overflow: 'hidden' },
  bookItGrad: { paddingHorizontal: 18, paddingVertical: 8 },
  bookItText: { color: '#0A0A0F', fontSize: 12, fontWeight: '800' },
});
