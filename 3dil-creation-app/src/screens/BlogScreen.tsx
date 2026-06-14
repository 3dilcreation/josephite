import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, TextInput,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';
import { blogPosts } from '../data/blog';

const categories = ['All', '3D Printing', 'Materials', 'Tips & Tricks', 'Industry', 'Projects'];

const BlogScreen: React.FC = () => {
  const navigation = useNavigation();
  const [activeCategory, setActiveCategory] = useState('All');
  const [search, setSearch] = useState('');

  const filtered = blogPosts.filter(p => {
    const matchCat = activeCategory === 'All' || p.category === activeCategory;
    const matchSearch = p.title.toLowerCase().includes(search.toLowerCase()) || p.excerpt.toLowerCase().includes(search.toLowerCase());
    return matchCat && matchSearch;
  });

  const featured = filtered[0];
  const rest = filtered.slice(1);

  return (
    <View style={styles.container}>
      <LinearGradient colors={[Colors.primary, '#FF8C42']} style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color={Colors.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>3D Printing Blog</Text>
        <Text style={styles.headerSubtitle}>Tips, news & inspiration</Text>
        <View style={styles.searchBox}>
          <Ionicons name="search" size={18} color={Colors.textLight} />
          <TextInput
            style={styles.searchInput}
            placeholder="Search articles..."
            value={search}
            onChangeText={setSearch}
            placeholderTextColor={Colors.textLight}
          />
        </View>
      </LinearGradient>

      <View style={styles.filterRow}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {categories.map(cat => (
            <TouchableOpacity
              key={cat}
              style={[styles.filterChip, activeCategory === cat && styles.filterChipActive]}
              onPress={() => setActiveCategory(cat)}
            >
              <Text style={[styles.filterText, activeCategory === cat && styles.filterTextActive]}>{cat}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: Spacing.md }}>
        {featured && (
          <TouchableOpacity style={styles.featuredCard}>
            <LinearGradient colors={[Colors.secondary, '#0F3460']} style={styles.featuredGradient}>
              <Text style={styles.featuredEmoji}>{featured.emoji}</Text>
              <View style={styles.featuredBadge}>
                <Text style={styles.featuredBadgeText}>FEATURED</Text>
              </View>
              <Text style={styles.featuredTitle}>{featured.title}</Text>
              <Text style={styles.featuredExcerpt} numberOfLines={2}>{featured.excerpt}</Text>
              <View style={styles.featuredMeta}>
                <Text style={styles.featuredAuthor}>By {featured.author}</Text>
                <Text style={styles.featuredTime}>⏱ {featured.readTime} min read</Text>
              </View>
            </LinearGradient>
          </TouchableOpacity>
        )}

        {rest.map(post => (
          <TouchableOpacity key={post.id} style={styles.postCard}>
            <View style={styles.postImageBox}>
              <Text style={styles.postEmoji}>{post.emoji}</Text>
            </View>
            <View style={styles.postContent}>
              <View style={styles.postCatRow}>
                <View style={styles.catBadge}>
                  <Text style={styles.catText}>{post.category}</Text>
                </View>
                <Text style={styles.postDate}>{post.date}</Text>
              </View>
              <Text style={styles.postTitle} numberOfLines={2}>{post.title}</Text>
              <Text style={styles.postExcerpt} numberOfLines={2}>{post.excerpt}</Text>
              <View style={styles.postFooter}>
                <Text style={styles.postAuthor}>By {post.author}</Text>
                <Text style={styles.postRead}>⏱ {post.readTime} min</Text>
              </View>
            </View>
          </TouchableOpacity>
        ))}

        {filtered.length === 0 && (
          <View style={styles.empty}>
            <Text style={styles.emptyEmoji}>📝</Text>
            <Text style={styles.emptyText}>No articles found</Text>
          </View>
        )}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { paddingTop: 52, paddingBottom: 16, paddingHorizontal: Spacing.md },
  backBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center', marginBottom: 8 },
  headerTitle: { color: Colors.white, fontSize: FontSize.xxl, fontWeight: '900', marginBottom: 2 },
  headerSubtitle: { color: 'rgba(255,255,255,0.75)', fontSize: FontSize.md, marginBottom: 12 },
  searchBox: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: Colors.white, borderRadius: BorderRadius.md, paddingHorizontal: 14, paddingVertical: 10 },
  searchInput: { flex: 1, fontSize: FontSize.md, color: Colors.textPrimary },
  filterRow: { paddingVertical: 12, paddingHorizontal: Spacing.md, backgroundColor: Colors.card, borderBottomWidth: 1, borderBottomColor: Colors.border },
  filterChip: { paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20, marginRight: 8, backgroundColor: Colors.background, borderWidth: 1.5, borderColor: Colors.border },
  filterChipActive: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  filterText: { fontSize: FontSize.sm, fontWeight: '700', color: Colors.textSecondary },
  filterTextActive: { color: Colors.white },
  featuredCard: { borderRadius: BorderRadius.lg, overflow: 'hidden', marginBottom: 16, ...Shadows.medium },
  featuredGradient: { padding: 24, minHeight: 200, justifyContent: 'flex-end' },
  featuredEmoji: { position: 'absolute', top: 20, right: 20, fontSize: 60 },
  featuredBadge: { backgroundColor: Colors.accent, alignSelf: 'flex-start', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8, marginBottom: 10 },
  featuredBadgeText: { color: Colors.secondary, fontSize: 10, fontWeight: '900', letterSpacing: 1 },
  featuredTitle: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '900', marginBottom: 8 },
  featuredExcerpt: { color: 'rgba(255,255,255,0.8)', fontSize: FontSize.sm, lineHeight: 20, marginBottom: 12 },
  featuredMeta: { flexDirection: 'row', justifyContent: 'space-between' },
  featuredAuthor: { color: 'rgba(255,255,255,0.7)', fontSize: FontSize.xs },
  featuredTime: { color: 'rgba(255,255,255,0.7)', fontSize: FontSize.xs },
  postCard: { flexDirection: 'row', backgroundColor: Colors.card, borderRadius: BorderRadius.md, marginBottom: 12, overflow: 'hidden', ...Shadows.small },
  postImageBox: { width: 90, backgroundColor: Colors.background, alignItems: 'center', justifyContent: 'center' },
  postEmoji: { fontSize: 40 },
  postContent: { flex: 1, padding: 12 },
  postCatRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
  catBadge: { backgroundColor: Colors.primary + '20', paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8 },
  catText: { fontSize: 9, color: Colors.primary, fontWeight: '800' },
  postDate: { fontSize: FontSize.xs, color: Colors.textLight },
  postTitle: { fontSize: FontSize.md, fontWeight: '800', color: Colors.textPrimary, marginBottom: 4 },
  postExcerpt: { fontSize: FontSize.xs, color: Colors.textSecondary, lineHeight: 18, marginBottom: 8 },
  postFooter: { flexDirection: 'row', justifyContent: 'space-between' },
  postAuthor: { fontSize: FontSize.xs, color: Colors.textSecondary },
  postRead: { fontSize: FontSize.xs, color: Colors.primary, fontWeight: '600' },
  empty: { alignItems: 'center', padding: 40 },
  emptyEmoji: { fontSize: 48, marginBottom: 8 },
  emptyText: { fontSize: FontSize.lg, color: Colors.textSecondary },
});

export default BlogScreen;
