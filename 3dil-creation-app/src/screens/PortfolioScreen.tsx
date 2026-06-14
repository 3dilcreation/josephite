import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  Dimensions, Modal,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';
import { portfolio } from '../data/portfolio';
import { PortfolioItem } from '../types';

const { width } = Dimensions.get('window');
const CARD_WIDTH = (width - 48) / 2;

const categories = ['All', 'Medals', 'Trophies', 'Architecture', 'Statues', 'Miniatures', 'Custom'];

const PortfolioScreen: React.FC = () => {
  const navigation = useNavigation();
  const [activeCategory, setActiveCategory] = useState('All');
  const [selectedItem, setSelectedItem] = useState<PortfolioItem | null>(null);

  const filtered = activeCategory === 'All'
    ? portfolio
    : portfolio.filter(p => p.category === activeCategory);

  return (
    <View style={styles.container}>
      <LinearGradient colors={[Colors.secondary, Colors.primary]} style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color={Colors.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Our Portfolio</Text>
        <Text style={styles.headerSubtitle}>{portfolio.length}+ Projects Completed</Text>
      </LinearGradient>

      <View style={styles.filterContainer}>
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

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.grid}>
        {filtered.map((item, i) => (
          <TouchableOpacity
            key={item.id}
            style={[styles.card, i % 2 === 0 ? styles.cardLeft : styles.cardRight]}
            onPress={() => setSelectedItem(item)}
          >
            <View style={styles.cardImage}>
              <Text style={styles.cardEmoji}>{item.emoji}</Text>
            </View>
            <View style={styles.cardContent}>
              <Text style={styles.cardTitle} numberOfLines={2}>{item.title}</Text>
              <Text style={styles.cardCategory}>{item.category}</Text>
              <View style={styles.tagRow}>
                {item.tags.slice(0, 2).map((tag, j) => (
                  <View key={j} style={styles.tag}>
                    <Text style={styles.tagText}>{tag}</Text>
                  </View>
                ))}
              </View>
            </View>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {selectedItem && (
        <Modal visible transparent animationType="fade">
          <View style={styles.modalOverlay}>
            <View style={styles.modalCard}>
              <View style={styles.modalImageBox}>
                <Text style={styles.modalEmoji}>{selectedItem.emoji}</Text>
              </View>
              <ScrollView style={styles.modalBody}>
                <Text style={styles.modalTitle}>{selectedItem.title}</Text>
                <View style={styles.modalMeta}>
                  <View style={styles.categoryBadge}>
                    <Text style={styles.categoryBadgeText}>{selectedItem.category}</Text>
                  </View>
                  {selectedItem.completedDate && (
                    <Text style={styles.modalDate}>📅 {selectedItem.completedDate}</Text>
                  )}
                </View>
                {selectedItem.client && (
                  <Text style={styles.modalClient}>Client: {selectedItem.client}</Text>
                )}
                <Text style={styles.modalDescription}>{selectedItem.description}</Text>
                <View style={styles.tagsContainer}>
                  {selectedItem.tags.map((tag, i) => (
                    <View key={i} style={styles.tagLarge}>
                      <Text style={styles.tagLargeText}>{tag}</Text>
                    </View>
                  ))}
                </View>
              </ScrollView>
              <View style={styles.modalFooter}>
                <TouchableOpacity style={styles.modalOrderBtn} onPress={() => { setSelectedItem(null); navigation.navigate('CustomOrder' as never); }}>
                  <Text style={styles.modalOrderText}>Order Similar</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.modalCloseBtn} onPress={() => setSelectedItem(null)}>
                  <Ionicons name="close" size={24} color={Colors.textSecondary} />
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { paddingTop: 52, paddingBottom: 20, paddingHorizontal: Spacing.md },
  backBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center', marginBottom: 12 },
  headerTitle: { color: Colors.white, fontSize: FontSize.xxxl, fontWeight: '900', marginBottom: 4 },
  headerSubtitle: { color: 'rgba(255,255,255,0.75)', fontSize: FontSize.md },
  filterContainer: { paddingVertical: 12, paddingHorizontal: Spacing.md, backgroundColor: Colors.card, borderBottomWidth: 1, borderBottomColor: Colors.border },
  filterChip: { paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20, marginRight: 8, backgroundColor: Colors.background, borderWidth: 1.5, borderColor: Colors.border },
  filterChipActive: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  filterText: { fontSize: FontSize.sm, fontWeight: '700', color: Colors.textSecondary },
  filterTextActive: { color: Colors.white },
  grid: { flexDirection: 'row', flexWrap: 'wrap', padding: Spacing.md, gap: 12 },
  card: { width: CARD_WIDTH, backgroundColor: Colors.card, borderRadius: BorderRadius.md, overflow: 'hidden', ...Shadows.small },
  cardLeft: {},
  cardRight: {},
  cardImage: { height: 140, backgroundColor: Colors.background, alignItems: 'center', justifyContent: 'center' },
  cardEmoji: { fontSize: 64 },
  cardContent: { padding: 12 },
  cardTitle: { fontSize: FontSize.sm, fontWeight: '800', color: Colors.textPrimary, marginBottom: 4 },
  cardCategory: { fontSize: FontSize.xs, color: Colors.primary, fontWeight: '600', marginBottom: 6 },
  tagRow: { flexDirection: 'row', gap: 4 },
  tag: { backgroundColor: Colors.primary + '15', paddingHorizontal: 6, paddingVertical: 2, borderRadius: 6 },
  tagText: { fontSize: 9, color: Colors.primary, fontWeight: '700' },
  modalOverlay: { flex: 1, backgroundColor: Colors.overlay, justifyContent: 'flex-end' },
  modalCard: { backgroundColor: Colors.card, borderTopLeftRadius: 24, borderTopRightRadius: 24, maxHeight: '85%' },
  modalImageBox: { height: 200, backgroundColor: Colors.background, alignItems: 'center', justifyContent: 'center', borderTopLeftRadius: 24, borderTopRightRadius: 24 },
  modalEmoji: { fontSize: 100 },
  modalBody: { padding: 20, maxHeight: 280 },
  modalTitle: { fontSize: FontSize.xxl, fontWeight: '900', color: Colors.textPrimary, marginBottom: 10 },
  modalMeta: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 8 },
  categoryBadge: { backgroundColor: Colors.primary, paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12 },
  categoryBadgeText: { color: Colors.white, fontSize: FontSize.xs, fontWeight: '800' },
  modalDate: { fontSize: FontSize.sm, color: Colors.textSecondary },
  modalClient: { fontSize: FontSize.md, color: Colors.textSecondary, fontWeight: '600', marginBottom: 8 },
  modalDescription: { fontSize: FontSize.md, color: Colors.textSecondary, lineHeight: 22, marginBottom: 12 },
  tagsContainer: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  tagLarge: { backgroundColor: Colors.background, paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12, borderWidth: 1, borderColor: Colors.border },
  tagLargeText: { fontSize: FontSize.sm, color: Colors.textSecondary },
  modalFooter: { flexDirection: 'row', padding: 16, gap: 12, borderTopWidth: 1, borderTopColor: Colors.border, paddingBottom: 28 },
  modalOrderBtn: { flex: 1, backgroundColor: Colors.primary, borderRadius: BorderRadius.lg, paddingVertical: 16, alignItems: 'center' },
  modalOrderText: { color: Colors.white, fontWeight: '800', fontSize: FontSize.lg },
  modalCloseBtn: { width: 52, height: 52, borderRadius: BorderRadius.md, backgroundColor: Colors.background, alignItems: 'center', justifyContent: 'center' },
});

export default PortfolioScreen;
