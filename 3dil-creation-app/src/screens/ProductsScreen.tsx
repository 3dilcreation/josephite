import React, { useState, useMemo } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TextInput,
  TouchableOpacity, FlatList, Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import { Colors, Spacing, FontSize, BorderRadius, Shadows } from '../theme';
import { products } from '../data/products';
import ProductCard from '../components/ProductCard';
import { useCart } from '../context/CartContext';

const { width } = Dimensions.get('window');
type Nav = NativeStackNavigationProp<RootStackParamList>;

const CATEGORIES = ['All', 'Medals', 'Trophies', 'Statues', 'Architecture', 'Miniatures', 'Functional Parts'];
const SORT_OPTIONS = ['Default', 'Price: Low to High', 'Price: High to Low', 'Top Rated', 'Bestsellers'];

const ProductsScreen: React.FC = () => {
  const navigation = useNavigation<Nav>();
  const { addToCart, cartCount } = useCart();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [sortBy, setSortBy] = useState('Default');
  const [showSortModal, setShowSortModal] = useState(false);

  const filteredProducts = useMemo(() => {
    let result = [...products];

    if (selectedCategory !== 'All') {
      result = result.filter(p => p.category === selectedCategory);
    }

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(p =>
        p.name.toLowerCase().includes(query) ||
        p.description.toLowerCase().includes(query) ||
        p.category.toLowerCase().includes(query)
      );
    }

    switch (sortBy) {
      case 'Price: Low to High':
        result.sort((a, b) => a.price - b.price);
        break;
      case 'Price: High to Low':
        result.sort((a, b) => b.price - a.price);
        break;
      case 'Top Rated':
        result.sort((a, b) => b.rating - a.rating);
        break;
      case 'Bestsellers':
        result = result.filter(p => p.isBestseller).concat(result.filter(p => !p.isBestseller));
        break;
    }

    return result;
  }, [searchQuery, selectedCategory, sortBy]);

  return (
    <View style={styles.container}>
      {/* Header */}
      <LinearGradient colors={['#1A1A2E', '#16213E']} style={styles.header}>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>Products</Text>
          <TouchableOpacity style={styles.cartBtn} onPress={() => navigation.navigate('Cart')}>
            <Ionicons name="cart-outline" size={24} color={Colors.white} />
            {cartCount > 0 && (
              <View style={styles.cartBadge}>
                <Text style={styles.cartBadgeText}>{cartCount}</Text>
              </View>
            )}
          </TouchableOpacity>
        </View>

        {/* Search Bar */}
        <View style={styles.searchBar}>
          <Ionicons name="search-outline" size={18} color={Colors.textSecondary} />
          <TextInput
            style={styles.searchInput}
            placeholder="Search products..."
            placeholderTextColor={Colors.textSecondary}
            value={searchQuery}
            onChangeText={setSearchQuery}
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity onPress={() => setSearchQuery('')}>
              <Ionicons name="close-circle" size={18} color={Colors.textSecondary} />
            </TouchableOpacity>
          )}
        </View>
      </LinearGradient>

      {/* Category Filters + Sort */}
      <View style={styles.filterSection}>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.categoriesScroll}
        >
          {CATEGORIES.map(cat => (
            <TouchableOpacity
              key={cat}
              style={[styles.categoryChip, selectedCategory === cat && styles.categoryChipActive]}
              onPress={() => setSelectedCategory(cat)}
            >
              <Text style={[
                styles.categoryChipText,
                selectedCategory === cat && styles.categoryChipTextActive,
              ]}>{cat}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
        <TouchableOpacity style={styles.sortBtn} onPress={() => setShowSortModal(!showSortModal)}>
          <Ionicons name="funnel-outline" size={16} color={Colors.primary} />
          <Text style={styles.sortBtnText}>Sort</Text>
        </TouchableOpacity>
      </View>

      {/* Sort Modal */}
      {showSortModal && (
        <View style={[styles.sortModal, Shadows.large]}>
          {SORT_OPTIONS.map(option => (
            <TouchableOpacity
              key={option}
              style={[styles.sortOption, sortBy === option && styles.sortOptionActive]}
              onPress={() => { setSortBy(option); setShowSortModal(false); }}
            >
              <Text style={[styles.sortOptionText, sortBy === option && styles.sortOptionTextActive]}>
                {option}
              </Text>
              {sortBy === option && <Ionicons name="checkmark" size={16} color={Colors.primary} />}
            </TouchableOpacity>
          ))}
        </View>
      )}

      {/* Results count */}
      <View style={styles.resultsRow}>
        <Text style={styles.resultsText}>{filteredProducts.length} products found</Text>
      </View>

      {/* Products Grid */}
      {filteredProducts.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyEmoji}>🔍</Text>
          <Text style={styles.emptyTitle}>No products found</Text>
          <Text style={styles.emptyDesc}>Try a different search or category</Text>
          <TouchableOpacity
            style={styles.clearFilterBtn}
            onPress={() => {
              setSearchQuery('');
              setSelectedCategory('All');
            }}
          >
            <Text style={styles.clearFilterText}>Clear Filters</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={filteredProducts}
          numColumns={2}
          keyExtractor={item => item.id}
          renderItem={({ item }) => (
            <ProductCard
              product={item}
              onPress={() => navigation.navigate('ProductDetail', { productId: item.id })}
              onAddToCart={() => addToCart(item)}
            />
          )}
          contentContainerStyle={styles.productGrid}
          showsVerticalScrollIndicator={false}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    paddingTop: 52,
    paddingBottom: 16,
    paddingHorizontal: Spacing.md,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  headerTitle: {
    fontSize: FontSize.xxl,
    fontWeight: '900',
    color: Colors.white,
  },
  cartBtn: {
    padding: 4,
    position: 'relative',
  },
  cartBadge: {
    position: 'absolute',
    top: 0,
    right: 0,
    backgroundColor: Colors.primary,
    borderRadius: 8,
    width: 16,
    height: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cartBadgeText: {
    color: Colors.white,
    fontSize: 9,
    fontWeight: '700',
  },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.white,
    borderRadius: BorderRadius.md,
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 8,
  },
  searchInput: {
    flex: 1,
    fontSize: FontSize.md,
    color: Colors.textPrimary,
  },
  filterSection: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    backgroundColor: Colors.white,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  categoriesScroll: {
    paddingHorizontal: 12,
    gap: 8,
  },
  categoryChip: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: BorderRadius.round,
    backgroundColor: Colors.background,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  categoryChipActive: {
    backgroundColor: Colors.primary,
    borderColor: Colors.primary,
  },
  categoryChipText: {
    fontSize: FontSize.sm,
    fontWeight: '600',
    color: Colors.textSecondary,
  },
  categoryChipTextActive: {
    color: Colors.white,
  },
  sortBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginRight: 8,
    borderWidth: 1,
    borderColor: Colors.primary,
    borderRadius: BorderRadius.sm,
  },
  sortBtnText: {
    fontSize: FontSize.sm,
    color: Colors.primary,
    fontWeight: '600',
  },
  sortModal: {
    position: 'absolute',
    top: 160,
    right: 16,
    backgroundColor: Colors.white,
    borderRadius: BorderRadius.lg,
    padding: 8,
    zIndex: 100,
    width: 220,
  },
  sortOption: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderRadius: BorderRadius.sm,
  },
  sortOptionActive: {
    backgroundColor: '#FFF3EE',
  },
  sortOptionText: {
    fontSize: FontSize.sm,
    color: Colors.textPrimary,
  },
  sortOptionTextActive: {
    color: Colors.primary,
    fontWeight: '700',
  },
  resultsRow: {
    paddingHorizontal: Spacing.md,
    paddingVertical: 8,
  },
  resultsText: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
  },
  productGrid: {
    paddingHorizontal: 6,
    paddingBottom: 24,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
  },
  emptyEmoji: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyTitle: {
    fontSize: FontSize.xl,
    fontWeight: '700',
    color: Colors.textPrimary,
    marginBottom: 8,
  },
  emptyDesc: {
    fontSize: FontSize.md,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: 24,
  },
  clearFilterBtn: {
    backgroundColor: Colors.primary,
    borderRadius: BorderRadius.md,
    paddingVertical: 12,
    paddingHorizontal: 24,
  },
  clearFilterText: {
    color: Colors.white,
    fontWeight: '700',
    fontSize: FontSize.md,
  },
});

export default ProductsScreen;
