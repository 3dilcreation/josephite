import React, { useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Animated,
  Dimensions,
  StatusBar,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';
import { services } from '../data/services';
import { products } from '../data/products';
import { testimonials } from '../data/testimonials';
import { useCart } from '../context/CartContext';
import { useAuth } from '../context/AuthContext';

const { width } = Dimensions.get('window');

type Nav = NativeStackNavigationProp<RootStackParamList>;

const HomeScreen: React.FC = () => {
  const navigation = useNavigation<Nav>();
  const { cartCount } = useCart();
  const { user } = useAuth();
  const scrollY = useRef(new Animated.Value(0)).current;

  const statsData = [
    { label: 'Orders', value: '500+', icon: '📦' },
    { label: 'Clients', value: '200+', icon: '😊' },
    { label: 'Rating', value: '4.9★', icon: '⭐' },
    { label: 'Delivery', value: '48hr', icon: '🚚' },
  ];

  const quickActions = [
    { label: 'Custom Order', icon: 'construct', screen: 'CustomOrder' as const, color: Colors.primary },
    { label: 'Get Quote', icon: 'calculator', screen: 'QuoteCalculator' as const, color: '#7C3AED' },
    { label: 'AR Preview', icon: 'camera', screen: 'ARViewer' as const, color: '#0EA5E9' },
    { label: 'Portfolio', icon: 'images', screen: 'Portfolio' as const, color: '#10B981' },
  ];

  const whyChooseUs = [
    { icon: '🏭', title: 'Industrial Grade', desc: 'Professional FDM & SLA printers' },
    { icon: '⚡', title: 'Fast Turnaround', desc: 'Delivery in 24–48 hours' },
    { icon: '💎', title: 'Premium Quality', desc: '100% quality guarantee' },
    { icon: '🇮🇳', title: 'Made in India', desc: 'Supporting local innovation' },
  ];

  const featuredProducts = products.filter(p => p.isBestseller).slice(0, 4);

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={Colors.primary} />

      {/* Header */}
      <LinearGradient
        colors={[Colors.primary, Colors.secondary]}
        style={styles.header}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        <View style={styles.headerRow}>
          <View>
            <Text style={styles.headerGreeting}>
              {user ? `Hello, ${user.name.split(' ')[0]}! 👋` : 'Welcome! 👋'}
            </Text>
            <Text style={styles.headerBrand}>3DIL CREATION</Text>
          </View>
          <View style={styles.headerIcons}>
            <TouchableOpacity
              style={styles.headerIcon}
              onPress={() => navigation.navigate('Cart')}
            >
              <Ionicons name="cart-outline" size={24} color={Colors.white} />
              {cartCount > 0 && (
                <View style={styles.cartBadge}>
                  <Text style={styles.cartBadgeText}>{cartCount}</Text>
                </View>
              )}
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.headerIcon}
              onPress={() => navigation.navigate('Contact')}
            >
              <Ionicons name="notifications-outline" size={24} color={Colors.white} />
            </TouchableOpacity>
          </View>
        </View>
      </LinearGradient>

      <ScrollView
        showsVerticalScrollIndicator={false}
        onScroll={Animated.event([{ nativeEvent: { contentOffset: { y: scrollY } } }], { useNativeDriver: false })}
        scrollEventThrottle={16}
      >
        {/* Hero Banner */}
        <LinearGradient
          colors={[Colors.primary, Colors.secondary]}
          style={styles.hero}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        >
          <Text style={styles.heroEmoji}>🖨️</Text>
          <Text style={styles.heroTitle}>Bring Your 3D{'\n'}Dreams to Life</Text>
          <Text style={styles.heroSubtitle}>
            Custom 3D printing for medals, trophies, architectural models, statues & more
          </Text>
          <View style={styles.heroButtons}>
            <TouchableOpacity
              style={styles.heroBtn}
              onPress={() => navigation.navigate('CustomOrder', {})}
            >
              <Text style={styles.heroBtnText}>Order Now →</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.heroBtnOutline}
              onPress={() => navigation.navigate('QuoteCalculator')}
            >
              <Text style={styles.heroBtnOutlineText}>Get Quote</Text>
            </TouchableOpacity>
          </View>
        </LinearGradient>

        {/* Stats */}
        <View style={styles.statsRow}>
          {statsData.map((stat, i) => (
            <View key={i} style={styles.statCard}>
              <Text style={styles.statEmoji}>{stat.icon}</Text>
              <Text style={styles.statValue}>{stat.value}</Text>
              <Text style={styles.statLabel}>{stat.label}</Text>
            </View>
          ))}
        </View>

        {/* Quick Actions */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Quick Actions</Text>
          <View style={styles.quickActionsGrid}>
            {quickActions.map((action, i) => (
              <TouchableOpacity
                key={i}
                style={[styles.quickActionCard, { borderTopColor: action.color, borderTopWidth: 3 }]}
                onPress={() =>
                  action.screen === 'ARViewer' || action.screen === 'Portfolio'
                    ? navigation.navigate(action.screen)
                    : navigation.navigate(action.screen as any, {} as any)
                }
              >
                <View style={[styles.quickActionIcon, { backgroundColor: action.color + '20' }]}>
                  <Ionicons name={action.icon as any} size={28} color={action.color} />
                </View>
                <Text style={styles.quickActionLabel}>{action.label}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Services */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Our Services</Text>
            <TouchableOpacity onPress={() => navigation.navigate('Portfolio')}>
              <Text style={styles.seeAll}>See All →</Text>
            </TouchableOpacity>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            {services.map(service => (
              <TouchableOpacity
                key={service.id}
                style={styles.serviceCard}
                onPress={() => navigation.navigate('ServiceDetail', { serviceId: service.id })}
              >
                <Text style={styles.serviceEmoji}>{service.icon}</Text>
                <Text style={styles.serviceName}>{service.name}</Text>
                <Text style={styles.servicePrice}>From ₹{service.startingPrice}</Text>
                <Text style={styles.serviceTurnaround}>{service.turnaround}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        {/* Quote CTA */}
        <TouchableOpacity
          style={styles.quoteBanner}
          onPress={() => navigation.navigate('QuoteCalculator')}
        >
          <LinearGradient
            colors={['#7C3AED', '#6D28D9']}
            style={styles.quoteBannerGradient}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <View style={styles.quoteBannerContent}>
              <Text style={styles.quoteBannerEmoji}>🤖</Text>
              <View style={{ flex: 1 }}>
                <Text style={styles.quoteBannerTitle}>AI Quote Calculator</Text>
                <Text style={styles.quoteBannerDesc}>
                  Get instant price estimate by material, size & complexity
                </Text>
              </View>
              <Ionicons name="arrow-forward" size={24} color={Colors.white} />
            </View>
          </LinearGradient>
        </TouchableOpacity>

        {/* Featured Products */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Bestsellers</Text>
            <TouchableOpacity>
              <Text style={styles.seeAll}>View All →</Text>
            </TouchableOpacity>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            {featuredProducts.map(product => (
              <TouchableOpacity
                key={product.id}
                style={styles.productCard}
                onPress={() => navigation.navigate('ProductDetail', { productId: product.id })}
              >
                <View style={styles.productImageBox}>
                  <Text style={styles.productEmoji}>{product.emoji}</Text>
                  <View style={styles.bestsellerBadge}>
                    <Text style={styles.bestsellerText}>⭐ Best</Text>
                  </View>
                </View>
                <Text style={styles.productName} numberOfLines={2}>{product.name}</Text>
                <View style={styles.productPriceRow}>
                  <Text style={styles.productPrice}>₹{product.price.toLocaleString()}</Text>
                  <Text style={styles.productOriginal}>₹{product.originalPrice.toLocaleString()}</Text>
                </View>
                <View style={styles.productRating}>
                  <Ionicons name="star" size={12} color={Colors.accent} />
                  <Text style={styles.productRatingText}>{product.rating} ({product.reviewCount})</Text>
                </View>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        {/* Why Choose Us */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Why Choose 3DIL?</Text>
          <View style={styles.whyGrid}>
            {whyChooseUs.map((item, i) => (
              <View key={i} style={styles.whyCard}>
                <Text style={styles.whyEmoji}>{item.icon}</Text>
                <Text style={styles.whyTitle}>{item.title}</Text>
                <Text style={styles.whyDesc}>{item.desc}</Text>
              </View>
            ))}
          </View>
        </View>

        {/* Subscription Teaser */}
        <TouchableOpacity
          style={styles.subscriptionBanner}
          onPress={() => navigation.navigate('Subscription')}
        >
          <LinearGradient
            colors={['#FFD700', '#FFA500']}
            style={styles.subGradient}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <View style={styles.subContent}>
              <View>
                <Text style={styles.subTitle}>🌟 3DIL Pro Plans</Text>
                <Text style={styles.subDesc}>Save up to 30% + free prints monthly</Text>
                <Text style={styles.subPrice}>Starting at ₹499/month</Text>
              </View>
              <Ionicons name="chevron-forward" size={28} color={Colors.secondary} />
            </View>
          </LinearGradient>
        </TouchableOpacity>

        {/* Testimonials */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>What Our Clients Say</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            {testimonials.slice(0, 4).map(t => (
              <View key={t.id} style={styles.testimonialCard}>
                <View style={styles.testimonialHeader}>
                  <View style={styles.testimonialAvatar}>
                    <Text style={styles.testimonialAvatarText}>{t.avatar}</Text>
                  </View>
                  <View>
                    <Text style={styles.testimonialName}>{t.name}</Text>
                    <Text style={styles.testimonialLocation}>📍 {t.location}</Text>
                  </View>
                </View>
                <View style={styles.starsRow}>
                  {Array(t.rating).fill(0).map((_, i) => (
                    <Ionicons key={i} name="star" size={14} color={Colors.accent} />
                  ))}
                </View>
                <Text style={styles.testimonialReview} numberOfLines={3}>
                  "{t.review}"
                </Text>
                <Text style={styles.testimonialService}>{t.service}</Text>
              </View>
            ))}
          </ScrollView>
        </View>

        {/* AR Preview CTA */}
        <TouchableOpacity
          style={styles.arBanner}
          onPress={() => navigation.navigate('ARViewer')}
        >
          <LinearGradient
            colors={['#0EA5E9', '#0284C7']}
            style={styles.arGradient}
          >
            <Text style={styles.arEmoji}>📱</Text>
            <View style={{ flex: 1 }}>
              <Text style={styles.arTitle}>Try AR Preview</Text>
              <Text style={styles.arDesc}>
                See how your 3D print looks in your space before ordering
              </Text>
            </View>
            <Ionicons name="arrow-forward" size={24} color={Colors.white} />
          </LinearGradient>
        </TouchableOpacity>

        {/* Footer */}
        <View style={styles.footer}>
          <Text style={styles.footerBrand}>🖨️ 3DIL CREATION</Text>
          <Text style={styles.footerDesc}>Premium 3D Printing Services in India</Text>
          <View style={styles.footerLinks}>
            <TouchableOpacity onPress={() => navigation.navigate('Contact')}>
              <Text style={styles.footerLink}>📞 Contact Us</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => navigation.navigate('Blog')}>
              <Text style={styles.footerLink}>📝 Blog</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => navigation.navigate('Portfolio')}>
              <Text style={styles.footerLink}>🎨 Portfolio</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.footerCopy}>© 2024 3DIL Creation. Made with ❤️ in India</Text>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: {
    paddingTop: 48,
    paddingBottom: 16,
    paddingHorizontal: Spacing.md,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerGreeting: { color: 'rgba(255,255,255,0.8)', fontSize: FontSize.sm },
  headerBrand: {
    color: Colors.white,
    fontSize: FontSize.xxl,
    fontWeight: '900',
    letterSpacing: 2,
  },
  headerIcons: { flexDirection: 'row', gap: 8 },
  headerIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  cartBadge: {
    position: 'absolute',
    top: -2,
    right: -2,
    backgroundColor: Colors.accent,
    borderRadius: 8,
    minWidth: 16,
    height: 16,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 3,
  },
  cartBadgeText: { color: Colors.secondary, fontSize: 9, fontWeight: '800' },
  hero: {
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.xl,
    paddingBottom: Spacing.xxl,
    alignItems: 'center',
  },
  heroEmoji: { fontSize: 72, marginBottom: 16 },
  heroTitle: {
    fontSize: 32,
    fontWeight: '900',
    color: Colors.white,
    textAlign: 'center',
    lineHeight: 40,
    marginBottom: 12,
  },
  heroSubtitle: {
    fontSize: FontSize.md,
    color: 'rgba(255,255,255,0.85)',
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 24,
  },
  heroButtons: { flexDirection: 'row', gap: 12 },
  heroBtn: {
    backgroundColor: Colors.white,
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: BorderRadius.lg,
  },
  heroBtnText: { color: Colors.primary, fontWeight: '800', fontSize: FontSize.lg },
  heroBtnOutline: {
    borderWidth: 2,
    borderColor: Colors.white,
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: BorderRadius.lg,
  },
  heroBtnOutlineText: { color: Colors.white, fontWeight: '700', fontSize: FontSize.lg },
  statsRow: {
    flexDirection: 'row',
    backgroundColor: Colors.secondary,
    paddingVertical: 16,
    paddingHorizontal: 8,
    marginTop: -1,
  },
  statCard: { flex: 1, alignItems: 'center' },
  statEmoji: { fontSize: 20, marginBottom: 4 },
  statValue: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '800' },
  statLabel: { color: 'rgba(255,255,255,0.6)', fontSize: FontSize.xs },
  section: { paddingVertical: Spacing.md, paddingHorizontal: Spacing.md },
  sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  sectionTitle: {
    fontSize: FontSize.xl,
    fontWeight: '800',
    color: Colors.textPrimary,
    marginBottom: 12,
  },
  seeAll: { color: Colors.primary, fontWeight: '700', fontSize: FontSize.md },
  quickActionsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 12 },
  quickActionCard: {
    width: (width - 48 - 12) / 2,
    backgroundColor: Colors.card,
    borderRadius: BorderRadius.md,
    padding: 16,
    alignItems: 'center',
    gap: 10,
    ...Shadows.small,
  },
  quickActionIcon: {
    width: 56,
    height: 56,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  quickActionLabel: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary },
  serviceCard: {
    width: 160,
    backgroundColor: Colors.card,
    borderRadius: BorderRadius.md,
    padding: 16,
    marginRight: 12,
    ...Shadows.small,
  },
  serviceEmoji: { fontSize: 36, marginBottom: 8 },
  serviceName: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary, marginBottom: 4 },
  servicePrice: { fontSize: FontSize.sm, color: Colors.primary, fontWeight: '600', marginBottom: 2 },
  serviceTurnaround: { fontSize: FontSize.xs, color: Colors.textLight },
  quoteBanner: { marginHorizontal: Spacing.md, marginVertical: 8, borderRadius: BorderRadius.lg, overflow: 'hidden' },
  quoteBannerGradient: { borderRadius: BorderRadius.lg },
  quoteBannerContent: { flexDirection: 'row', alignItems: 'center', padding: 20, gap: 12 },
  quoteBannerEmoji: { fontSize: 36 },
  quoteBannerTitle: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '800', marginBottom: 4 },
  quoteBannerDesc: { color: 'rgba(255,255,255,0.8)', fontSize: FontSize.sm },
  productCard: {
    width: 160,
    backgroundColor: Colors.card,
    borderRadius: BorderRadius.md,
    marginRight: 12,
    overflow: 'hidden',
    ...Shadows.small,
  },
  productImageBox: {
    height: 120,
    backgroundColor: Colors.background,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  productEmoji: { fontSize: 56 },
  bestsellerBadge: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: Colors.accent,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 8,
  },
  bestsellerText: { fontSize: 9, fontWeight: '800', color: Colors.secondary },
  productName: {
    fontSize: FontSize.sm,
    fontWeight: '700',
    color: Colors.textPrimary,
    padding: 10,
    paddingBottom: 4,
  },
  productPriceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
  },
  productPrice: { fontSize: FontSize.md, fontWeight: '800', color: Colors.primary },
  productOriginal: {
    fontSize: FontSize.xs,
    color: Colors.textLight,
    textDecorationLine: 'line-through',
  },
  productRating: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 10,
    paddingBottom: 10,
    paddingTop: 4,
  },
  productRatingText: { fontSize: FontSize.xs, color: Colors.textSecondary },
  whyGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 12 },
  whyCard: {
    width: (width - 56) / 2,
    backgroundColor: Colors.card,
    borderRadius: BorderRadius.md,
    padding: 16,
    ...Shadows.small,
  },
  whyEmoji: { fontSize: 32, marginBottom: 8 },
  whyTitle: { fontSize: FontSize.md, fontWeight: '800', color: Colors.textPrimary, marginBottom: 4 },
  whyDesc: { fontSize: FontSize.xs, color: Colors.textSecondary, lineHeight: 18 },
  subscriptionBanner: {
    marginHorizontal: Spacing.md,
    marginVertical: 8,
    borderRadius: BorderRadius.lg,
    overflow: 'hidden',
  },
  subGradient: { borderRadius: BorderRadius.lg },
  subContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
  },
  subTitle: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.secondary, marginBottom: 4 },
  subDesc: { fontSize: FontSize.sm, color: Colors.secondary, opacity: 0.8 },
  subPrice: { fontSize: FontSize.md, fontWeight: '800', color: Colors.secondary, marginTop: 4 },
  testimonialCard: {
    width: 260,
    backgroundColor: Colors.card,
    borderRadius: BorderRadius.md,
    padding: 16,
    marginRight: 12,
    ...Shadows.small,
  },
  testimonialHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 10 },
  testimonialAvatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: Colors.primary + '20',
    alignItems: 'center',
    justifyContent: 'center',
  },
  testimonialAvatarText: { fontSize: 22 },
  testimonialName: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary },
  testimonialLocation: { fontSize: FontSize.xs, color: Colors.textSecondary },
  starsRow: { flexDirection: 'row', gap: 2, marginBottom: 8 },
  testimonialReview: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
    lineHeight: 20,
    fontStyle: 'italic',
  },
  testimonialService: {
    fontSize: FontSize.xs,
    color: Colors.primary,
    fontWeight: '600',
    marginTop: 8,
  },
  arBanner: {
    marginHorizontal: Spacing.md,
    marginVertical: 8,
    borderRadius: BorderRadius.lg,
    overflow: 'hidden',
  },
  arGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    gap: 12,
  },
  arEmoji: { fontSize: 36 },
  arTitle: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '800', marginBottom: 4 },
  arDesc: { color: 'rgba(255,255,255,0.85)', fontSize: FontSize.sm },
  footer: {
    backgroundColor: Colors.secondary,
    padding: 24,
    alignItems: 'center',
    gap: 8,
    marginTop: 16,
  },
  footerBrand: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '900', letterSpacing: 2 },
  footerDesc: { color: 'rgba(255,255,255,0.6)', fontSize: FontSize.sm },
  footerLinks: { flexDirection: 'row', gap: 20, marginTop: 8 },
  footerLink: { color: Colors.primary, fontSize: FontSize.sm, fontWeight: '600' },
  footerCopy: { color: 'rgba(255,255,255,0.4)', fontSize: FontSize.xs, marginTop: 8 },
});

export default HomeScreen;
