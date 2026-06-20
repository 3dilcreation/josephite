import React, { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  Dimensions, FlatList, Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, Spacing, Radius } from '../theme/colors';

const { width, height } = Dimensions.get('window');

const SLIDES = [
  {
    id: '1',
    icon: 'magic-staff',
    iconColor: '#C8A96E',
    gradient: ['#1A1000', '#0A0A0F'],
    accentGrad: ['#C8A96E', '#A07840'],
    title: 'Welcome to\nFadeBlades',
    subtitle: 'The smartest barber app ever built. AI-powered cuts, AR try-on, and real-time queue — all in one place.',
    badge: 'The Future of Grooming',
  },
  {
    id: '2',
    icon: 'face-recognition',
    iconColor: '#5C9EE0',
    gradient: ['#000A1A', '#0A0A0F'],
    accentGrad: ['#5C9EE0', '#1A4A80'],
    title: 'AI Face Analysis\n& Style Match',
    subtitle: 'Our AI scans your face shape and recommends the perfect hairstyle just for you. No more guessing.',
    badge: 'Powered by AI',
  },
  {
    id: '3',
    icon: 'augmented-reality',
    iconColor: '#6C5CE7',
    gradient: ['#0A001A', '#0A0A0F'],
    accentGrad: ['#6C5CE7', '#2D1B69'],
    title: 'Try Before\nYou Cut',
    subtitle: 'See how any hairstyle looks on you with our AR Try-On. Choose your cut with confidence.',
    badge: 'AR Technology',
  },
  {
    id: '4',
    icon: 'crown',
    iconColor: '#F0C849',
    gradient: ['#1A1000', '#0A0A0F'],
    accentGrad: ['#F0C849', '#A07830'],
    title: 'Earn Rewards\nEvery Visit',
    subtitle: 'Level up, unlock badges, and earn free cuts with our gamified loyalty system. The more you visit, the more you win.',
    badge: 'Gamified Loyalty',
  },
];

export default function OnboardingScreen({ onFinish }) {
  const insets = useSafeAreaInsets();
  const [current, setCurrent] = useState(0);
  const flatListRef = useRef(null);
  const dotAnim = useRef(SLIDES.map(() => new Animated.Value(0))).current;

  const animateDot = (index) => {
    dotAnim.forEach((anim, i) => {
      Animated.spring(anim, {
        toValue: i === index ? 1 : 0,
        useNativeDriver: false,
      }).start();
    });
  };

  const handleScroll = (e) => {
    const index = Math.round(e.nativeEvent.contentOffset.x / width);
    if (index !== current) {
      setCurrent(index);
      animateDot(index);
    }
  };

  const goNext = () => {
    if (current < SLIDES.length - 1) {
      flatListRef.current?.scrollToIndex({ index: current + 1 });
      setCurrent(current + 1);
      animateDot(current + 1);
    } else {
      onFinish();
    }
  };

  const slide = SLIDES[current];

  return (
    <View style={styles.container}>
      <LinearGradient colors={['#0A0A0F', '#0A0A0F']} style={StyleSheet.absoluteFill} />

      <FlatList
        ref={flatListRef}
        data={SLIDES}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        onScroll={handleScroll}
        scrollEventThrottle={16}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <View style={[styles.slide, { width }]}>
            <LinearGradient colors={item.gradient} style={StyleSheet.absoluteFill} />

            <View style={styles.iconContainer}>
              <LinearGradient
                colors={item.accentGrad}
                style={styles.iconBg}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
              >
                <MaterialCommunityIcons name={item.icon} size={64} color="#FFF" />
              </LinearGradient>
              <View style={[styles.iconGlow, { backgroundColor: item.iconColor + '30' }]} />
            </View>

            <View style={[styles.badge, { backgroundColor: item.iconColor + '20', borderColor: item.iconColor + '40' }]}>
              <MaterialCommunityIcons name="star-four-points" size={12} color={item.iconColor} />
              <Text style={[styles.badgeText, { color: item.iconColor }]}>{item.badge}</Text>
            </View>

            <Text style={styles.title}>{item.title}</Text>
            <Text style={styles.subtitle}>{item.subtitle}</Text>
          </View>
        )}
      />

      <View style={[styles.bottom, { paddingBottom: insets.bottom + Spacing.lg }]}>
        <View style={styles.dots}>
          {SLIDES.map((_, i) => {
            const dotWidth = dotAnim[i].interpolate({
              inputRange: [0, 1],
              outputRange: [8, 28],
            });
            const dotColor = dotAnim[i].interpolate({
              inputRange: [0, 1],
              outputRange: [Colors.textMuted, Colors.primary],
            });
            return (
              <Animated.View
                key={i}
                style={[styles.dot, { width: dotWidth, backgroundColor: dotColor }]}
              />
            );
          })}
        </View>

        <TouchableOpacity onPress={goNext} activeOpacity={0.85}>
          <LinearGradient
            colors={Colors.gradientGold}
            style={styles.btn}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <Text style={styles.btnText}>
              {current === SLIDES.length - 1 ? "Let's Get Sharp" : 'Next'}
            </Text>
            <MaterialCommunityIcons
              name={current === SLIDES.length - 1 ? 'check' : 'arrow-right'}
              size={20}
              color="#0A0A0F"
            />
          </LinearGradient>
        </TouchableOpacity>

        {current < SLIDES.length - 1 && (
          <TouchableOpacity onPress={onFinish} style={styles.skipBtn}>
            <Text style={styles.skipText}>Skip</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  slide: {
    flex: 1,
    height,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Spacing.xl,
    paddingTop: 80,
  },
  iconContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.xl,
  },
  iconBg: {
    width: 140,
    height: 140,
    borderRadius: 70,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconGlow: {
    position: 'absolute',
    width: 200,
    height: 200,
    borderRadius: 100,
    zIndex: -1,
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: Radius.full,
    borderWidth: 1,
    marginBottom: Spacing.lg,
  },
  badgeText: { fontSize: 12, fontWeight: '700', letterSpacing: 1 },
  title: {
    fontSize: 36,
    fontWeight: '800',
    color: Colors.textPrimary,
    textAlign: 'center',
    lineHeight: 44,
    marginBottom: Spacing.md,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 16,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: 26,
    maxWidth: 320,
  },
  bottom: {
    paddingHorizontal: Spacing.xl,
    paddingTop: Spacing.lg,
    alignItems: 'center',
    gap: Spacing.md,
  },
  dots: {
    flexDirection: 'row',
    gap: 6,
    marginBottom: Spacing.sm,
  },
  dot: {
    height: 8,
    borderRadius: 4,
  },
  btn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 18,
    paddingHorizontal: 48,
    borderRadius: Radius.full,
    minWidth: 200,
    justifyContent: 'center',
  },
  btnText: {
    fontSize: 17,
    fontWeight: '800',
    color: '#0A0A0F',
    letterSpacing: 0.5,
  },
  skipBtn: { paddingVertical: 8 },
  skipText: { color: Colors.textMuted, fontSize: 14 },
});
