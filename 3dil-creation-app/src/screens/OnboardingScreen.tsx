import React, { useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  ScrollView,
  NativeSyntheticEvent,
  NativeScrollEvent,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Colors, FontSize, Spacing } from '../theme';

const { width } = Dimensions.get('window');
const ONBOARDING_KEY = '@3dil_onboarded';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Onboarding'>;
};

const slides = [
  {
    emoji: '🖨️',
    title: '3D Print Anything',
    description:
      'From custom medals and trophies to architectural models and statues — we print your vision with precision and premium quality.',
    gradient: ['#FF6B35', '#FF8C42'] as [string, string],
    highlight: 'Starting at just ₹199',
  },
  {
    emoji: '📱',
    title: 'AR Preview',
    description:
      'Use your camera to see exactly how your 3D printed product will look in your real space before placing the order.',
    gradient: ['#1A1A2E', '#16213E'] as [string, string],
    highlight: 'No surprises — see before you order',
  },
  {
    emoji: '🏆',
    title: 'Track & Earn Rewards',
    description:
      'Follow your order live — from design to delivery. Earn loyalty points with every purchase and unlock exclusive discounts.',
    gradient: ['#FFD700', '#FFA500'] as [string, string],
    highlight: 'Earn 1 point per ₹10 spent',
  },
];

const OnboardingScreen: React.FC<Props> = ({ navigation }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const scrollRef = useRef<ScrollView>(null);

  const handleScroll = (e: NativeSyntheticEvent<NativeScrollEvent>) => {
    const index = Math.round(e.nativeEvent.contentOffset.x / width);
    setCurrentIndex(index);
  };

  const goToNext = () => {
    if (currentIndex < slides.length - 1) {
      scrollRef.current?.scrollTo({ x: (currentIndex + 1) * width, animated: true });
    } else {
      finish();
    }
  };

  const finish = async () => {
    await AsyncStorage.setItem(ONBOARDING_KEY, 'true');
    navigation.replace('Main');
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.skip} onPress={finish}>
        <Text style={styles.skipText}>Skip</Text>
      </TouchableOpacity>

      <ScrollView
        ref={scrollRef}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        onMomentumScrollEnd={handleScroll}
      >
        {slides.map((slide, index) => (
          <LinearGradient
            key={index}
            colors={slide.gradient}
            style={styles.slide}
          >
            <View style={styles.emojiContainer}>
              <Text style={styles.emoji}>{slide.emoji}</Text>
            </View>
            <Text style={styles.title}>{slide.title}</Text>
            <Text style={styles.description}>{slide.description}</Text>
            <View style={styles.highlightBadge}>
              <Text style={styles.highlightText}>{slide.highlight}</Text>
            </View>
          </LinearGradient>
        ))}
      </ScrollView>

      <View style={styles.footer}>
        <View style={styles.dots}>
          {slides.map((_, i) => (
            <View
              key={i}
              style={[styles.dot, i === currentIndex && styles.activeDot]}
            />
          ))}
        </View>

        <TouchableOpacity style={styles.button} onPress={goToNext}>
          <LinearGradient
            colors={['#FF6B35', '#FF8C42']}
            style={styles.buttonGradient}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <Text style={styles.buttonText}>
              {currentIndex === slides.length - 1 ? 'Get Started 🚀' : 'Next →'}
            </Text>
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  skip: {
    position: 'absolute',
    top: 56,
    right: 24,
    zIndex: 10,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 20,
  },
  skipText: { color: Colors.white, fontSize: FontSize.md, fontWeight: '600' },
  slide: {
    width,
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 40,
    paddingTop: 80,
    paddingBottom: 160,
  },
  emojiContainer: {
    width: 140,
    height: 140,
    borderRadius: 35,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.3)',
    marginBottom: 32,
  },
  emoji: { fontSize: 72 },
  title: {
    fontSize: FontSize.xxxl,
    fontWeight: '800',
    color: Colors.white,
    textAlign: 'center',
    marginBottom: 16,
  },
  description: {
    fontSize: FontSize.lg,
    color: 'rgba(255,255,255,0.85)',
    textAlign: 'center',
    lineHeight: 26,
    marginBottom: 24,
  },
  highlightBadge: {
    backgroundColor: 'rgba(255,255,255,0.25)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.4)',
  },
  highlightText: {
    color: Colors.white,
    fontSize: FontSize.md,
    fontWeight: '700',
  },
  footer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingBottom: 48,
    paddingHorizontal: 24,
    alignItems: 'center',
    gap: 24,
    backgroundColor: 'transparent',
  },
  dots: { flexDirection: 'row', gap: 8 },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255,255,255,0.4)',
  },
  activeDot: {
    width: 24,
    backgroundColor: Colors.white,
  },
  button: { width: '100%', borderRadius: 16, overflow: 'hidden' },
  buttonGradient: {
    paddingVertical: 18,
    alignItems: 'center',
    borderRadius: 16,
  },
  buttonText: {
    color: Colors.white,
    fontSize: FontSize.xl,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
});

export default OnboardingScreen;
