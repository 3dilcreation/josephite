import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width, height } = Dimensions.get('window');
const ONBOARDING_KEY = '@3dil_onboarded';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Splash'>;
};

const SplashScreen: React.FC<Props> = ({ navigation }) => {
  const scaleAnim = useRef(new Animated.Value(0)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const rotateAnim = useRef(new Animated.Value(0)).current;
  const taglineAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.parallel([
        Animated.spring(scaleAnim, {
          toValue: 1,
          tension: 60,
          friction: 7,
          useNativeDriver: true,
        }),
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 600,
          useNativeDriver: true,
        }),
      ]),
      Animated.timing(rotateAnim, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }),
      Animated.timing(taglineAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }),
    ]).start();

    const timer = setTimeout(async () => {
      try {
        const onboarded = await AsyncStorage.getItem(ONBOARDING_KEY);
        navigation.replace(onboarded ? 'Main' : 'Onboarding');
      } catch {
        navigation.replace('Onboarding');
      }
    }, 2800);

    return () => clearTimeout(timer);
  }, [navigation, scaleAnim, fadeAnim, rotateAnim, taglineAnim]);

  const rotate = rotateAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  return (
    <LinearGradient
      colors={['#FF6B35', '#E55A2B', '#1A1A2E']}
      style={styles.container}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      <View style={styles.content}>
        <Animated.View
          style={[
            styles.logoContainer,
            {
              transform: [{ scale: scaleAnim }, { rotate }],
              opacity: fadeAnim,
            },
          ]}
        >
          <Text style={styles.logoEmoji}>🖨️</Text>
        </Animated.View>

        <Animated.View style={{ opacity: fadeAnim }}>
          <Text style={styles.title}>3DIL</Text>
          <Text style={styles.subtitle}>CREATION</Text>
        </Animated.View>

        <Animated.View
          style={[
            styles.taglineContainer,
            { opacity: taglineAnim, transform: [{ translateY: taglineAnim.interpolate({ inputRange: [0, 1], outputRange: [20, 0] }) }] },
          ]}
        >
          <Text style={styles.tagline}>Bringing Your Ideas to Life</Text>
          <Text style={styles.taglineSub}>Premium 3D Printing Services</Text>
        </Animated.View>

        <View style={styles.dotsContainer}>
          {[0, 1, 2].map(i => (
            <Animated.View
              key={i}
              style={[
                styles.dot,
                {
                  opacity: fadeAnim,
                  transform: [
                    {
                      scale: rotateAnim.interpolate({
                        inputRange: [0, 0.5, 1],
                        outputRange: [0.5, 1.2, 1],
                      }),
                    },
                  ],
                },
              ]}
            />
          ))}
        </View>
      </View>

      <Text style={styles.version}>v1.0.0</Text>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  content: { alignItems: 'center', gap: 16 },
  logoContainer: {
    width: 120,
    height: 120,
    borderRadius: 30,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.4)',
  },
  logoEmoji: { fontSize: 64 },
  title: {
    fontSize: 48,
    fontWeight: '900',
    color: '#FFFFFF',
    letterSpacing: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 18,
    fontWeight: '600',
    color: 'rgba(255,255,255,0.85)',
    letterSpacing: 12,
    textAlign: 'center',
    marginTop: -8,
  },
  taglineContainer: { alignItems: 'center', marginTop: 8 },
  tagline: {
    fontSize: 16,
    color: 'rgba(255,255,255,0.9)',
    fontWeight: '500',
  },
  taglineSub: {
    fontSize: 13,
    color: 'rgba(255,255,255,0.65)',
    marginTop: 4,
  },
  dotsContainer: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 32,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255,255,255,0.7)',
  },
  version: {
    position: 'absolute',
    bottom: 48,
    color: 'rgba(255,255,255,0.4)',
    fontSize: 12,
  },
});

export default SplashScreen;
