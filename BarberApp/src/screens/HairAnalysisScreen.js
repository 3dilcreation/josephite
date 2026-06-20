import React, { useState, useRef, useEffect } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Animated, Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Colors, Spacing, Radius } from '../theme/colors';
import { HAIRSTYLES } from '../data/mockData';

const { width } = Dimensions.get('window');

const FACE_SHAPES = [
  { name: 'Oval', emoji: '🥚', best: 'Most styles work great!', match: 98 },
  { name: 'Square', emoji: '⬜', best: 'Fades and tapers soften angles', match: 0 },
  { name: 'Round', emoji: '🔵', best: 'High fades elongate the face', match: 0 },
  { name: 'Heart', emoji: '💜', best: 'Side-swept styles balance proportions', match: 0 },
  { name: 'Diamond', emoji: '🔷', best: 'Volume at crown looks amazing', match: 0 },
];

const RECOMMENDATIONS = HAIRSTYLES.filter(h => ['h1', 'h5', 'h8', 'h11'].includes(h.id));

const TRAITS = [
  { label: 'Face Shape', value: 'Oval', icon: 'face-outline', confidence: 94, color: '#6C5CE7' },
  { label: 'Hair Type', value: 'Wavy', icon: 'waves', confidence: 89, color: '#5C9EE0' },
  { label: 'Hair Density', value: 'Medium', icon: 'dots-horizontal', confidence: 96, color: '#4CAF82' },
  { label: 'Hair Length', value: '2 inches', icon: 'ruler', confidence: 99, color: '#C8A96E' },
];

export default function HairAnalysisScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const [phase, setPhase] = useState('intro'); // intro | scanning | results
  const scanAnim = useRef(new Animated.Value(0)).current;
  const resultsAnim = useRef(new Animated.Value(0)).current;
  const [scanProgress, setScanProgress] = useState(0);

  useEffect(() => {
    if (phase === 'scanning') {
      Animated.loop(
        Animated.timing(scanAnim, { toValue: 1, duration: 2000, useNativeDriver: true })
      ).start();

      // Progress simulation
      let progress = 0;
      const interval = setInterval(() => {
        progress += Math.random() * 15 + 5;
        if (progress >= 100) {
          progress = 100;
          clearInterval(interval);
          setTimeout(() => {
            setPhase('results');
            Animated.spring(resultsAnim, { toValue: 1, tension: 50, friction: 10, useNativeDriver: true }).start();
          }, 500);
        }
        setScanProgress(Math.min(progress, 100));
      }, 200);

      return () => clearInterval(interval);
    }
  }, [phase]);

  const scanY = scanAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [-200, 200],
  });

  if (phase === 'intro') {
    return (
      <View style={[styles.container, { paddingTop: insets.top }]}>
        <View style={styles.header}>
          <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
            <MaterialCommunityIcons name="arrow-left" size={22} color={Colors.textPrimary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>AI Hair Analysis</Text>
        </View>

        <ScrollView contentContainerStyle={styles.introContent}>
          <LinearGradient colors={['#0A001A', '#050012']} style={styles.introHero}>
            <View style={styles.introIconBg}>
              <MaterialCommunityIcons name="face-recognition" size={80} color="#6C5CE750" />
            </View>
            <View style={[styles.introIconBadge]}>
              <LinearGradient colors={['#6C5CE7', '#2D1B69']} style={styles.introIconBadgeGrad}>
                <MaterialCommunityIcons name="brain" size={36} color="#FFF" />
              </LinearGradient>
            </View>
          </LinearGradient>

          <View style={styles.introText}>
            <Text style={styles.introTitle}>Your Perfect Style{'\n'}Starts with AI</Text>
            <Text style={styles.introSub}>
              Our AI analyzes your face shape, hair type, texture, and growth patterns to recommend the ideal hairstyles just for you.
            </Text>
          </View>

          <View style={styles.introFeatures}>
            {[
              { icon: 'face-recognition', text: 'Face Shape Detection', color: '#6C5CE7' },
              { icon: 'texture', text: 'Hair Type Analysis', color: '#5C9EE0' },
              { icon: 'palette', text: 'Style Matching Algorithm', color: '#C8A96E' },
              { icon: 'star-shooting', text: 'Personalized Recommendations', color: '#4CAF82' },
            ].map((f) => (
              <View key={f.text} style={styles.introFeatureRow}>
                <View style={[styles.introFeatureIcon, { backgroundColor: f.color + '20' }]}>
                  <MaterialCommunityIcons name={f.icon} size={20} color={f.color} />
                </View>
                <Text style={styles.introFeatureText}>{f.text}</Text>
                <MaterialCommunityIcons name="check-circle" size={16} color={Colors.success} />
              </View>
            ))}
          </View>

          <View style={styles.privacyNote}>
            <MaterialCommunityIcons name="shield-check" size={16} color={Colors.success} />
            <Text style={styles.privacyText}>
              Your photo is analyzed locally and never stored on our servers.
            </Text>
          </View>

          <TouchableOpacity style={styles.startBtn} onPress={() => setPhase('scanning')} activeOpacity={0.85}>
            <LinearGradient colors={['#6C5CE7', '#2D1B69']} style={styles.startBtnGrad} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}>
              <MaterialCommunityIcons name="camera" size={20} color="#FFF" />
              <Text style={styles.startBtnText}>Start AI Analysis</Text>
            </LinearGradient>
          </TouchableOpacity>
        </ScrollView>
      </View>
    );
  }

  if (phase === 'scanning') {
    return (
      <View style={[styles.container, { paddingTop: insets.top, alignItems: 'center', justifyContent: 'center' }]}>
        <LinearGradient colors={['#0A001A', '#050012', '#0A0A0F']} style={StyleSheet.absoluteFill} />

        <View style={styles.scanViewport}>
          {/* Face guide corners */}
          {[
            { top: 0, left: 0, borderTopWidth: 3, borderLeftWidth: 3 },
            { top: 0, right: 0, borderTopWidth: 3, borderRightWidth: 3 },
            { bottom: 0, left: 0, borderBottomWidth: 3, borderLeftWidth: 3 },
            { bottom: 0, right: 0, borderBottomWidth: 3, borderRightWidth: 3 },
          ].map((corner, i) => (
            <View key={i} style={[styles.scanCorner, corner]} />
          ))}

          {/* Face emoji */}
          <Text style={styles.scanFaceEmoji}>😐</Text>

          {/* Scan line */}
          <Animated.View style={[styles.scanLine, { transform: [{ translateY: scanY }] }]}>
            <LinearGradient
              colors={['transparent', '#6C5CE780', 'transparent']}
              style={styles.scanLineGrad}
            />
          </Animated.View>
        </View>

        <View style={styles.scanInfo}>
          <Text style={styles.scanTitle}>Analyzing Your Features...</Text>

          <View style={styles.scanTasks}>
            {[
              { task: 'Face shape detection', done: scanProgress > 25 },
              { task: 'Hair type analysis', done: scanProgress > 50 },
              { task: 'Texture mapping', done: scanProgress > 75 },
              { task: 'Style matching', done: scanProgress >= 100 },
            ].map((t) => (
              <View key={t.task} style={styles.scanTask}>
                <MaterialCommunityIcons
                  name={t.done ? 'check-circle' : 'circle-outline'}
                  size={16}
                  color={t.done ? Colors.success : Colors.textMuted}
                />
                <Text style={[styles.scanTaskText, t.done && { color: Colors.success }]}>
                  {t.task}
                </Text>
              </View>
            ))}
          </View>

          <View style={styles.progressBar}>
            <View style={[styles.progressFill, { width: `${scanProgress}%` }]} />
          </View>
          <Text style={styles.progressText}>{Math.round(scanProgress)}%</Text>
        </View>
      </View>
    );
  }

  // Results phase
  return (
    <Animated.View style={[styles.container, { paddingTop: insets.top, opacity: resultsAnim }]}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <MaterialCommunityIcons name="arrow-left" size={22} color={Colors.textPrimary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Your Results</Text>
        <TouchableOpacity style={styles.rescanBtn} onPress={() => { setPhase('intro'); setScanProgress(0); }}>
          <Text style={styles.rescanText}>Rescan</Text>
        </TouchableOpacity>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 40 }}>
        {/* Face Shape Result */}
        <View style={styles.faceShapeCard}>
          <LinearGradient colors={['#0A001A', '#050012']} style={StyleSheet.absoluteFill} borderRadius={Radius.xl} />
          <View style={styles.faceShapeResult}>
            <Text style={styles.faceShapeEmoji}>🥚</Text>
            <View>
              <Text style={styles.faceShapeLabel}>Your Face Shape</Text>
              <Text style={styles.faceShapeName}>Oval</Text>
              <Text style={styles.faceShapeDesc}>Most versatile face shape — nearly any style works!</Text>
            </View>
          </View>
          <View style={styles.confidenceBar}>
            <View style={styles.confidenceFill} />
          </View>
          <Text style={styles.confidenceText}>94% confidence</Text>
        </View>

        {/* Analysis Traits */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Analysis Results</Text>
          <View style={styles.traitsGrid}>
            {TRAITS.map((trait) => (
              <View key={trait.label} style={styles.traitCard}>
                <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.md} />
                <View style={[styles.traitIcon, { backgroundColor: trait.color + '20' }]}>
                  <MaterialCommunityIcons name={trait.icon} size={20} color={trait.color} />
                </View>
                <Text style={styles.traitLabel}>{trait.label}</Text>
                <Text style={[styles.traitValue, { color: trait.color }]}>{trait.value}</Text>
                <View style={styles.traitConfBar}>
                  <View style={[styles.traitConfFill, { width: `${trait.confidence}%`, backgroundColor: trait.color }]} />
                </View>
              </View>
            ))}
          </View>
        </View>

        {/* Recommendations */}
        <View style={styles.section}>
          <View style={styles.recHeader}>
            <MaterialCommunityIcons name="star-four-points" size={18} color="#6C5CE7" />
            <Text style={styles.sectionTitle}>AI Recommended Styles</Text>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.recScroll}>
            {RECOMMENDATIONS.map((style) => (
              <TouchableOpacity key={style.id} style={styles.recCard} activeOpacity={0.85}>
                <LinearGradient
                  colors={[style.color + '30', style.color + '10']}
                  style={styles.recCardGrad}
                >
                  <Text style={styles.recEmoji}>{style.emoji}</Text>
                  <Text style={styles.recName}>{style.name}</Text>
                  <View style={styles.recMatch}>
                    <Text style={[styles.recMatchText, { color: style.color }]}>Great Match</Text>
                  </View>
                </LinearGradient>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        {/* Book Now CTA */}
        <View style={styles.section}>
          <TouchableOpacity
            style={styles.bookCta}
            onPress={() => navigation.navigate('Main', { screen: 'Book' })}
            activeOpacity={0.9}
          >
            <LinearGradient colors={['#6C5CE7', '#2D1B69']} style={styles.bookCtaGrad} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}>
              <MaterialCommunityIcons name="calendar-plus" size={22} color="#FFF" />
              <Text style={styles.bookCtaText}>Book with AI Recommendation</Text>
              <MaterialCommunityIcons name="arrow-right" size={18} color="#FFF80" />
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.md,
    gap: 12,
  },
  backBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.bgGlass,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: { flex: 1, color: Colors.textPrimary, fontSize: 20, fontWeight: '800' },
  rescanBtn: { paddingHorizontal: 12, paddingVertical: 6 },
  rescanText: { color: Colors.primary, fontSize: 14, fontWeight: '700' },
  introContent: { padding: Spacing.md, paddingBottom: 40 },
  introHero: {
    height: 200,
    borderRadius: Radius.xl,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.xl,
    overflow: 'hidden',
    position: 'relative',
  },
  introIconBg: { position: 'absolute' },
  introIconBadge: { width: 90, height: 90, borderRadius: 45, overflow: 'hidden' },
  introIconBadgeGrad: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  introText: { marginBottom: Spacing.xl },
  introTitle: { color: Colors.textPrimary, fontSize: 32, fontWeight: '800', lineHeight: 40, marginBottom: 10 },
  introSub: { color: Colors.textSecondary, fontSize: 15, lineHeight: 24 },
  introFeatures: { gap: 10, marginBottom: Spacing.lg },
  introFeatureRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: Spacing.md,
    backgroundColor: Colors.bgCardAlt,
    borderRadius: Radius.md,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
  },
  introFeatureIcon: {
    width: 38,
    height: 38,
    borderRadius: Radius.sm,
    alignItems: 'center',
    justifyContent: 'center',
  },
  introFeatureText: { flex: 1, color: Colors.textPrimary, fontSize: 14, fontWeight: '600' },
  privacyNote: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: Colors.success + '15',
    borderRadius: Radius.md,
    padding: Spacing.md,
    marginBottom: Spacing.lg,
    borderWidth: 1,
    borderColor: Colors.success + '30',
  },
  privacyText: { color: Colors.success, fontSize: 12, flex: 1, lineHeight: 18 },
  startBtn: { borderRadius: Radius.full, overflow: 'hidden' },
  startBtnGrad: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 18,
  },
  startBtnText: { color: '#FFF', fontSize: 16, fontWeight: '800' },
  scanViewport: {
    width: width * 0.65,
    height: width * 0.8,
    borderRadius: Radius.xl,
    overflow: 'hidden',
    backgroundColor: Colors.bgCardAlt,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    marginBottom: Spacing.xl,
  },
  scanCorner: {
    position: 'absolute',
    width: 30,
    height: 30,
    borderColor: '#6C5CE7',
  },
  scanFaceEmoji: { fontSize: 80 },
  scanLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 40,
    alignItems: 'center',
  },
  scanLineGrad: {
    width: '100%',
    height: '100%',
  },
  scanInfo: { alignItems: 'center', paddingHorizontal: Spacing.xl, width: '100%' },
  scanTitle: { color: Colors.textPrimary, fontSize: 20, fontWeight: '700', marginBottom: Spacing.lg },
  scanTasks: { width: '100%', gap: 10, marginBottom: Spacing.lg },
  scanTask: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  scanTaskText: { color: Colors.textMuted, fontSize: 14 },
  progressBar: {
    width: '100%',
    height: 6,
    backgroundColor: Colors.bgCardAlt,
    borderRadius: 3,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressFill: { height: '100%', backgroundColor: '#6C5CE7', borderRadius: 3 },
  progressText: { color: '#6C5CE7', fontSize: 14, fontWeight: '700' },
  faceShapeCard: {
    marginHorizontal: Spacing.md,
    borderRadius: Radius.xl,
    padding: Spacing.lg,
    marginBottom: Spacing.lg,
    borderWidth: 1,
    borderColor: '#6C5CE730',
    overflow: 'hidden',
  },
  faceShapeResult: { flexDirection: 'row', alignItems: 'center', gap: 14, marginBottom: Spacing.md },
  faceShapeEmoji: { fontSize: 48 },
  faceShapeLabel: { color: Colors.textSecondary, fontSize: 12, marginBottom: 4 },
  faceShapeName: { color: Colors.textPrimary, fontSize: 28, fontWeight: '900', marginBottom: 4 },
  faceShapeDesc: { color: Colors.textSecondary, fontSize: 13 },
  confidenceBar: {
    height: 6,
    backgroundColor: Colors.bgGlass,
    borderRadius: 3,
    overflow: 'hidden',
    marginBottom: 6,
  },
  confidenceFill: { height: '100%', width: '94%', backgroundColor: '#6C5CE7', borderRadius: 3 },
  confidenceText: { color: '#6C5CE7', fontSize: 12, fontWeight: '600' },
  section: { paddingHorizontal: Spacing.md, marginBottom: Spacing.lg },
  sectionTitle: { color: Colors.textPrimary, fontSize: 18, fontWeight: '700', marginBottom: 12 },
  traitsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  traitCard: {
    width: (width - Spacing.md * 2 - 8) / 2,
    borderRadius: Radius.md,
    padding: Spacing.md,
    gap: 6,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
  },
  traitIcon: {
    width: 38,
    height: 38,
    borderRadius: Radius.sm,
    alignItems: 'center',
    justifyContent: 'center',
  },
  traitLabel: { color: Colors.textSecondary, fontSize: 11 },
  traitValue: { fontSize: 16, fontWeight: '800' },
  traitConfBar: { height: 4, backgroundColor: Colors.bgGlass, borderRadius: 2, overflow: 'hidden' },
  traitConfFill: { height: '100%', borderRadius: 2 },
  recHeader: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 12 },
  recScroll: { marginHorizontal: -Spacing.md, paddingHorizontal: Spacing.md },
  recCard: { marginRight: 12 },
  recCardGrad: {
    width: 120,
    height: 140,
    borderRadius: Radius.lg,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    padding: Spacing.md,
  },
  recEmoji: { fontSize: 40 },
  recName: { color: Colors.textPrimary, fontSize: 13, fontWeight: '700', textAlign: 'center' },
  recMatch: { backgroundColor: Colors.success + '20', borderRadius: Radius.full, paddingHorizontal: 8, paddingVertical: 3 },
  recMatchText: { fontSize: 10, fontWeight: '700' },
  bookCta: { borderRadius: Radius.full, overflow: 'hidden' },
  bookCtaGrad: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 18,
  },
  bookCtaText: { color: '#FFF', fontSize: 16, fontWeight: '800' },
});
