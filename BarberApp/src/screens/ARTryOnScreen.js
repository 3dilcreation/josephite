import React, { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, ScrollView,
  Dimensions, Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Colors, Spacing, Radius } from '../theme/colors';
import { HAIRSTYLES } from '../data/mockData';

const { width, height } = Dimensions.get('window');

const FILTERS = [
  { name: 'All', icon: 'view-grid' },
  { name: 'Fade', icon: 'chevron-up' },
  { name: 'Classic', icon: 'crown' },
  { name: 'Textured', icon: 'texture' },
  { name: 'Bold', icon: 'flash' },
];

export default function ARTryOnScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const [selectedStyle, setSelectedStyle] = useState(HAIRSTYLES[0]);
  const [activeFilter, setActiveFilter] = useState('All');
  const [cameraReady, setCameraReady] = useState(false);
  const [captured, setCaptured] = useState(false);
  const pulseAnim = useRef(new Animated.Value(1)).current;

  const startPulse = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.15, duration: 600, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
      ])
    ).start();
  };

  const filteredStyles = activeFilter === 'All'
    ? HAIRSTYLES
    : HAIRSTYLES.filter(s => s.category === activeFilter);

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <MaterialCommunityIcons name="arrow-left" size={22} color={Colors.textPrimary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>AR Try-On</Text>
        <View style={styles.aiBadge}>
          <MaterialCommunityIcons name="star-four-points" size={12} color="#6C5CE7" />
          <Text style={styles.aiBadgeText}>AI-Powered</Text>
        </View>
      </View>

      {/* Camera Viewport */}
      <View style={styles.viewport}>
        <LinearGradient
          colors={['#1A1A2E', '#0A0A1A']}
          style={StyleSheet.absoluteFill}
        />

        {/* Simulated face/camera area */}
        <View style={styles.faceArea}>
          <View style={styles.faceGuide}>
            {/* Corner guides */}
            {[
              { top: 0, left: 0, borderTopWidth: 3, borderLeftWidth: 3 },
              { top: 0, right: 0, borderTopWidth: 3, borderRightWidth: 3 },
              { bottom: 0, left: 0, borderBottomWidth: 3, borderLeftWidth: 3 },
              { bottom: 0, right: 0, borderBottomWidth: 3, borderRightWidth: 3 },
            ].map((corner, i) => (
              <View key={i} style={[styles.corner, corner, { borderColor: Colors.primary }]} />
            ))}

            {/* Style Overlay Display */}
            <View style={styles.styleOverlay}>
              <LinearGradient
                colors={[selectedStyle.color + '40', selectedStyle.color + '15']}
                style={styles.styleOverlayGrad}
              >
                <Text style={styles.styleOverlayEmoji}>{selectedStyle.emoji}</Text>
                <Text style={styles.styleOverlayName}>{selectedStyle.name}</Text>
                <Text style={styles.styleOverlaySub}>Preview Active</Text>
              </LinearGradient>
            </View>
          </View>

          {/* AI Analysis floating card */}
          <View style={styles.aiAnalysisCard}>
            <LinearGradient colors={['#1A0A2E', '#0A0010']} style={StyleSheet.absoluteFill} borderRadius={12} />
            <MaterialCommunityIcons name="face-recognition" size={18} color="#6C5CE7" />
            <View>
              <Text style={styles.aiAnalysisTitle}>Face Shape: Oval</Text>
              <Text style={styles.aiAnalysisSub}>Best match for this style</Text>
            </View>
            <View style={styles.matchScore}>
              <Text style={styles.matchScoreText}>98%</Text>
            </View>
          </View>
        </View>

        {/* Camera Controls */}
        <View style={styles.cameraControls}>
          <TouchableOpacity style={styles.controlBtn}>
            <MaterialCommunityIcons name="camera-flip" size={22} color={Colors.textPrimary} />
          </TouchableOpacity>

          <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
            <TouchableOpacity
              style={styles.captureBtn}
              onPress={() => { setCaptured(true); startPulse(); }}
              activeOpacity={0.8}
            >
              <LinearGradient colors={Colors.gradientGold} style={styles.captureBtnInner}>
                <MaterialCommunityIcons name="camera" size={28} color="#0A0A0F" />
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View>

          <TouchableOpacity style={styles.controlBtn}>
            <MaterialCommunityIcons name="share-variant" size={22} color={Colors.textPrimary} />
          </TouchableOpacity>
        </View>

        {captured && (
          <View style={styles.capturedBanner}>
            <MaterialCommunityIcons name="check-circle" size={18} color={Colors.success} />
            <Text style={styles.capturedText}>Saved to Hair Journey!</Text>
          </View>
        )}
      </View>

      {/* Style Selector */}
      <View style={styles.stylePanel}>
        {/* Filters */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filtersScroll}>
          {FILTERS.map((f) => (
            <TouchableOpacity
              key={f.name}
              style={[styles.filterChip, activeFilter === f.name && styles.filterChipActive]}
              onPress={() => setActiveFilter(f.name)}
              activeOpacity={0.8}
            >
              {activeFilter === f.name && (
                <LinearGradient colors={Colors.gradientGold} style={StyleSheet.absoluteFill} borderRadius={Radius.full} />
              )}
              <MaterialCommunityIcons
                name={f.icon}
                size={14}
                color={activeFilter === f.name ? '#0A0A0F' : Colors.textSecondary}
              />
              <Text style={[styles.filterText, activeFilter === f.name && { color: '#0A0A0F' }]}>
                {f.name}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Styles */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.stylesScroll}>
          {filteredStyles.map((style) => (
            <TouchableOpacity
              key={style.id}
              style={[styles.styleChip, selectedStyle.id === style.id && styles.styleChipSelected]}
              onPress={() => { setSelectedStyle(style); setCaptured(false); }}
              activeOpacity={0.85}
            >
              <View style={[
                styles.styleChipIcon,
                { backgroundColor: style.color + (selectedStyle.id === style.id ? '40' : '15') },
                selectedStyle.id === style.id && { borderColor: style.color, borderWidth: 2 },
              ]}>
                <Text style={styles.styleChipEmoji}>{style.emoji}</Text>
              </View>
              <Text style={[
                styles.styleChipName,
                selectedStyle.id === style.id && { color: Colors.textPrimary },
              ]}>
                {style.name}
              </Text>
              {style.trending && (
                <View style={[styles.trendingDot, { backgroundColor: style.color }]} />
              )}
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Book This Style Button */}
        <TouchableOpacity
          style={styles.bookStyleBtn}
          onPress={() => navigation.navigate('Main', { screen: 'Book' })}
          activeOpacity={0.85}
        >
          <LinearGradient
            colors={Colors.gradientGold}
            style={styles.bookStyleGrad}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <MaterialCommunityIcons name="calendar-plus" size={20} color="#0A0A0F" />
            <Text style={styles.bookStyleText}>Book This Style · {selectedStyle.name}</Text>
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
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
  aiBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#6C5CE720',
    borderRadius: Radius.full,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderWidth: 1,
    borderColor: '#6C5CE730',
  },
  aiBadgeText: { color: '#6C5CE7', fontSize: 11, fontWeight: '700' },
  viewport: {
    flex: 1,
    marginHorizontal: Spacing.md,
    borderRadius: Radius.xl,
    overflow: 'hidden',
    position: 'relative',
  },
  faceArea: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: Spacing.xl,
  },
  faceGuide: {
    width: width * 0.55,
    height: width * 0.7,
    position: 'relative',
    alignItems: 'center',
    justifyContent: 'center',
  },
  corner: {
    position: 'absolute',
    width: 24,
    height: 24,
    borderColor: Colors.primary,
  },
  styleOverlay: {
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: Radius.lg,
    overflow: 'hidden',
    padding: Spacing.lg,
  },
  styleOverlayGrad: {
    padding: Spacing.xl,
    borderRadius: Radius.lg,
    alignItems: 'center',
  },
  styleOverlayEmoji: { fontSize: 64, marginBottom: 8 },
  styleOverlayName: { color: Colors.textPrimary, fontSize: 22, fontWeight: '800', marginBottom: 4 },
  styleOverlaySub: { color: Colors.textSecondary, fontSize: 12 },
  aiAnalysisCard: {
    position: 'absolute',
    top: Spacing.md,
    left: Spacing.md,
    right: Spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#6C5CE730',
    overflow: 'hidden',
  },
  aiAnalysisTitle: { color: Colors.textPrimary, fontSize: 13, fontWeight: '700' },
  aiAnalysisSub: { color: Colors.textSecondary, fontSize: 11 },
  matchScore: {
    marginLeft: 'auto',
    backgroundColor: Colors.success + '20',
    borderRadius: Radius.full,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  matchScoreText: { color: Colors.success, fontSize: 13, fontWeight: '800' },
  cameraControls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
    padding: Spacing.lg,
    paddingBottom: Spacing.xl,
  },
  controlBtn: {
    width: 46,
    height: 46,
    borderRadius: 23,
    backgroundColor: Colors.bgGlassStrong,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
  },
  captureBtn: {
    width: 70,
    height: 70,
    borderRadius: 35,
    padding: 4,
    backgroundColor: Colors.primary + '30',
  },
  captureBtnInner: {
    flex: 1,
    borderRadius: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  capturedBanner: {
    position: 'absolute',
    top: Spacing.md,
    right: Spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: Colors.success + '20',
    borderRadius: Radius.full,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: Colors.success + '40',
  },
  capturedText: { color: Colors.success, fontSize: 12, fontWeight: '700' },
  stylePanel: {
    backgroundColor: Colors.bgCard,
    borderTopLeftRadius: Radius.xl,
    borderTopRightRadius: Radius.xl,
    padding: Spacing.md,
    paddingTop: Spacing.lg,
    gap: Spacing.md,
  },
  filtersScroll: { marginBottom: 4 },
  filterChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: Radius.full,
    backgroundColor: Colors.bgCardAlt,
    marginRight: 8,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
  },
  filterChipActive: { borderColor: Colors.primary },
  filterText: { color: Colors.textSecondary, fontSize: 12, fontWeight: '600' },
  stylesScroll: { marginBottom: 4 },
  styleChip: { alignItems: 'center', marginRight: 14 },
  styleChipSelected: {},
  styleChipIcon: {
    width: 60,
    height: 60,
    borderRadius: Radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 6,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  styleChipEmoji: { fontSize: 28 },
  styleChipName: { color: Colors.textSecondary, fontSize: 10, fontWeight: '600', textAlign: 'center', maxWidth: 60 },
  trendingDot: { width: 6, height: 6, borderRadius: 3, marginTop: 2 },
  bookStyleBtn: { borderRadius: Radius.full, overflow: 'hidden' },
  bookStyleGrad: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 16,
  },
  bookStyleText: { color: '#0A0A0F', fontSize: 15, fontWeight: '800' },
});
