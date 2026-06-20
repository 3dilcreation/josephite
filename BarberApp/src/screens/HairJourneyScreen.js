import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Colors, Spacing, Radius } from '../theme/colors';
import { HAIR_JOURNEY } from '../data/mockData';

const { width } = Dimensions.get('window');

const STATS = [
  { label: 'Total Cuts', value: '14', icon: 'scissors-cutting', color: '#C8A96E' },
  { label: 'Months Active', value: '8', icon: 'calendar-month', color: '#5C9EE0' },
  { label: 'Styles Tried', value: '6', icon: 'palette', color: '#6C5CE7' },
  { label: 'Avg Rating', value: '4.8★', icon: 'star', color: '#F0C849' },
];

export default function HairJourneyScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const [activeView, setActiveView] = useState('timeline');

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <MaterialCommunityIcons name="arrow-left" size={22} color={Colors.textPrimary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Hair Journey</Text>
        <TouchableOpacity style={styles.addBtn}>
          <LinearGradient colors={Colors.gradientGold} style={styles.addBtnGrad}>
            <MaterialCommunityIcons name="camera-plus" size={18} color="#0A0A0F" />
          </LinearGradient>
        </TouchableOpacity>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        {/* Stats */}
        <View style={styles.statsGrid}>
          {STATS.map((stat) => (
            <View key={stat.label} style={styles.statCard}>
              <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.md} />
              <MaterialCommunityIcons name={stat.icon} size={20} color={stat.color} />
              <Text style={styles.statValue}>{stat.value}</Text>
              <Text style={styles.statLabel}>{stat.label}</Text>
            </View>
          ))}
        </View>

        {/* Transformation Showcase */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Your Transformation</Text>
          <View style={styles.beforeAfter}>
            <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.xl} />
            <View style={styles.beforeAfterItem}>
              <LinearGradient colors={['#1A1A2E', '#0A0A1E']} style={styles.photoPlaceholder} borderRadius={Radius.lg}>
                <Text style={styles.photoEmoji}>💈</Text>
                <Text style={styles.photoLabel}>Before</Text>
                <Text style={styles.photoSub}>Oct 2025</Text>
              </LinearGradient>
            </View>
            <View style={styles.vsCircle}>
              <LinearGradient colors={Colors.gradientGold} style={styles.vsCircleGrad}>
                <Text style={styles.vsText}>→</Text>
              </LinearGradient>
            </View>
            <View style={styles.beforeAfterItem}>
              <LinearGradient colors={['#1A1200', '#0F0D00']} style={styles.photoPlaceholder} borderRadius={Radius.lg}>
                <Text style={styles.photoEmoji}>✂️</Text>
                <Text style={styles.photoLabel}>Latest</Text>
                <Text style={styles.photoSub}>Jun 2026</Text>
              </LinearGradient>
            </View>
          </View>
        </View>

        {/* View Toggle */}
        <View style={styles.viewToggle}>
          {['timeline', 'grid'].map((v) => (
            <TouchableOpacity
              key={v}
              style={[styles.viewToggleBtn, activeView === v && styles.viewToggleBtnActive]}
              onPress={() => setActiveView(v)}
            >
              {activeView === v && (
                <LinearGradient colors={Colors.gradientGold} style={StyleSheet.absoluteFill} borderRadius={Radius.full} />
              )}
              <MaterialCommunityIcons
                name={v === 'timeline' ? 'timeline-text' : 'view-grid'}
                size={16}
                color={activeView === v ? '#0A0A0F' : Colors.textSecondary}
              />
              <Text style={[styles.viewToggleText, activeView === v && { color: '#0A0A0F' }]}>
                {v.charAt(0).toUpperCase() + v.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Timeline View */}
        {activeView === 'timeline' && (
          <View style={styles.timeline}>
            {HAIR_JOURNEY.map((item, index) => (
              <View key={item.id} style={styles.timelineItem}>
                {/* Timeline line */}
                {index < HAIR_JOURNEY.length - 1 && (
                  <View style={[styles.timelineLine, { backgroundColor: item.color + '30' }]} />
                )}

                {/* Dot */}
                <View style={[styles.timelineDot, { backgroundColor: item.color, borderColor: item.color + '40' }]}>
                  <MaterialCommunityIcons name="scissors-cutting" size={14} color="#FFF" />
                </View>

                {/* Card */}
                <View style={styles.timelineCard}>
                  <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.lg} />

                  {/* Photo Placeholder */}
                  <LinearGradient
                    colors={[item.color + '20', item.color + '10']}
                    style={styles.timelinePhoto}
                    borderRadius={Radius.md}
                  >
                    <Text style={styles.timelinePhotoEmoji}>✂️</Text>
                    <Text style={[styles.timelinePhotoStyle, { color: item.color }]}>{item.style}</Text>
                  </LinearGradient>

                  <View style={styles.timelineInfo}>
                    <Text style={styles.timelineStyle}>{item.style}</Text>
                    <Text style={styles.timelineDate}>{item.date}</Text>
                    <View style={styles.timelineBarber}>
                      <MaterialCommunityIcons name="account" size={12} color={Colors.textMuted} />
                      <Text style={styles.timelineBarberText}>by {item.barber}</Text>
                    </View>
                    <View style={styles.timelineStars}>
                      {Array(5).fill(0).map((_, i) => (
                        <MaterialCommunityIcons
                          key={i}
                          name={i < item.rating ? 'star' : 'star-outline'}
                          size={14}
                          color={Colors.primary}
                        />
                      ))}
                    </View>
                    <Text style={styles.timelineNote}>"{item.note}"</Text>
                  </View>
                </View>
              </View>
            ))}
          </View>
        )}

        {/* Grid View */}
        {activeView === 'grid' && (
          <View style={styles.gridView}>
            {HAIR_JOURNEY.map((item) => (
              <TouchableOpacity key={item.id} style={styles.gridItem} activeOpacity={0.85}>
                <LinearGradient
                  colors={[item.color + '30', item.color + '10']}
                  style={styles.gridItemGrad}
                >
                  <Text style={styles.gridEmoji}>✂️</Text>
                  <View style={styles.gridOverlay}>
                    <Text style={styles.gridStyle}>{item.style}</Text>
                    <Text style={styles.gridDate}>{item.date}</Text>
                  </View>
                </LinearGradient>
              </TouchableOpacity>
            ))}
          </View>
        )}

        {/* Add Photo CTA */}
        <TouchableOpacity style={styles.addPhotoCTA} activeOpacity={0.85}>
          <LinearGradient colors={['#1A1A2E', '#0F0F1E']} style={StyleSheet.absoluteFill} borderRadius={Radius.xl} />
          <MaterialCommunityIcons name="camera-plus-outline" size={32} color={Colors.textMuted} />
          <Text style={styles.addPhotoTitle}>Add Your Latest Look</Text>
          <Text style={styles.addPhotoSub}>
            Share your fresh cut and earn +50 XP!
          </Text>
          <View style={styles.addPhotoBtn}>
            <LinearGradient colors={Colors.gradientGold} style={styles.addPhotoBtnGrad} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}>
              <Text style={styles.addPhotoBtnText}>Take Photo</Text>
            </LinearGradient>
          </View>
        </TouchableOpacity>

        {/* Hair Health Score */}
        <View style={[styles.section, { marginBottom: 40 }]}>
          <Text style={styles.sectionTitle}>Hair Health Score</Text>
          <View style={styles.healthCard}>
            <LinearGradient colors={['#001A0A', '#000F05']} style={StyleSheet.absoluteFill} borderRadius={Radius.xl} />
            <View style={styles.healthScore}>
              <Text style={styles.healthScoreNum}>87</Text>
              <Text style={styles.healthScoreLabel}>/100</Text>
            </View>
            <Text style={styles.healthTitle}>Great Health!</Text>
            <Text style={styles.healthSub}>Based on your haircut frequency, scalp treatments, and product use</Text>
            <View style={styles.healthMetrics}>
              {[
                { label: 'Scalp Health', score: 9, color: Colors.success },
                { label: 'Growth Rate', score: 7, color: Colors.primary },
                { label: 'Moisture', score: 8, color: Colors.accentCool },
              ].map((m) => (
                <View key={m.label} style={styles.healthMetric}>
                  <View style={styles.healthMetricBar}>
                    <View style={[styles.healthMetricFill, { width: `${m.score * 10}%`, backgroundColor: m.color }]} />
                  </View>
                  <Text style={styles.healthMetricLabel}>{m.label}</Text>
                  <Text style={[styles.healthMetricScore, { color: m.color }]}>{m.score}/10</Text>
                </View>
              ))}
            </View>
          </View>
        </View>
      </ScrollView>
    </View>
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
  addBtn: { width: 40, height: 40, borderRadius: 20, overflow: 'hidden' },
  addBtnGrad: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  scrollContent: { paddingBottom: 40 },
  statsGrid: {
    flexDirection: 'row',
    paddingHorizontal: Spacing.md,
    gap: 8,
    marginBottom: Spacing.lg,
  },
  statCard: {
    flex: 1,
    borderRadius: Radius.md,
    padding: 10,
    alignItems: 'center',
    gap: 4,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
  },
  statValue: { color: Colors.textPrimary, fontSize: 16, fontWeight: '900' },
  statLabel: { color: Colors.textMuted, fontSize: 9, textAlign: 'center' },
  section: { paddingHorizontal: Spacing.md, marginBottom: Spacing.lg },
  sectionTitle: { color: Colors.textPrimary, fontSize: 18, fontWeight: '700', marginBottom: 12 },
  beforeAfter: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: Radius.xl,
    padding: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
    gap: 10,
  },
  beforeAfterItem: { flex: 1 },
  photoPlaceholder: {
    height: 140,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  photoEmoji: { fontSize: 40 },
  photoLabel: { color: Colors.textPrimary, fontSize: 16, fontWeight: '700' },
  photoSub: { color: Colors.textMuted, fontSize: 11 },
  vsCircle: { width: 36, height: 36, borderRadius: 18, overflow: 'hidden' },
  vsCircleGrad: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  vsText: { color: '#0A0A0F', fontSize: 18, fontWeight: '900' },
  viewToggle: {
    flexDirection: 'row',
    marginHorizontal: Spacing.md,
    backgroundColor: Colors.bgCardAlt,
    borderRadius: Radius.full,
    padding: 4,
    marginBottom: Spacing.md,
  },
  viewToggleBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    borderRadius: Radius.full,
    overflow: 'hidden',
    gap: 6,
  },
  viewToggleBtnActive: {},
  viewToggleText: { color: Colors.textSecondary, fontSize: 13, fontWeight: '700' },
  timeline: { paddingHorizontal: Spacing.md },
  timelineItem: { flexDirection: 'row', gap: 12, marginBottom: Spacing.lg, position: 'relative' },
  timelineLine: {
    position: 'absolute',
    left: 19,
    top: 32,
    width: 2,
    height: '100%',
    zIndex: 0,
  },
  timelineDot: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    zIndex: 1,
    flexShrink: 0,
  },
  timelineCard: {
    flex: 1,
    borderRadius: Radius.lg,
    padding: Spacing.md,
    flexDirection: 'row',
    gap: 12,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
  },
  timelinePhoto: {
    width: 80,
    height: 80,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
  },
  timelinePhotoEmoji: { fontSize: 28 },
  timelinePhotoStyle: { fontSize: 9, fontWeight: '700', textAlign: 'center' },
  timelineInfo: { flex: 1 },
  timelineStyle: { color: Colors.textPrimary, fontSize: 14, fontWeight: '700', marginBottom: 2 },
  timelineDate: { color: Colors.textMuted, fontSize: 11, marginBottom: 4 },
  timelineBarber: { flexDirection: 'row', alignItems: 'center', gap: 4, marginBottom: 6 },
  timelineBarberText: { color: Colors.textMuted, fontSize: 11 },
  timelineStars: { flexDirection: 'row', gap: 1, marginBottom: 6 },
  timelineNote: { color: Colors.textSecondary, fontSize: 11, fontStyle: 'italic' },
  gridView: { flexDirection: 'row', flexWrap: 'wrap', paddingHorizontal: Spacing.md, gap: 8, marginBottom: Spacing.lg },
  gridItem: { width: (width - Spacing.md * 2 - 8) / 2, height: 140, borderRadius: Radius.lg, overflow: 'hidden' },
  gridItemGrad: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  gridEmoji: { fontSize: 40, marginBottom: 8 },
  gridOverlay: { alignItems: 'center' },
  gridStyle: { color: Colors.textPrimary, fontSize: 12, fontWeight: '700' },
  gridDate: { color: Colors.textMuted, fontSize: 10 },
  addPhotoCTA: {
    marginHorizontal: Spacing.md,
    borderRadius: Radius.xl,
    padding: Spacing.xl,
    alignItems: 'center',
    gap: 10,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    borderStyle: 'dashed',
    overflow: 'hidden',
    marginBottom: Spacing.lg,
  },
  addPhotoTitle: { color: Colors.textPrimary, fontSize: 18, fontWeight: '700' },
  addPhotoSub: { color: Colors.textSecondary, fontSize: 13, textAlign: 'center' },
  addPhotoBtn: { borderRadius: Radius.full, overflow: 'hidden', marginTop: 4 },
  addPhotoBtnGrad: { paddingHorizontal: 28, paddingVertical: 12 },
  addPhotoBtnText: { color: '#0A0A0F', fontSize: 14, fontWeight: '800' },
  healthCard: {
    borderRadius: Radius.xl,
    padding: Spacing.lg,
    alignItems: 'center',
    gap: Spacing.sm,
    borderWidth: 1,
    borderColor: Colors.success + '30',
    overflow: 'hidden',
  },
  healthScore: { flexDirection: 'row', alignItems: 'flex-end', gap: 4 },
  healthScoreNum: { color: Colors.success, fontSize: 56, fontWeight: '900' },
  healthScoreLabel: { color: Colors.textMuted, fontSize: 16, marginBottom: 14 },
  healthTitle: { color: Colors.textPrimary, fontSize: 20, fontWeight: '800' },
  healthSub: { color: Colors.textSecondary, fontSize: 13, textAlign: 'center', lineHeight: 20, marginBottom: 8 },
  healthMetrics: { width: '100%', gap: 10 },
  healthMetric: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  healthMetricBar: { flex: 1, height: 6, backgroundColor: Colors.bgGlass, borderRadius: 3, overflow: 'hidden' },
  healthMetricFill: { height: '100%', borderRadius: 3 },
  healthMetricLabel: { color: Colors.textSecondary, fontSize: 12, width: 80 },
  healthMetricScore: { fontSize: 12, fontWeight: '700', width: 32, textAlign: 'right' },
});
