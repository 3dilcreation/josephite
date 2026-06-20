import React, { useState, useEffect, useRef } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Colors, Spacing, Radius } from '../theme/colors';
import { QUEUE_DATA } from '../data/mockData';

export default function QueueScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const [waitTime, setWaitTime] = useState(QUEUE_DATA.currentWait);
  const [checkedIn, setCheckedIn] = useState(false);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const scanAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.08, duration: 800, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
      ])
    ).start();

    Animated.loop(
      Animated.timing(scanAnim, { toValue: 1, duration: 2000, useNativeDriver: true })
    ).start();
  }, []);

  const BARBERS = QUEUE_DATA.barbers;

  const statusColor = (status) => {
    if (status === 'available') return Colors.success;
    if (status === 'busy') return Colors.warning;
    return Colors.textMuted;
  };

  const statusLabel = (b) => {
    if (b.status === 'available') return 'Available Now';
    if (b.status === 'busy') return `Done in ~${b.eta}m`;
    return 'On Break';
  };

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <MaterialCommunityIcons name="arrow-left" size={22} color={Colors.textPrimary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Live Queue</Text>
        <View style={styles.liveDot}>
          <Animated.View style={[styles.livePulse, { transform: [{ scale: pulseAnim }] }]} />
          <Text style={styles.liveText}>LIVE</Text>
        </View>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        {/* Main Wait Card */}
        <LinearGradient colors={['#1A1200', '#0A0A0F']} style={styles.waitCard}>
          {checkedIn ? (
            <>
              <View style={styles.positionDisplay}>
                <Text style={styles.positionLabel}>You are</Text>
                <Text style={styles.positionNumber}>#{QUEUE_DATA.position}</Text>
                <Text style={styles.positionSub}>in queue</Text>
              </View>

              <View style={styles.waitTimeDisplay}>
                <Animated.View style={[styles.waitCircle, { transform: [{ scale: pulseAnim }] }]}>
                  <LinearGradient colors={Colors.gradientGold} style={styles.waitCircleGrad}>
                    <Text style={styles.waitMinutes}>{waitTime}</Text>
                    <Text style={styles.waitMinsLabel}>MIN</Text>
                  </LinearGradient>
                </Animated.View>
                <Text style={styles.waitETA}>Est. ready at {QUEUE_DATA.estimatedTime}</Text>
              </View>

              <View style={styles.queueInfo}>
                <View style={styles.queueInfoItem}>
                  <MaterialCommunityIcons name="account-group" size={16} color={Colors.textSecondary} />
                  <Text style={styles.queueInfoText}>{QUEUE_DATA.totalInQueue} people ahead</Text>
                </View>
                <View style={styles.queueInfoItem}>
                  <MaterialCommunityIcons name="refresh" size={16} color={Colors.textSecondary} />
                  <Text style={styles.queueInfoText}>Updates every 30s</Text>
                </View>
              </View>

              <TouchableOpacity
                style={styles.leaveQueueBtn}
                onPress={() => setCheckedIn(false)}
                activeOpacity={0.8}
              >
                <Text style={styles.leaveQueueText}>Leave Queue</Text>
              </TouchableOpacity>
            </>
          ) : (
            <>
              <Text style={styles.checkInTitle}>Walk-In Queue</Text>
              <Text style={styles.checkInSub}>
                Check in now and we'll notify you when your barber is ready — no waiting at the shop!
              </Text>

              <View style={styles.checkInStats}>
                <View style={styles.checkInStat}>
                  <Text style={styles.checkInStatVal}>{QUEUE_DATA.totalInQueue}</Text>
                  <Text style={styles.checkInStatLabel}>In Queue</Text>
                </View>
                <View style={styles.checkInStatDivider} />
                <View style={styles.checkInStat}>
                  <Text style={styles.checkInStatVal}>{waitTime}m</Text>
                  <Text style={styles.checkInStatLabel}>Est. Wait</Text>
                </View>
                <View style={styles.checkInStatDivider} />
                <View style={styles.checkInStat}>
                  <Text style={styles.checkInStatVal}>
                    {BARBERS.filter(b => b.status === 'available').length}
                  </Text>
                  <Text style={styles.checkInStatLabel}>Available</Text>
                </View>
              </View>

              {/* QR Check In */}
              <View style={styles.qrArea}>
                <View style={styles.qrCode}>
                  {/* Simulated QR */}
                  <View style={styles.qrGrid}>
                    {Array(25).fill(0).map((_, i) => (
                      <View
                        key={i}
                        style={[
                          styles.qrCell,
                          Math.random() > 0.5 && styles.qrCellFilled,
                        ]}
                      />
                    ))}
                  </View>
                </View>
                <Text style={styles.qrLabel}>Scan at shop to check in</Text>
                <Text style={styles.qrOr}>— or —</Text>
              </View>

              <TouchableOpacity
                style={styles.checkInBtn}
                onPress={() => setCheckedIn(true)}
                activeOpacity={0.85}
              >
                <LinearGradient
                  colors={Colors.gradientGold}
                  style={styles.checkInBtnGrad}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                >
                  <MaterialCommunityIcons name="qrcode-scan" size={20} color="#0A0A0F" />
                  <Text style={styles.checkInBtnText}>Join Queue (Remote)</Text>
                </LinearGradient>
              </TouchableOpacity>
            </>
          )}
        </LinearGradient>

        {/* Barber Status Grid */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Barber Status</Text>
          {BARBERS.map((barber, i) => (
            <View key={i} style={styles.barberStatusCard}>
              <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.md} />
              <View style={[styles.barberStatusDot, { backgroundColor: statusColor(barber.status) }]} />
              <View style={styles.barberStatusAvatar}>
                <Text style={styles.barberStatusInitial}>{barber.name[0]}</Text>
              </View>
              <View style={styles.barberStatusInfo}>
                <Text style={styles.barberStatusName}>{barber.name}</Text>
                <Text style={[styles.barberStatusLabel, { color: statusColor(barber.status) }]}>
                  {statusLabel(barber)}
                </Text>
                {barber.client && (
                  <Text style={styles.barberStatusClient}>Serving: {barber.client}</Text>
                )}
              </View>
              {barber.status === 'available' && (
                <TouchableOpacity style={styles.grabBtn} onPress={() => setCheckedIn(true)}>
                  <LinearGradient colors={Colors.gradientGold} style={styles.grabBtnGrad}>
                    <Text style={styles.grabBtnText}>Grab Slot</Text>
                  </LinearGradient>
                </TouchableOpacity>
              )}
            </View>
          ))}
        </View>

        {/* AI Wait Prediction */}
        <View style={[styles.section, { marginBottom: 40 }]}>
          <View style={styles.aiPrediction}>
            <LinearGradient colors={['#0A001A', '#050012']} style={StyleSheet.absoluteFill} borderRadius={Radius.lg} />
            <View style={styles.aiPredictionHeader}>
              <MaterialCommunityIcons name="brain" size={20} color="#6C5CE7" />
              <Text style={styles.aiPredictionTitle}>AI Queue Prediction</Text>
            </View>
            <Text style={styles.aiPredictionText}>
              Based on historical patterns, the best times to walk in today are:
            </Text>
            <View style={styles.aiTimes}>
              {['9:00-9:30 AM', '12:30-1:00 PM', '4:30-5:00 PM'].map((time) => (
                <View key={time} style={styles.aiTimeChip}>
                  <MaterialCommunityIcons name="clock-fast" size={12} color={Colors.success} />
                  <Text style={styles.aiTimeText}>{time}</Text>
                </View>
              ))}
            </View>
            <Text style={styles.aiPredictionNote}>
              Weekend afternoons (2-4PM) are typically 40% busier — plan accordingly!
            </Text>
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
  liveDot: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: Colors.error + '20',
    borderRadius: Radius.full,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: Colors.error + '40',
  },
  livePulse: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.error,
  },
  liveText: { color: Colors.error, fontSize: 11, fontWeight: '800', letterSpacing: 1 },
  scrollContent: { paddingBottom: 40 },
  waitCard: {
    margin: Spacing.md,
    borderRadius: Radius.xl,
    padding: Spacing.xl,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  positionDisplay: { alignItems: 'center', marginBottom: Spacing.lg },
  positionLabel: { color: Colors.textSecondary, fontSize: 14 },
  positionNumber: { color: Colors.primary, fontSize: 72, fontWeight: '900', lineHeight: 80 },
  positionSub: { color: Colors.textSecondary, fontSize: 16 },
  waitTimeDisplay: { alignItems: 'center', marginBottom: Spacing.lg },
  waitCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    overflow: 'hidden',
    marginBottom: 12,
  },
  waitCircleGrad: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  waitMinutes: { color: '#0A0A0F', fontSize: 40, fontWeight: '900' },
  waitMinsLabel: { color: '#0A0A0F80', fontSize: 12, fontWeight: '700' },
  waitETA: { color: Colors.textSecondary, fontSize: 14 },
  queueInfo: { flexDirection: 'row', gap: Spacing.lg, marginBottom: Spacing.lg },
  queueInfoItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  queueInfoText: { color: Colors.textSecondary, fontSize: 13 },
  leaveQueueBtn: {
    paddingVertical: 10,
    paddingHorizontal: 24,
    borderRadius: Radius.full,
    borderWidth: 1,
    borderColor: Colors.error + '40',
  },
  leaveQueueText: { color: Colors.error, fontSize: 14, fontWeight: '600' },
  checkInTitle: { color: Colors.textPrimary, fontSize: 26, fontWeight: '800', marginBottom: 8 },
  checkInSub: { color: Colors.textSecondary, fontSize: 14, textAlign: 'center', lineHeight: 22, marginBottom: Spacing.xl },
  checkInStats: { flexDirection: 'row', alignItems: 'center', marginBottom: Spacing.xl },
  checkInStat: { flex: 1, alignItems: 'center' },
  checkInStatVal: { color: Colors.textPrimary, fontSize: 28, fontWeight: '900', marginBottom: 4 },
  checkInStatLabel: { color: Colors.textMuted, fontSize: 12 },
  checkInStatDivider: { width: 1, height: 40, backgroundColor: Colors.borderSubtle },
  qrArea: { alignItems: 'center', marginBottom: Spacing.lg },
  qrCode: {
    width: 120,
    height: 120,
    backgroundColor: Colors.bgCardAlt,
    borderRadius: Radius.md,
    padding: 10,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  qrGrid: { flex: 1, flexDirection: 'row', flexWrap: 'wrap' },
  qrCell: { width: '20%', aspectRatio: 1 },
  qrCellFilled: { backgroundColor: Colors.textPrimary },
  qrLabel: { color: Colors.textSecondary, fontSize: 12, marginBottom: 8 },
  qrOr: { color: Colors.textMuted, fontSize: 12 },
  checkInBtn: { width: '100%', borderRadius: Radius.full, overflow: 'hidden', marginTop: 8 },
  checkInBtnGrad: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 16,
  },
  checkInBtnText: { color: '#0A0A0F', fontSize: 16, fontWeight: '800' },
  section: { paddingHorizontal: Spacing.md, marginBottom: Spacing.md },
  sectionTitle: { color: Colors.textPrimary, fontSize: 18, fontWeight: '700', marginBottom: 12 },
  barberStatusCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    borderRadius: Radius.md,
    padding: Spacing.md,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
    position: 'relative',
  },
  barberStatusDot: {
    position: 'absolute',
    top: 12,
    right: 12,
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  barberStatusAvatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: Colors.primary + '20',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: Colors.border,
  },
  barberStatusInitial: { color: Colors.primary, fontSize: 18, fontWeight: '800' },
  barberStatusInfo: { flex: 1 },
  barberStatusName: { color: Colors.textPrimary, fontSize: 15, fontWeight: '700', marginBottom: 2 },
  barberStatusLabel: { fontSize: 12, fontWeight: '700', marginBottom: 2 },
  barberStatusClient: { color: Colors.textMuted, fontSize: 11 },
  grabBtn: { borderRadius: Radius.full, overflow: 'hidden' },
  grabBtnGrad: { paddingHorizontal: 16, paddingVertical: 8 },
  grabBtnText: { color: '#0A0A0F', fontSize: 12, fontWeight: '800' },
  aiPrediction: {
    borderRadius: Radius.lg,
    padding: Spacing.lg,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#6C5CE730',
  },
  aiPredictionHeader: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 },
  aiPredictionTitle: { color: Colors.textPrimary, fontSize: 16, fontWeight: '700' },
  aiPredictionText: { color: Colors.textSecondary, fontSize: 13, lineHeight: 20, marginBottom: 12 },
  aiTimes: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 12 },
  aiTimeChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: Colors.success + '15',
    borderRadius: Radius.full,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderWidth: 1,
    borderColor: Colors.success + '30',
  },
  aiTimeText: { color: Colors.success, fontSize: 12, fontWeight: '600' },
  aiPredictionNote: { color: Colors.textMuted, fontSize: 12, lineHeight: 18, fontStyle: 'italic' },
});
