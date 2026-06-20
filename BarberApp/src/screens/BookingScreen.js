import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  Dimensions, Image,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, Spacing, Radius } from '../theme/colors';
import { BARBERS, SERVICES } from '../data/mockData';

const { width } = Dimensions.get('window');

const DAYS = [
  { day: 'Mon', date: '23', available: true },
  { day: 'Tue', date: '24', available: true },
  { day: 'Wed', date: '25', available: true },
  { day: 'Thu', date: '26', available: false },
  { day: 'Fri', date: '27', available: true },
  { day: 'Sat', date: '28', available: true },
  { day: 'Sun', date: '29', available: false },
];

const TIME_SLOTS = [
  { time: '9:00 AM', peak: false, discount: 0 },
  { time: '9:30 AM', peak: false, discount: 15 },
  { time: '10:00 AM', peak: true, discount: 0 },
  { time: '10:30 AM', peak: true, discount: 0 },
  { time: '11:00 AM', peak: true, discount: 0 },
  { time: '11:30 AM', peak: true, discount: 0 },
  { time: '12:00 PM', peak: false, discount: 10 },
  { time: '12:30 PM', peak: false, discount: 10 },
  { time: '2:00 PM', peak: true, discount: 0 },
  { time: '2:30 PM', peak: true, discount: 0 },
  { time: '4:00 PM', peak: false, discount: 0 },
  { time: '5:00 PM', peak: true, discount: 0 },
];

const STEPS = ['Service', 'Barber', 'Time', 'Confirm'];

export default function BookingScreen() {
  const insets = useSafeAreaInsets();
  const [step, setStep] = useState(0);
  const [selectedService, setSelectedService] = useState(null);
  const [selectedBarber, setSelectedBarber] = useState(null);
  const [selectedDay, setSelectedDay] = useState(0);
  const [selectedTime, setSelectedTime] = useState(null);
  const [groupBooking, setGroupBooking] = useState(false);
  const [confirmed, setConfirmed] = useState(false);

  const canNext = () => {
    if (step === 0) return !!selectedService;
    if (step === 1) return !!selectedBarber;
    if (step === 2) return selectedDay !== null && !!selectedTime;
    return true;
  };

  if (confirmed) {
    return (
      <View style={[styles.container, { paddingTop: insets.top, alignItems: 'center', justifyContent: 'center', paddingHorizontal: Spacing.xl }]}>
        <LinearGradient colors={Colors.gradientGold} style={styles.confirmIcon}>
          <MaterialCommunityIcons name="check" size={48} color="#0A0A0F" />
        </LinearGradient>
        <Text style={styles.confirmTitle}>You're booked!</Text>
        <Text style={styles.confirmSub}>
          {selectedService?.name} with {selectedBarber?.name.split(' ')[0]}{'\n'}
          {DAYS[selectedDay].day} Jun {DAYS[selectedDay].date} · {selectedTime}
        </Text>
        <View style={styles.confirmCard}>
          <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.lg} />
          <View style={styles.confirmRow}>
            <MaterialCommunityIcons name="map-marker" size={18} color={Colors.primary} />
            <Text style={styles.confirmRowText}>FadeBlades Studio · 123 Main St</Text>
          </View>
          <View style={styles.confirmRow}>
            <MaterialCommunityIcons name="cash" size={18} color={Colors.primary} />
            <Text style={styles.confirmRowText}>${selectedService?.price} · Paid on arrival</Text>
          </View>
          <View style={styles.confirmRow}>
            <MaterialCommunityIcons name="message" size={18} color={Colors.primary} />
            <Text style={styles.confirmRowText}>Message your barber before</Text>
          </View>
        </View>
        <TouchableOpacity style={styles.doneBtn} onPress={() => { setConfirmed(false); setStep(0); setSelectedService(null); setSelectedBarber(null); setSelectedTime(null); }}>
          <LinearGradient colors={Colors.gradientGold} style={styles.doneBtnGrad} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}>
            <Text style={styles.doneBtnText}>Done</Text>
          </LinearGradient>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Book Appointment</Text>
        <TouchableOpacity
          style={styles.groupToggle}
          onPress={() => setGroupBooking(!groupBooking)}
        >
          <MaterialCommunityIcons
            name={groupBooking ? 'account-group' : 'account-group-outline'}
            size={20}
            color={groupBooking ? Colors.primary : Colors.textSecondary}
          />
          <Text style={[styles.groupText, groupBooking && { color: Colors.primary }]}>Group</Text>
        </TouchableOpacity>
      </View>

      {/* Step Indicator */}
      <View style={styles.stepBar}>
        {STEPS.map((s, i) => (
          <React.Fragment key={s}>
            <View style={styles.stepItem}>
              <View style={[
                styles.stepDot,
                i < step && styles.stepDotDone,
                i === step && styles.stepDotActive,
              ]}>
                {i < step
                  ? <MaterialCommunityIcons name="check" size={14} color="#0A0A0F" />
                  : <Text style={[styles.stepNum, i === step && { color: '#0A0A0F' }]}>{i + 1}</Text>
                }
              </View>
              <Text style={[styles.stepLabel, i === step && styles.stepLabelActive]}>{s}</Text>
            </View>
            {i < STEPS.length - 1 && (
              <View style={[styles.stepLine, i < step && styles.stepLineDone]} />
            )}
          </React.Fragment>
        ))}
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {/* Step 0: Service */}
        {step === 0 && (
          <View>
            <Text style={styles.stepTitle}>Choose a Service</Text>
            {SERVICES.map((service) => (
              <TouchableOpacity
                key={service.id}
                style={[styles.serviceCard, selectedService?.id === service.id && styles.serviceCardSelected]}
                onPress={() => setSelectedService(service)}
                activeOpacity={0.85}
              >
                <LinearGradient
                  colors={selectedService?.id === service.id ? ['#1A1200', '#120D00'] : Colors.gradientCard}
                  style={StyleSheet.absoluteFill}
                  borderRadius={Radius.md}
                />
                <View style={styles.serviceLeft}>
                  <View style={[
                    styles.serviceIconBg,
                    selectedService?.id === service.id && { backgroundColor: Colors.primary + '20' }
                  ]}>
                    <MaterialCommunityIcons
                      name={service.icon}
                      size={24}
                      color={selectedService?.id === service.id ? Colors.primary : Colors.textSecondary}
                    />
                  </View>
                  <View>
                    <View style={styles.serviceNameRow}>
                      <Text style={styles.serviceName}>{service.name}</Text>
                      {service.popular && (
                        <View style={styles.popularBadge}>
                          <Text style={styles.popularText}>Popular</Text>
                        </View>
                      )}
                      {service.aiEnhanced && (
                        <View style={styles.aiBadge}>
                          <MaterialCommunityIcons name="star-four-points" size={10} color="#6C5CE7" />
                          <Text style={styles.aiText}>AI</Text>
                        </View>
                      )}
                    </View>
                    <Text style={styles.serviceDesc} numberOfLines={2}>{service.description}</Text>
                    <View style={styles.serviceMeta}>
                      <MaterialCommunityIcons name="clock-outline" size={12} color={Colors.textMuted} />
                      <Text style={styles.serviceMetaText}>{service.duration} min</Text>
                    </View>
                  </View>
                </View>
                <View style={styles.servicePrice}>
                  <Text style={styles.servicePriceVal}>${service.price}</Text>
                  {selectedService?.id === service.id && (
                    <MaterialCommunityIcons name="check-circle" size={22} color={Colors.primary} />
                  )}
                </View>
              </TouchableOpacity>
            ))}
          </View>
        )}

        {/* Step 1: Barber */}
        {step === 1 && (
          <View>
            <Text style={styles.stepTitle}>Choose Your Barber</Text>
            {BARBERS.map((barber) => (
              <TouchableOpacity
                key={barber.id}
                style={[styles.barberCard, selectedBarber?.id === barber.id && styles.barberCardSelected]}
                onPress={() => setSelectedBarber(barber)}
                activeOpacity={0.85}
              >
                <LinearGradient
                  colors={selectedBarber?.id === barber.id ? ['#1A1200', '#120D00'] : Colors.gradientCard}
                  style={StyleSheet.absoluteFill}
                  borderRadius={Radius.md}
                />
                <Image source={{ uri: barber.avatar }} style={styles.barberAvatar} />
                <View style={styles.barberDetails}>
                  <Text style={styles.barberName}>{barber.name}</Text>
                  <Text style={styles.barberSpecialty}>{barber.specialty}</Text>
                  <View style={styles.barberRatingRow}>
                    <MaterialCommunityIcons name="star" size={13} color={Colors.primary} />
                    <Text style={styles.barberRatingText}>{barber.rating} · {barber.reviews} reviews</Text>
                  </View>
                  <Text style={styles.barberExp}>{barber.experience} experience</Text>
                </View>
                <View style={styles.barberRight}>
                  <Text style={styles.barberPrice}>${barber.price}</Text>
                  {!barber.available && (
                    <View style={styles.waitBadge}>
                      <Text style={styles.waitText}>{barber.waitTime}m wait</Text>
                    </View>
                  )}
                  {selectedBarber?.id === barber.id && (
                    <MaterialCommunityIcons name="check-circle" size={22} color={Colors.primary} style={{ marginTop: 8 }} />
                  )}
                </View>
              </TouchableOpacity>
            ))}
          </View>
        )}

        {/* Step 2: Time */}
        {step === 2 && (
          <View>
            <Text style={styles.stepTitle}>Pick a Time</Text>

            {/* Dynamic Pricing Info */}
            <View style={styles.pricingInfo}>
              <MaterialCommunityIcons name="information" size={16} color={Colors.accentCool} />
              <Text style={styles.pricingText}>
                Off-peak slots save you up to 15% — smart scheduling!
              </Text>
            </View>

            {/* Days */}
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.daysScroll}>
              {DAYS.map((d, i) => (
                <TouchableOpacity
                  key={i}
                  style={[
                    styles.dayChip,
                    !d.available && styles.dayChipDisabled,
                    selectedDay === i && styles.dayChipSelected,
                  ]}
                  onPress={() => d.available && setSelectedDay(i)}
                  activeOpacity={0.8}
                >
                  {selectedDay === i && (
                    <LinearGradient
                      colors={Colors.gradientGold}
                      style={StyleSheet.absoluteFill}
                      borderRadius={Radius.md}
                    />
                  )}
                  <Text style={[styles.dayName, selectedDay === i && { color: '#0A0A0F' }]}>{d.day}</Text>
                  <Text style={[styles.dayDate, selectedDay === i && { color: '#0A0A0F' }]}>{d.date}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>

            {/* Time Slots */}
            <View style={styles.timesGrid}>
              {TIME_SLOTS.map((slot, i) => (
                <TouchableOpacity
                  key={i}
                  style={[
                    styles.timeSlot,
                    selectedTime === slot.time && styles.timeSlotSelected,
                    slot.peak && styles.timeSlotPeak,
                  ]}
                  onPress={() => setSelectedTime(slot.time)}
                  activeOpacity={0.8}
                >
                  {selectedTime === slot.time && (
                    <LinearGradient colors={Colors.gradientGold} style={StyleSheet.absoluteFill} borderRadius={Radius.sm} />
                  )}
                  <Text style={[styles.timeText, selectedTime === slot.time && { color: '#0A0A0F' }]}>
                    {slot.time}
                  </Text>
                  {slot.discount > 0 && (
                    <View style={styles.discountPill}>
                      <Text style={styles.discountText}>-{slot.discount}%</Text>
                    </View>
                  )}
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}

        {/* Step 3: Confirm */}
        {step === 3 && (
          <View>
            <Text style={styles.stepTitle}>Review & Confirm</Text>
            <View style={styles.summaryCard}>
              <LinearGradient colors={Colors.gradientCard} style={StyleSheet.absoluteFill} borderRadius={Radius.lg} />
              <SummaryRow icon="cut" label="Service" value={selectedService?.name} />
              <SummaryRow icon="account" label="Barber" value={selectedBarber?.name.split('"')[0].trim()} />
              <SummaryRow icon="calendar" label="Date" value={`Jun ${DAYS[selectedDay].date} (${DAYS[selectedDay].day})`} />
              <SummaryRow icon="clock" label="Time" value={selectedTime} />
              <SummaryRow icon="timer" label="Duration" value={`~${selectedService?.duration} min`} />
              <View style={styles.divider} />
              <View style={styles.totalRow}>
                <Text style={styles.totalLabel}>Total</Text>
                <Text style={styles.totalValue}>${selectedService?.price}</Text>
              </View>
            </View>

            <View style={styles.addOnsSection}>
              <Text style={styles.addOnsTitle}>Add-ons</Text>
              {[{ name: 'Hair Design (+$15)', icon: 'palette' }, { name: 'Hot Towel (+$8)', icon: 'water' }].map((a) => (
                <TouchableOpacity key={a.name} style={styles.addOnRow}>
                  <MaterialCommunityIcons name={a.icon} size={18} color={Colors.textSecondary} />
                  <Text style={styles.addOnText}>{a.name}</Text>
                  <MaterialCommunityIcons name="plus-circle-outline" size={20} color={Colors.primary} />
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}
      </ScrollView>

      {/* Next / Confirm Button */}
      <View style={[styles.footer, { paddingBottom: insets.bottom + Spacing.md }]}>
        {step > 0 && (
          <TouchableOpacity style={styles.backBtn} onPress={() => setStep(step - 1)}>
            <MaterialCommunityIcons name="arrow-left" size={20} color={Colors.textSecondary} />
          </TouchableOpacity>
        )}
        <TouchableOpacity
          style={[styles.nextBtn, !canNext() && styles.nextBtnDisabled]}
          onPress={() => {
            if (!canNext()) return;
            if (step < 3) setStep(step + 1);
            else setConfirmed(true);
          }}
          activeOpacity={0.85}
        >
          <LinearGradient
            colors={canNext() ? Colors.gradientGold : [Colors.textMuted, Colors.textMuted]}
            style={styles.nextBtnGrad}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <Text style={styles.nextBtnText}>{step < 3 ? 'Continue' : 'Confirm Booking'}</Text>
            <MaterialCommunityIcons name={step < 3 ? 'arrow-right' : 'check'} size={18} color="#0A0A0F" />
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
}

function SummaryRow({ icon, label, value }) {
  return (
    <View style={styles.summaryRow}>
      <MaterialCommunityIcons name={icon} size={16} color={Colors.primary} />
      <Text style={styles.summaryLabel}>{label}</Text>
      <Text style={styles.summaryValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.md,
  },
  headerTitle: { color: Colors.textPrimary, fontSize: 22, fontWeight: '800' },
  groupToggle: { flexDirection: 'row', alignItems: 'center', gap: 6, padding: 8 },
  groupText: { color: Colors.textSecondary, fontSize: 13, fontWeight: '600' },
  stepBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: Spacing.lg,
    marginBottom: Spacing.md,
  },
  stepItem: { alignItems: 'center', gap: 4 },
  stepDot: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: Colors.bgCardAlt,
    borderWidth: 2,
    borderColor: Colors.borderSubtle,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepDotActive: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  stepDotDone: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  stepNum: { color: Colors.textMuted, fontSize: 12, fontWeight: '700' },
  stepLabel: { color: Colors.textMuted, fontSize: 10, fontWeight: '600' },
  stepLabelActive: { color: Colors.primary },
  stepLine: { flex: 1, height: 2, backgroundColor: Colors.borderSubtle, marginBottom: 18, marginHorizontal: 4 },
  stepLineDone: { backgroundColor: Colors.primary },
  scrollContent: { padding: Spacing.md, paddingBottom: 120 },
  stepTitle: { color: Colors.textPrimary, fontSize: 20, fontWeight: '700', marginBottom: 16 },
  serviceCard: {
    borderRadius: Radius.md,
    padding: Spacing.md,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    flexDirection: 'row',
    alignItems: 'center',
    overflow: 'hidden',
  },
  serviceCardSelected: { borderColor: Colors.primary },
  serviceLeft: { flex: 1, flexDirection: 'row', alignItems: 'flex-start', gap: 12 },
  serviceIconBg: {
    width: 44,
    height: 44,
    borderRadius: Radius.sm,
    backgroundColor: Colors.bgGlass,
    alignItems: 'center',
    justifyContent: 'center',
  },
  serviceNameRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 4 },
  serviceName: { color: Colors.textPrimary, fontSize: 15, fontWeight: '700' },
  popularBadge: { backgroundColor: Colors.primary + '20', borderRadius: Radius.full, paddingHorizontal: 8, paddingVertical: 2 },
  popularText: { color: Colors.primary, fontSize: 9, fontWeight: '700' },
  aiBadge: { flexDirection: 'row', alignItems: 'center', gap: 3, backgroundColor: '#6C5CE720', borderRadius: Radius.full, paddingHorizontal: 6, paddingVertical: 2 },
  aiText: { color: '#6C5CE7', fontSize: 9, fontWeight: '700' },
  serviceDesc: { color: Colors.textSecondary, fontSize: 12, lineHeight: 18, marginBottom: 6, maxWidth: 200 },
  serviceMeta: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  serviceMetaText: { color: Colors.textMuted, fontSize: 11 },
  servicePrice: { alignItems: 'flex-end', gap: 6 },
  servicePriceVal: { color: Colors.primary, fontSize: 18, fontWeight: '800' },
  barberCard: {
    borderRadius: Radius.md,
    padding: Spacing.md,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    overflow: 'hidden',
  },
  barberCardSelected: { borderColor: Colors.primary },
  barberAvatar: { width: 60, height: 60, borderRadius: 30, borderWidth: 2, borderColor: Colors.border },
  barberDetails: { flex: 1 },
  barberName: { color: Colors.textPrimary, fontSize: 14, fontWeight: '700', marginBottom: 2 },
  barberSpecialty: { color: Colors.textSecondary, fontSize: 12, marginBottom: 4 },
  barberRatingRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginBottom: 2 },
  barberRatingText: { color: Colors.textSecondary, fontSize: 12 },
  barberExp: { color: Colors.textMuted, fontSize: 11 },
  barberRight: { alignItems: 'flex-end' },
  barberPrice: { color: Colors.primary, fontSize: 16, fontWeight: '800' },
  waitBadge: { backgroundColor: Colors.warning + '20', borderRadius: Radius.full, paddingHorizontal: 8, paddingVertical: 2, marginTop: 4 },
  waitText: { color: Colors.warning, fontSize: 10, fontWeight: '700' },
  pricingInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: Colors.accentCool + '15',
    borderRadius: Radius.md,
    padding: Spacing.sm + 4,
    marginBottom: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.accentCool + '30',
  },
  pricingText: { color: Colors.accentCool, fontSize: 12, flex: 1 },
  daysScroll: { marginBottom: Spacing.md },
  dayChip: {
    width: 58,
    height: 70,
    borderRadius: Radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 8,
    backgroundColor: Colors.bgCardAlt,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
  },
  dayChipDisabled: { opacity: 0.4 },
  dayChipSelected: { borderColor: Colors.primary },
  dayName: { color: Colors.textSecondary, fontSize: 12, fontWeight: '600', marginBottom: 4 },
  dayDate: { color: Colors.textPrimary, fontSize: 18, fontWeight: '800' },
  timesGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  timeSlot: {
    width: (width - Spacing.md * 2 - 8 * 3) / 4,
    paddingVertical: 12,
    alignItems: 'center',
    borderRadius: Radius.sm,
    backgroundColor: Colors.bgCardAlt,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
    position: 'relative',
  },
  timeSlotSelected: { borderColor: Colors.primary },
  timeSlotPeak: { borderColor: Colors.borderSubtle },
  timeText: { color: Colors.textPrimary, fontSize: 12, fontWeight: '600' },
  discountPill: {
    position: 'absolute',
    top: 4,
    right: 4,
    backgroundColor: Colors.success + '20',
    borderRadius: 4,
    paddingHorizontal: 4,
    paddingVertical: 1,
  },
  discountText: { color: Colors.success, fontSize: 9, fontWeight: '800' },
  summaryCard: {
    borderRadius: Radius.lg,
    padding: Spacing.md,
    marginBottom: Spacing.lg,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    overflow: 'hidden',
  },
  summaryRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 12 },
  summaryLabel: { color: Colors.textSecondary, fontSize: 14, flex: 1 },
  summaryValue: { color: Colors.textPrimary, fontSize: 14, fontWeight: '600' },
  divider: { height: 1, backgroundColor: Colors.borderSubtle, marginVertical: 12 },
  totalRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  totalLabel: { color: Colors.textPrimary, fontSize: 16, fontWeight: '700' },
  totalValue: { color: Colors.primary, fontSize: 24, fontWeight: '800' },
  addOnsSection: { marginBottom: 20 },
  addOnsTitle: { color: Colors.textPrimary, fontSize: 16, fontWeight: '700', marginBottom: 12 },
  addOnRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderSubtle,
  },
  addOnText: { flex: 1, color: Colors.textSecondary, fontSize: 14 },
  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingHorizontal: Spacing.md,
    paddingTop: Spacing.md,
    backgroundColor: Colors.bg,
    borderTopWidth: 1,
    borderTopColor: Colors.borderSubtle,
  },
  backBtn: {
    width: 48,
    height: 56,
    borderRadius: Radius.md,
    backgroundColor: Colors.bgCardAlt,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    alignItems: 'center',
    justifyContent: 'center',
  },
  nextBtn: { flex: 1, borderRadius: Radius.full, overflow: 'hidden' },
  nextBtnDisabled: { opacity: 0.5 },
  nextBtnGrad: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 18,
  },
  nextBtnText: { color: '#0A0A0F', fontSize: 16, fontWeight: '800' },
  confirmIcon: {
    width: 96,
    height: 96,
    borderRadius: 48,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.lg,
  },
  confirmTitle: { color: Colors.textPrimary, fontSize: 32, fontWeight: '800', marginBottom: 8 },
  confirmSub: { color: Colors.textSecondary, fontSize: 16, textAlign: 'center', lineHeight: 24, marginBottom: Spacing.xl },
  confirmCard: {
    borderRadius: Radius.lg,
    padding: Spacing.lg,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    marginBottom: Spacing.xl,
    width: '100%',
    overflow: 'hidden',
  },
  confirmRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 14 },
  confirmRowText: { color: Colors.textPrimary, fontSize: 14 },
  doneBtn: { width: '100%', borderRadius: Radius.full, overflow: 'hidden' },
  doneBtnGrad: { paddingVertical: 18, alignItems: 'center' },
  doneBtnText: { color: '#0A0A0F', fontSize: 17, fontWeight: '800' },
});
