import React from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';
import { services } from '../data/services';

type Nav = NativeStackNavigationProp<RootStackParamList>;
type Route = RouteProp<RootStackParamList, 'ServiceDetail'>;

const ServiceDetailScreen: React.FC = () => {
  const navigation = useNavigation<Nav>();
  const route = useRoute<Route>();
  const service = services.find(s => s.id === route.params.serviceId) || services[0];

  return (
    <View style={styles.container}>
      <LinearGradient colors={[Colors.primary, Colors.secondary]} style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color={Colors.white} />
        </TouchableOpacity>
        <Text style={styles.headerEmoji}>{service.icon}</Text>
        <Text style={styles.headerTitle}>{service.name}</Text>
        <Text style={styles.headerPrice}>Starting at ₹{service.startingPrice}</Text>
        <View style={styles.turnaroundBadge}>
          <Ionicons name="time-outline" size={14} color={Colors.white} />
          <Text style={styles.turnaroundText}>{service.turnaround}</Text>
        </View>
      </LinearGradient>

      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Description */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>About This Service</Text>
          <Text style={styles.description}>{service.longDescription}</Text>
        </View>

        {/* Features */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>What's Included</Text>
          {service.features.map((f, i) => (
            <View key={i} style={styles.featureRow}>
              <View style={styles.checkIcon}>
                <Ionicons name="checkmark" size={14} color={Colors.white} />
              </View>
              <Text style={styles.featureText}>{f}</Text>
            </View>
          ))}
        </View>

        {/* Process */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Our Process</Text>
          {service.process.map((step, i) => (
            <View key={i} style={styles.processStep}>
              <View style={styles.stepNumber}>
                <Text style={styles.stepNumberText}>{i + 1}</Text>
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.stepText}>{step}</Text>
              </View>
              {i < service.process.length - 1 && <View style={styles.stepConnector} />}
            </View>
          ))}
        </View>

        {/* FAQs */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>FAQs</Text>
          {service.faqs.map((faq, i) => (
            <View key={i} style={styles.faqCard}>
              <Text style={styles.faqQuestion}>Q: {faq.question}</Text>
              <Text style={styles.faqAnswer}>A: {faq.answer}</Text>
            </View>
          ))}
        </View>

        {/* CTA */}
        <View style={styles.ctaSection}>
          <TouchableOpacity
            style={styles.orderBtn}
            onPress={() => navigation.navigate('CustomOrder', { serviceId: service.id })}
          >
            <LinearGradient colors={[Colors.primary, '#FF8C42']} style={styles.orderBtnGradient}>
              <Text style={styles.orderBtnText}>Place Custom Order</Text>
              <Ionicons name="arrow-forward" size={20} color={Colors.white} />
            </LinearGradient>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.quoteBtn}
            onPress={() => navigation.navigate('QuoteCalculator')}
          >
            <Text style={styles.quoteBtnText}>Get Instant Quote</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { paddingTop: 52, paddingBottom: 32, paddingHorizontal: Spacing.md, alignItems: 'center' },
  backBtn: { position: 'absolute', top: 52, left: 16, width: 40, height: 40, borderRadius: 20, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  headerEmoji: { fontSize: 64, marginBottom: 12 },
  headerTitle: { fontSize: FontSize.xxxl, fontWeight: '900', color: Colors.white, textAlign: 'center', marginBottom: 8 },
  headerPrice: { fontSize: FontSize.xl, color: 'rgba(255,255,255,0.9)', fontWeight: '700', marginBottom: 8 },
  turnaroundBadge: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: 'rgba(255,255,255,0.2)', paddingHorizontal: 14, paddingVertical: 6, borderRadius: 20 },
  turnaroundText: { color: Colors.white, fontSize: FontSize.sm, fontWeight: '600' },
  section: { margin: Spacing.md, backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: Spacing.md, ...Shadows.small },
  sectionTitle: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.textPrimary, marginBottom: 12 },
  description: { fontSize: FontSize.md, color: Colors.textSecondary, lineHeight: 24 },
  featureRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 10 },
  checkIcon: { width: 22, height: 22, borderRadius: 11, backgroundColor: Colors.success, alignItems: 'center', justifyContent: 'center' },
  featureText: { fontSize: FontSize.md, color: Colors.textPrimary, flex: 1 },
  processStep: { flexDirection: 'row', alignItems: 'flex-start', gap: 12, marginBottom: 16, position: 'relative' },
  stepNumber: { width: 32, height: 32, borderRadius: 16, backgroundColor: Colors.primary, alignItems: 'center', justifyContent: 'center' },
  stepNumberText: { color: Colors.white, fontWeight: '800', fontSize: FontSize.md },
  stepText: { fontSize: FontSize.md, color: Colors.textSecondary, lineHeight: 22, flex: 1 },
  stepConnector: { position: 'absolute', left: 15, top: 36, width: 2, height: 16, backgroundColor: Colors.border },
  faqCard: { backgroundColor: Colors.background, borderRadius: BorderRadius.sm, padding: 14, marginBottom: 10, borderLeftWidth: 3, borderLeftColor: Colors.primary },
  faqQuestion: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary, marginBottom: 6 },
  faqAnswer: { fontSize: FontSize.md, color: Colors.textSecondary, lineHeight: 22 },
  ctaSection: { padding: Spacing.md, gap: 12, marginBottom: 24 },
  orderBtn: { borderRadius: BorderRadius.lg, overflow: 'hidden' },
  orderBtnGradient: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 18, gap: 10, borderRadius: BorderRadius.lg },
  orderBtnText: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '800' },
  quoteBtn: { backgroundColor: Colors.card, paddingVertical: 16, borderRadius: BorderRadius.lg, alignItems: 'center', borderWidth: 2, borderColor: Colors.primary },
  quoteBtnText: { color: Colors.primary, fontSize: FontSize.lg, fontWeight: '700' },
});

export default ServiceDetailScreen;
