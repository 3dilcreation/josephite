import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  TextInput, ActivityIndicator, Alert, Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';
import { services } from '../data/services';

const { width } = Dimensions.get('window');
const STEPS = ['Service', 'Material', 'Dimensions', 'Review'];

const materials = [
  { name: 'PLA', desc: 'Eco-friendly, great detail', price: '₹', color: '#10B981' },
  { name: 'ABS', desc: 'Durable, heat-resistant', price: '₹₹', color: '#3B82F6' },
  { name: 'PETG', desc: 'Strong, flexible, food-safe', price: '₹₹', color: '#8B5CF6' },
  { name: 'Resin', desc: 'Ultra-fine detail, premium', price: '₹₹₹', color: '#F59E0B' },
];

const colors = ['White', 'Black', 'Gray', 'Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Gold', 'Silver'];

const CustomOrderScreen: React.FC = () => {
  const navigation = useNavigation();
  const [step, setStep] = useState(0);
  const [selectedService, setSelectedService] = useState(services[0].id);
  const [selectedMaterial, setSelectedMaterial] = useState('PLA');
  const [selectedColor, setSelectedColor] = useState('White');
  const [description, setDescription] = useState('');
  const [length, setLength] = useState('10');
  const [widthVal, setWidthVal] = useState('10');
  const [height, setHeight] = useState('10');
  const [quantity, setQuantity] = useState('1');
  const [engravingText, setEngravingText] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const materialPriceMultiplier = { PLA: 1, ABS: 1.4, PETG: 1.5, Resin: 2.2 };
  const basePrice = 199;
  const volume = (parseFloat(length) || 0) * (parseFloat(widthVal) || 0) * (parseFloat(height) || 0);
  const estimated = Math.max(basePrice, Math.round((volume / 100) * (materialPriceMultiplier[selectedMaterial as keyof typeof materialPriceMultiplier] || 1))) * (parseInt(quantity) || 1);

  const handleSubmit = async () => {
    setSubmitting(true);
    await new Promise(r => setTimeout(r, 2000));
    setSubmitting(false);
    Alert.alert(
      '✅ Order Submitted!',
      'Our team will review your custom order and contact you within 2 hours with a confirmation and final quote.',
      [{ text: 'Track Order', onPress: () => navigation.navigate('OrderTracking' as never) }]
    );
  };

  const canNext = () => {
    if (step === 0) return description.length > 0;
    if (step === 1) return !!selectedMaterial;
    if (step === 2) return parseFloat(length) > 0 && parseFloat(widthVal) > 0 && parseFloat(height) > 0;
    return true;
  };

  return (
    <View style={styles.container}>
      <LinearGradient colors={[Colors.primary, Colors.secondary]} style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="close" size={24} color={Colors.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Custom Order</Text>
        <View style={styles.stepProgress}>
          {STEPS.map((s, i) => (
            <View key={i} style={styles.stepItem}>
              <View style={[styles.stepDot, i <= step && styles.stepDotActive]}>
                {i < step ? (
                  <Ionicons name="checkmark" size={12} color={Colors.white} />
                ) : (
                  <Text style={styles.stepDotText}>{i + 1}</Text>
                )}
              </View>
              {i < STEPS.length - 1 && <View style={[styles.stepLine, i < step && styles.stepLineActive]} />}
            </View>
          ))}
        </View>
        <Text style={styles.stepLabel}>{STEPS[step]}</Text>
      </LinearGradient>

      <ScrollView style={styles.body} showsVerticalScrollIndicator={false}>
        {step === 0 && (
          <View style={styles.stepContent}>
            <Text style={styles.stepTitle}>Select Service & Describe Your Order</Text>
            <Text style={styles.fieldLabel}>Service Type</Text>
            {services.map(s => (
              <TouchableOpacity
                key={s.id}
                style={[styles.serviceOption, selectedService === s.id && styles.serviceOptionActive]}
                onPress={() => setSelectedService(s.id)}
              >
                <Text style={styles.serviceEmoji}>{s.icon}</Text>
                <View style={{ flex: 1 }}>
                  <Text style={[styles.serviceName, selectedService === s.id && { color: Colors.primary }]}>{s.name}</Text>
                  <Text style={styles.serviceDesc}>{s.description.substring(0, 60)}...</Text>
                </View>
                {selectedService === s.id && <Ionicons name="checkmark-circle" size={22} color={Colors.primary} />}
              </TouchableOpacity>
            ))}
            <Text style={styles.fieldLabel}>Describe Your Requirements *</Text>
            <TextInput
              style={styles.textArea}
              multiline
              numberOfLines={4}
              placeholder="Describe what you want to print — dimensions, style, purpose, any special requirements..."
              value={description}
              onChangeText={setDescription}
              placeholderTextColor={Colors.textLight}
            />
          </View>
        )}

        {step === 1 && (
          <View style={styles.stepContent}>
            <Text style={styles.stepTitle}>Choose Material & Color</Text>
            {materials.map(m => (
              <TouchableOpacity
                key={m.name}
                style={[styles.materialCard, selectedMaterial === m.name && styles.materialCardActive]}
                onPress={() => setSelectedMaterial(m.name)}
              >
                <View style={[styles.materialDot, { backgroundColor: m.color }]} />
                <View style={{ flex: 1 }}>
                  <Text style={[styles.materialName, selectedMaterial === m.name && { color: Colors.primary }]}>{m.name}</Text>
                  <Text style={styles.materialDesc}>{m.desc}</Text>
                </View>
                <Text style={styles.materialPrice}>{m.price}</Text>
                {selectedMaterial === m.name && <Ionicons name="checkmark-circle" size={22} color={Colors.primary} />}
              </TouchableOpacity>
            ))}
            <Text style={styles.fieldLabel}>Color</Text>
            <View style={styles.colorGrid}>
              {colors.map(c => (
                <TouchableOpacity
                  key={c}
                  style={[styles.colorBtn, selectedColor === c && styles.colorBtnActive]}
                  onPress={() => setSelectedColor(c)}
                >
                  <Text style={[styles.colorText, selectedColor === c && { color: Colors.primary, fontWeight: '700' }]}>{c}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}

        {step === 2 && (
          <View style={styles.stepContent}>
            <Text style={styles.stepTitle}>Dimensions & Quantity</Text>
            <View style={styles.dimensionsRow}>
              {[
                { label: 'Length (cm)', value: length, set: setLength },
                { label: 'Width (cm)', value: widthVal, set: setWidthVal },
                { label: 'Height (cm)', value: height, set: setHeight },
              ].map((d, i) => (
                <View key={i} style={styles.dimField}>
                  <Text style={styles.fieldLabel}>{d.label}</Text>
                  <TextInput
                    style={styles.dimInput}
                    keyboardType="numeric"
                    value={d.value}
                    onChangeText={d.set}
                    placeholder="0"
                    placeholderTextColor={Colors.textLight}
                  />
                </View>
              ))}
            </View>
            <Text style={styles.fieldLabel}>Quantity</Text>
            <View style={styles.qtyRow}>
              <TouchableOpacity style={styles.qtyBtn} onPress={() => setQuantity(q => String(Math.max(1, parseInt(q) - 1)))}>
                <Ionicons name="remove" size={24} color={Colors.primary} />
              </TouchableOpacity>
              <Text style={styles.qtyValue}>{quantity}</Text>
              <TouchableOpacity style={styles.qtyBtn} onPress={() => setQuantity(q => String(parseInt(q) + 1))}>
                <Ionicons name="add" size={24} color={Colors.primary} />
              </TouchableOpacity>
            </View>
            <Text style={styles.fieldLabel}>Engraving / Custom Text (Optional)</Text>
            <TextInput
              style={styles.input}
              placeholder="Text to engrave on the product"
              value={engravingText}
              onChangeText={setEngravingText}
              placeholderTextColor={Colors.textLight}
            />
            <View style={styles.estimateCard}>
              <Text style={styles.estimateLabel}>Estimated Price</Text>
              <Text style={styles.estimatePrice}>₹{estimated.toLocaleString()}</Text>
              <Text style={styles.estimateNote}>* Final price confirmed after team review</Text>
            </View>
          </View>
        )}

        {step === 3 && (
          <View style={styles.stepContent}>
            <Text style={styles.stepTitle}>Review & Submit</Text>
            {[
              { label: 'Service', value: services.find(s => s.id === selectedService)?.name || '' },
              { label: 'Material', value: selectedMaterial },
              { label: 'Color', value: selectedColor },
              { label: 'Dimensions', value: `${length} × ${widthVal} × ${height} cm` },
              { label: 'Quantity', value: quantity },
              { label: 'Engraving', value: engravingText || 'None' },
              { label: 'Description', value: description },
            ].map((item, i) => (
              <View key={i} style={styles.reviewRow}>
                <Text style={styles.reviewLabel}>{item.label}</Text>
                <Text style={styles.reviewValue}>{item.value}</Text>
              </View>
            ))}
            <View style={styles.totalBox}>
              <Text style={styles.totalLabel}>Estimated Total</Text>
              <Text style={styles.totalPrice}>₹{estimated.toLocaleString()}</Text>
            </View>
            <View style={styles.infoBox}>
              <Ionicons name="information-circle-outline" size={18} color={Colors.info} />
              <Text style={styles.infoText}>
                Our team will contact you within 2 hours to confirm the order and finalize the exact price.
              </Text>
            </View>
          </View>
        )}
      </ScrollView>

      <View style={styles.footer}>
        {step > 0 && (
          <TouchableOpacity style={styles.backFooterBtn} onPress={() => setStep(s => s - 1)}>
            <Text style={styles.backFooterText}>← Back</Text>
          </TouchableOpacity>
        )}
        <TouchableOpacity
          style={[styles.nextBtn, !canNext() && styles.nextBtnDisabled]}
          disabled={!canNext() || submitting}
          onPress={() => step < 3 ? setStep(s => s + 1) : handleSubmit()}
        >
          <LinearGradient
            colors={canNext() ? [Colors.primary, '#FF8C42'] : [Colors.border, Colors.border]}
            style={styles.nextBtnGradient}
          >
            {submitting ? (
              <ActivityIndicator color={Colors.white} />
            ) : (
              <Text style={styles.nextBtnText}>{step < 3 ? 'Next →' : 'Submit Order 🚀'}</Text>
            )}
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { paddingTop: 52, paddingBottom: 20, paddingHorizontal: Spacing.md },
  backBtn: { position: 'absolute', top: 52, right: 16, width: 36, height: 36, borderRadius: 18, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center', zIndex: 10 },
  headerTitle: { color: Colors.white, fontSize: FontSize.xxl, fontWeight: '900', marginBottom: 16 },
  stepProgress: { flexDirection: 'row', alignItems: 'center', marginBottom: 8 },
  stepItem: { flexDirection: 'row', alignItems: 'center' },
  stepDot: { width: 28, height: 28, borderRadius: 14, backgroundColor: 'rgba(255,255,255,0.3)', alignItems: 'center', justifyContent: 'center' },
  stepDotActive: { backgroundColor: Colors.white },
  stepDotText: { color: Colors.primary, fontWeight: '800', fontSize: FontSize.sm },
  stepLine: { width: 40, height: 2, backgroundColor: 'rgba(255,255,255,0.3)', marginHorizontal: 4 },
  stepLineActive: { backgroundColor: Colors.white },
  stepLabel: { color: 'rgba(255,255,255,0.8)', fontSize: FontSize.sm, fontWeight: '600' },
  body: { flex: 1 },
  stepContent: { padding: Spacing.md },
  stepTitle: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.textPrimary, marginBottom: 20 },
  fieldLabel: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary, marginBottom: 8, marginTop: 16 },
  serviceOption: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 14, marginBottom: 10, borderWidth: 2, borderColor: 'transparent', ...Shadows.small },
  serviceOptionActive: { borderColor: Colors.primary, backgroundColor: Colors.primary + '08' },
  serviceEmoji: { fontSize: 28 },
  serviceName: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary, marginBottom: 2 },
  serviceDesc: { fontSize: FontSize.xs, color: Colors.textSecondary },
  textArea: { backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 14, fontSize: FontSize.md, color: Colors.textPrimary, borderWidth: 1.5, borderColor: Colors.border, textAlignVertical: 'top', minHeight: 100 },
  input: { backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 14, fontSize: FontSize.md, color: Colors.textPrimary, borderWidth: 1.5, borderColor: Colors.border },
  materialCard: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 14, marginBottom: 10, borderWidth: 2, borderColor: 'transparent', ...Shadows.small },
  materialCardActive: { borderColor: Colors.primary, backgroundColor: Colors.primary + '08' },
  materialDot: { width: 14, height: 14, borderRadius: 7 },
  materialName: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary, marginBottom: 2 },
  materialDesc: { fontSize: FontSize.xs, color: Colors.textSecondary },
  materialPrice: { fontSize: FontSize.lg, fontWeight: '800', color: Colors.accent, marginRight: 8 },
  colorGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  colorBtn: { paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20, borderWidth: 1.5, borderColor: Colors.border, backgroundColor: Colors.card },
  colorBtnActive: { borderColor: Colors.primary, backgroundColor: Colors.primary + '15' },
  colorText: { fontSize: FontSize.sm, color: Colors.textSecondary, fontWeight: '600' },
  dimensionsRow: { flexDirection: 'row', gap: 10, marginTop: 8 },
  dimField: { flex: 1 },
  dimInput: { backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 12, fontSize: FontSize.lg, color: Colors.textPrimary, borderWidth: 1.5, borderColor: Colors.border, textAlign: 'center', fontWeight: '700' },
  qtyRow: { flexDirection: 'row', alignItems: 'center', gap: 20, marginTop: 4 },
  qtyBtn: { width: 44, height: 44, borderRadius: 22, borderWidth: 2, borderColor: Colors.primary, alignItems: 'center', justifyContent: 'center' },
  qtyValue: { fontSize: 28, fontWeight: '900', color: Colors.textPrimary, minWidth: 40, textAlign: 'center' },
  estimateCard: { backgroundColor: Colors.primary + '15', borderRadius: BorderRadius.md, padding: 20, marginTop: 20, alignItems: 'center', borderWidth: 2, borderColor: Colors.primary + '30' },
  estimateLabel: { fontSize: FontSize.md, color: Colors.textSecondary, marginBottom: 4 },
  estimatePrice: { fontSize: 40, fontWeight: '900', color: Colors.primary },
  estimateNote: { fontSize: FontSize.xs, color: Colors.textLight, marginTop: 4 },
  reviewRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: Colors.border },
  reviewLabel: { fontSize: FontSize.md, color: Colors.textSecondary, fontWeight: '600' },
  reviewValue: { fontSize: FontSize.md, color: Colors.textPrimary, fontWeight: '700', flex: 1, textAlign: 'right' },
  totalBox: { backgroundColor: Colors.secondary, borderRadius: BorderRadius.md, padding: 20, marginTop: 16, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  totalLabel: { color: Colors.white, fontSize: FontSize.lg, fontWeight: '700' },
  totalPrice: { color: Colors.accent, fontSize: FontSize.xxxl, fontWeight: '900' },
  infoBox: { flexDirection: 'row', gap: 10, backgroundColor: Colors.info + '15', borderRadius: BorderRadius.md, padding: 14, marginTop: 12 },
  infoText: { flex: 1, fontSize: FontSize.sm, color: Colors.info, lineHeight: 20 },
  footer: { flexDirection: 'row', gap: 12, padding: Spacing.md, backgroundColor: Colors.card, borderTopWidth: 1, borderTopColor: Colors.border, paddingBottom: 28 },
  backFooterBtn: { width: 100, borderWidth: 2, borderColor: Colors.border, borderRadius: BorderRadius.lg, alignItems: 'center', justifyContent: 'center', paddingVertical: 16 },
  backFooterText: { color: Colors.textSecondary, fontWeight: '700', fontSize: FontSize.md },
  nextBtn: { flex: 1, borderRadius: BorderRadius.lg, overflow: 'hidden' },
  nextBtnDisabled: { opacity: 0.6 },
  nextBtnGradient: { paddingVertical: 18, alignItems: 'center', borderRadius: BorderRadius.lg },
  nextBtnText: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '800' },
});

export default CustomOrderScreen;
