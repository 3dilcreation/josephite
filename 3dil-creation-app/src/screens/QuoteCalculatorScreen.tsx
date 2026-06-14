import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  TextInput, ActivityIndicator, Linking,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';

const complexities = ['Simple', 'Moderate', 'Complex', 'Highly Detailed'];
const finishes = ['Raw', 'Sanded', 'Painted', 'Premium'];
const materialOptions = [
  { name: 'PLA', multiplier: 1.0 },
  { name: 'ABS', multiplier: 1.4 },
  { name: 'PETG', multiplier: 1.5 },
  { name: 'Resin', multiplier: 2.2 },
  { name: "Don't know", multiplier: 1.2 },
];
const complexityMultiplier: Record<string, number> = { Simple: 1, Moderate: 1.5, Complex: 2.2, 'Highly Detailed': 3.0 };
const finishMultiplier: Record<string, number> = { Raw: 1, Sanded: 1.3, Painted: 1.7, Premium: 2.2 };

const QuoteCalculatorScreen: React.FC = () => {
  const navigation = useNavigation();
  const [description, setDescription] = useState('');
  const [quantity, setQuantity] = useState('1');
  const [length, setLength] = useState('');
  const [widthVal, setWidthVal] = useState('');
  const [heightVal, setHeightVal] = useState('');
  const [material, setMaterial] = useState('PLA');
  const [complexity, setComplexity] = useState('Moderate');
  const [finish, setFinish] = useState('Raw');
  const [calculating, setCalculating] = useState(false);
  const [quoteResult, setQuoteResult] = useState<{ min: number; max: number; breakdown: Record<string, number> } | null>(null);

  const calculateQuote = async () => {
    setCalculating(true);
    await new Promise(r => setTimeout(r, 2500));
    const vol = (parseFloat(length) || 5) * (parseFloat(widthVal) || 5) * (parseFloat(heightVal) || 5);
    const matMul = materialOptions.find(m => m.name === material)?.multiplier || 1;
    const compMul = complexityMultiplier[complexity] || 1;
    const finMul = finishMultiplier[finish] || 1;
    const qty = parseInt(quantity) || 1;
    const materialCost = Math.round((vol / 50) * matMul * 80);
    const printingCost = Math.round(compMul * 150);
    const finishingCost = Math.round((finMul - 1) * 100);
    const deliveryCost = qty > 5 ? 0 : 50;
    const base = (materialCost + printingCost + finishingCost + deliveryCost) * qty;
    setQuoteResult({
      min: Math.max(199, Math.round(base * 0.85)),
      max: Math.round(base * 1.15),
      breakdown: { Material: materialCost * qty, Printing: printingCost * qty, Finishing: finishingCost * qty, Delivery: deliveryCost },
    });
    setCalculating(false);
  };

  const openWhatsApp = () => {
    const msg = `Hi 3DIL Creation! I'd like a quote for:\n- Description: ${description}\n- Material: ${material}\n- Size: ${length}×${widthVal}×${heightVal} cm\n- Complexity: ${complexity}\n- Finish: ${finish}\n- Quantity: ${quantity}\n- Estimated Budget: ₹${quoteResult?.min.toLocaleString()} – ₹${quoteResult?.max.toLocaleString()}`;
    Linking.openURL(`https://wa.me/919999999999?text=${encodeURIComponent(msg)}`);
  };

  return (
    <View style={styles.container}>
      <LinearGradient colors={['#7C3AED', '#6D28D9']} style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="close" size={24} color={Colors.white} />
        </TouchableOpacity>
        <Text style={styles.headerEmoji}>🤖</Text>
        <Text style={styles.headerTitle}>AI Quote Calculator</Text>
        <Text style={styles.headerSubtitle}>Get an instant price estimate</Text>
      </LinearGradient>

      <ScrollView style={styles.body} showsVerticalScrollIndicator={false}>
        <View style={styles.form}>
          <Text style={styles.fieldLabel}>Describe Your Product</Text>
          <TextInput
            style={styles.textArea}
            multiline
            numberOfLines={3}
            placeholder="e.g., Custom medal with logo engraving for sports event..."
            value={description}
            onChangeText={setDescription}
            placeholderTextColor={Colors.textLight}
          />

          <Text style={styles.fieldLabel}>Dimensions (cm)</Text>
          <View style={styles.dimRow}>
            {[
              { label: 'L', val: length, set: setLength },
              { label: 'W', val: widthVal, set: setWidthVal },
              { label: 'H', val: heightVal, set: setHeightVal },
            ].map((d, i) => (
              <View key={i} style={styles.dimBox}>
                <Text style={styles.dimLabel}>{d.label}</Text>
                <TextInput
                  style={styles.dimInput}
                  keyboardType="numeric"
                  value={d.val}
                  onChangeText={d.set}
                  placeholder="cm"
                  placeholderTextColor={Colors.textLight}
                />
              </View>
            ))}
          </View>

          <Text style={styles.fieldLabel}>Quantity</Text>
          <View style={styles.qtyRow}>
            <TouchableOpacity style={styles.qtyBtn} onPress={() => setQuantity(q => String(Math.max(1, parseInt(q) - 1)))}>
              <Ionicons name="remove" size={20} color={Colors.primary} />
            </TouchableOpacity>
            <Text style={styles.qty}>{quantity}</Text>
            <TouchableOpacity style={styles.qtyBtn} onPress={() => setQuantity(q => String(parseInt(q) + 1))}>
              <Ionicons name="add" size={20} color={Colors.primary} />
            </TouchableOpacity>
          </View>

          <Text style={styles.fieldLabel}>Material Preference</Text>
          <View style={styles.chipRow}>
            {materialOptions.map(m => (
              <TouchableOpacity
                key={m.name}
                style={[styles.chip, material === m.name && styles.chipActive]}
                onPress={() => setMaterial(m.name)}
              >
                <Text style={[styles.chipText, material === m.name && styles.chipTextActive]}>{m.name}</Text>
              </TouchableOpacity>
            ))}
          </View>

          <Text style={styles.fieldLabel}>Model Complexity</Text>
          <View style={styles.chipRow}>
            {complexities.map(c => (
              <TouchableOpacity
                key={c}
                style={[styles.chip, complexity === c && styles.chipActive]}
                onPress={() => setComplexity(c)}
              >
                <Text style={[styles.chipText, complexity === c && styles.chipTextActive]}>{c}</Text>
              </TouchableOpacity>
            ))}
          </View>

          <Text style={styles.fieldLabel}>Finish Type</Text>
          <View style={styles.chipRow}>
            {finishes.map(f => (
              <TouchableOpacity
                key={f}
                style={[styles.chip, finish === f && styles.chipActive]}
                onPress={() => setFinish(f)}
              >
                <Text style={[styles.chipText, finish === f && styles.chipTextActive]}>{f}</Text>
              </TouchableOpacity>
            ))}
          </View>

          <TouchableOpacity style={styles.calculateBtn} onPress={calculateQuote} disabled={calculating}>
            <LinearGradient colors={['#7C3AED', '#6D28D9']} style={styles.calculateBtnGradient}>
              {calculating ? (
                <View style={styles.calculatingRow}>
                  <ActivityIndicator color={Colors.white} />
                  <Text style={styles.calculateBtnText}>Calculating...</Text>
                </View>
              ) : (
                <Text style={styles.calculateBtnText}>🤖 Calculate Quote</Text>
              )}
            </LinearGradient>
          </TouchableOpacity>

          {quoteResult && (
            <View style={styles.resultCard}>
              <Text style={styles.resultTitle}>Estimated Price Range</Text>
              <Text style={styles.resultPrice}>
                ₹{quoteResult.min.toLocaleString()} – ₹{quoteResult.max.toLocaleString()}
              </Text>
              <Text style={styles.resultNote}>Based on {quantity} unit(s) · {material} · {finish} finish</Text>

              <View style={styles.breakdown}>
                <Text style={styles.breakdownTitle}>Price Breakdown</Text>
                {Object.entries(quoteResult.breakdown).map(([key, val]) => (
                  <View key={key} style={styles.breakdownRow}>
                    <Text style={styles.breakdownLabel}>{key}</Text>
                    <Text style={styles.breakdownValue}>₹{val.toLocaleString()}</Text>
                  </View>
                ))}
              </View>

              <View style={styles.actionBtns}>
                <TouchableOpacity style={styles.whatsappBtn} onPress={openWhatsApp}>
                  <Text style={styles.whatsappBtnText}>💬 Request Quote on WhatsApp</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.orderBtn}
                  onPress={() => navigation.navigate('CustomOrder' as never)}
                >
                  <Text style={styles.orderBtnText}>Place Order →</Text>
                </TouchableOpacity>
              </View>
            </View>
          )}
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { paddingTop: 52, paddingBottom: 24, paddingHorizontal: Spacing.md, alignItems: 'center' },
  backBtn: { position: 'absolute', top: 52, right: 16, width: 36, height: 36, borderRadius: 18, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  headerEmoji: { fontSize: 48, marginBottom: 8 },
  headerTitle: { color: Colors.white, fontSize: FontSize.xxxl, fontWeight: '900', marginBottom: 4 },
  headerSubtitle: { color: 'rgba(255,255,255,0.8)', fontSize: FontSize.md },
  body: { flex: 1 },
  form: { padding: Spacing.md },
  fieldLabel: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary, marginBottom: 8, marginTop: 16 },
  textArea: { backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 14, fontSize: FontSize.md, color: Colors.textPrimary, borderWidth: 1.5, borderColor: Colors.border, textAlignVertical: 'top', minHeight: 80 },
  dimRow: { flexDirection: 'row', gap: 10 },
  dimBox: { flex: 1 },
  dimLabel: { fontSize: FontSize.sm, color: Colors.textSecondary, fontWeight: '600', marginBottom: 4 },
  dimInput: { backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 12, fontSize: FontSize.lg, color: Colors.textPrimary, borderWidth: 1.5, borderColor: Colors.border, textAlign: 'center', fontWeight: '700' },
  qtyRow: { flexDirection: 'row', alignItems: 'center', gap: 16 },
  qtyBtn: { width: 40, height: 40, borderRadius: 20, borderWidth: 2, borderColor: Colors.primary, alignItems: 'center', justifyContent: 'center' },
  qty: { fontSize: 24, fontWeight: '900', color: Colors.textPrimary, minWidth: 36, textAlign: 'center' },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: { paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20, borderWidth: 1.5, borderColor: Colors.border, backgroundColor: Colors.card },
  chipActive: { borderColor: '#7C3AED', backgroundColor: '#7C3AED15' },
  chipText: { fontSize: FontSize.md, color: Colors.textSecondary, fontWeight: '600' },
  chipTextActive: { color: '#7C3AED', fontWeight: '800' },
  calculateBtn: { marginTop: 24, borderRadius: BorderRadius.lg, overflow: 'hidden' },
  calculateBtnGradient: { paddingVertical: 18, alignItems: 'center', borderRadius: BorderRadius.lg },
  calculatingRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  calculateBtnText: { color: Colors.white, fontSize: FontSize.xl, fontWeight: '800' },
  resultCard: { marginTop: 20, backgroundColor: Colors.card, borderRadius: BorderRadius.lg, padding: 20, ...Shadows.medium, marginBottom: 32 },
  resultTitle: { fontSize: FontSize.lg, fontWeight: '700', color: Colors.textSecondary, textAlign: 'center', marginBottom: 8 },
  resultPrice: { fontSize: 36, fontWeight: '900', color: '#7C3AED', textAlign: 'center', marginBottom: 4 },
  resultNote: { fontSize: FontSize.sm, color: Colors.textLight, textAlign: 'center', marginBottom: 16 },
  breakdown: { backgroundColor: Colors.background, borderRadius: BorderRadius.md, padding: 14, marginBottom: 16 },
  breakdownTitle: { fontSize: FontSize.md, fontWeight: '800', color: Colors.textPrimary, marginBottom: 10 },
  breakdownRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: Colors.border },
  breakdownLabel: { fontSize: FontSize.md, color: Colors.textSecondary },
  breakdownValue: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary },
  actionBtns: { gap: 10 },
  whatsappBtn: { backgroundColor: '#25D366', borderRadius: BorderRadius.lg, paddingVertical: 16, alignItems: 'center' },
  whatsappBtnText: { color: Colors.white, fontSize: FontSize.lg, fontWeight: '800' },
  orderBtn: { backgroundColor: Colors.primary, borderRadius: BorderRadius.lg, paddingVertical: 16, alignItems: 'center' },
  orderBtnText: { color: Colors.white, fontSize: FontSize.lg, fontWeight: '800' },
});

export default QuoteCalculatorScreen;
