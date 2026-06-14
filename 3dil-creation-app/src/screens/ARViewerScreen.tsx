import React, { useState, useEffect, useRef } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, Animated, Dimensions,
  Modal, ScrollView, Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { Colors, FontSize, Spacing, BorderRadius } from '../theme';
import { products } from '../data/products';

const { width, height } = Dimensions.get('window');

const ARViewerScreen: React.FC = () => {
  const navigation = useNavigation();
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [selectedModel, setSelectedModel] = useState(products[0]);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [isPlaced, setIsPlaced] = useState(false);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const floatAnim = useRef(new Animated.Value(0)).current;
  const scanAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    setTimeout(() => setHasPermission(true), 500);

    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.3, duration: 800, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
      ])
    ).start();

    Animated.loop(
      Animated.timing(scanAnim, { toValue: 1, duration: 2000, useNativeDriver: true })
    ).start();
  }, []);

  useEffect(() => {
    if (isPlaced) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(floatAnim, { toValue: -10, duration: 1500, useNativeDriver: true }),
          Animated.timing(floatAnim, { toValue: 0, duration: 1500, useNativeDriver: true }),
        ])
      ).start();
    }
  }, [isPlaced]);

  const handleCapture = () => {
    Alert.alert('📸 Captured!', 'AR screenshot saved to your gallery.', [{ text: 'OK' }]);
  };

  const scanY = scanAnim.interpolate({ inputRange: [0, 1], outputRange: [0, height * 0.6] });

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionView}>
          <Text style={styles.permEmoji}>📷</Text>
          <Text style={styles.permTitle}>Camera Access Needed</Text>
          <Text style={styles.permDesc}>
            3DIL AR Preview needs camera access to show you how products look in your space.
          </Text>
          <TouchableOpacity style={styles.permBtn} onPress={() => setHasPermission(true)}>
            <Text style={styles.permBtnText}>Allow Camera Access</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Text style={styles.cancelLink}>Not now</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Simulated Camera View */}
      <View style={styles.cameraView}>
        <LinearGradient
          colors={['#1a3a1a', '#2d5a2d', '#1a3a1a']}
          style={styles.cameraBackground}
        >
          <View style={styles.gridOverlay}>
            {Array(8).fill(0).map((_, i) => (
              <View key={i} style={[styles.gridLine, { top: `${(i + 1) * 12.5}%` }]} />
            ))}
            {Array(8).fill(0).map((_, i) => (
              <View key={i} style={[styles.gridLineV, { left: `${(i + 1) * 12.5}%` }]} />
            ))}
          </View>

          {!isPlaced && (
            <Animated.View style={[styles.scanLine, { transform: [{ translateY: scanY }] }]} />
          )}

          {/* AR Object */}
          {isPlaced ? (
            <Animated.View style={[styles.arObject, { transform: [{ translateY: floatAnim }] }]}>
              <Text style={styles.arEmoji}>{selectedModel.emoji}</Text>
              <View style={styles.arShadow} />
              <View style={styles.arLabel}>
                <Text style={styles.arLabelText}>{selectedModel.name}</Text>
                <Text style={styles.arLabelPrice}>₹{selectedModel.price.toLocaleString()}</Text>
              </View>
            </Animated.View>
          ) : (
            <View style={styles.centerArea}>
              <Animated.View style={[styles.placementCircle, { transform: [{ scale: pulseAnim }] }]}>
                <Text style={styles.placementEmoji}>👆</Text>
              </Animated.View>
              <Text style={styles.tapHint}>Tap to place {selectedModel.name}</Text>
            </View>
          )}
        </LinearGradient>

        <TouchableOpacity
          style={StyleSheet.absoluteFillObject}
          onPress={() => setIsPlaced(p => !p)}
          activeOpacity={1}
        />
      </View>

      {/* Top Bar */}
      <View style={styles.topBar}>
        <TouchableOpacity style={styles.topBtn} onPress={() => navigation.goBack()}>
          <Ionicons name="close" size={24} color={Colors.white} />
        </TouchableOpacity>
        <View style={styles.arBadge}>
          <Text style={styles.arBadgeText}>AR PREVIEW</Text>
        </View>
        <TouchableOpacity style={styles.topBtn} onPress={handleCapture}>
          <Ionicons name="camera-outline" size={24} color={Colors.white} />
        </TouchableOpacity>
      </View>

      {/* Bottom Controls */}
      <View style={styles.bottomControls}>
        <TouchableOpacity style={styles.modelPickerBtn} onPress={() => setShowModelPicker(true)}>
          <Text style={styles.modelPickerEmoji}>{selectedModel.emoji}</Text>
          <View>
            <Text style={styles.modelPickerName} numberOfLines={1}>{selectedModel.name}</Text>
            <Text style={styles.modelPickerPrice}>₹{selectedModel.price.toLocaleString()}</Text>
          </View>
          <Ionicons name="chevron-up" size={20} color={Colors.white} />
        </TouchableOpacity>

        <View style={styles.actionRow}>
          <TouchableOpacity style={styles.actionBtn} onPress={() => setIsPlaced(false)}>
            <Ionicons name="refresh" size={22} color={Colors.white} />
            <Text style={styles.actionBtnText}>Reset</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.captureBtn} onPress={handleCapture}>
            <View style={styles.captureBtnInner} />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.actionBtn}
            onPress={() => navigation.navigate('ProductDetail' as never, { productId: selectedModel.id } as never)}
          >
            <Ionicons name="cart-outline" size={22} color={Colors.white} />
            <Text style={styles.actionBtnText}>Order</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Model Picker Modal */}
      <Modal visible={showModelPicker} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalSheet}>
            <View style={styles.modalHandle} />
            <Text style={styles.modalTitle}>Select Model</Text>
            <ScrollView showsVerticalScrollIndicator={false}>
              {products.map(p => (
                <TouchableOpacity
                  key={p.id}
                  style={[styles.modelOption, selectedModel.id === p.id && styles.modelOptionActive]}
                  onPress={() => { setSelectedModel(p); setShowModelPicker(false); setIsPlaced(false); }}
                >
                  <Text style={styles.modelOptionEmoji}>{p.emoji}</Text>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.modelOptionName}>{p.name}</Text>
                    <Text style={styles.modelOptionCategory}>{p.category} · {p.material}</Text>
                  </View>
                  <Text style={styles.modelOptionPrice}>₹{p.price.toLocaleString()}</Text>
                  {selectedModel.id === p.id && <Ionicons name="checkmark-circle" size={22} color={Colors.primary} />}
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.black },
  cameraView: { flex: 1 },
  cameraBackground: { flex: 1, position: 'relative' },
  gridOverlay: { ...StyleSheet.absoluteFillObject, opacity: 0.15 },
  gridLine: { position: 'absolute', left: 0, right: 0, height: 1, backgroundColor: '#00FF00' },
  gridLineV: { position: 'absolute', top: 0, bottom: 0, width: 1, backgroundColor: '#00FF00' },
  scanLine: { position: 'absolute', left: 0, right: 0, height: 2, backgroundColor: 'rgba(0,255,0,0.8)' },
  centerArea: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 16 },
  placementCircle: { width: 100, height: 100, borderRadius: 50, backgroundColor: 'rgba(255,107,53,0.3)', borderWidth: 2, borderColor: Colors.primary, alignItems: 'center', justifyContent: 'center' },
  placementEmoji: { fontSize: 40 },
  tapHint: { color: Colors.white, fontSize: FontSize.lg, fontWeight: '600', textAlign: 'center', backgroundColor: 'rgba(0,0,0,0.5)', paddingHorizontal: 20, paddingVertical: 8, borderRadius: 20 },
  arObject: { position: 'absolute', bottom: 120, alignSelf: 'center', alignItems: 'center' },
  arEmoji: { fontSize: 100 },
  arShadow: { width: 100, height: 20, backgroundColor: 'rgba(0,0,0,0.4)', borderRadius: 50, marginTop: -10 },
  arLabel: { backgroundColor: 'rgba(0,0,0,0.7)', paddingHorizontal: 14, paddingVertical: 6, borderRadius: 12, marginTop: 8, alignItems: 'center' },
  arLabelText: { color: Colors.white, fontSize: FontSize.sm, fontWeight: '700' },
  arLabelPrice: { color: Colors.primary, fontSize: FontSize.md, fontWeight: '800' },
  topBar: { position: 'absolute', top: 52, left: 0, right: 0, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: Spacing.md },
  topBtn: { width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(0,0,0,0.5)', alignItems: 'center', justifyContent: 'center' },
  arBadge: { backgroundColor: Colors.primary, paddingHorizontal: 14, paddingVertical: 6, borderRadius: 20 },
  arBadgeText: { color: Colors.white, fontSize: FontSize.sm, fontWeight: '900', letterSpacing: 2 },
  bottomControls: { position: 'absolute', bottom: 0, left: 0, right: 0, backgroundColor: 'rgba(0,0,0,0.85)', paddingBottom: 40, paddingTop: 16, paddingHorizontal: Spacing.md },
  modelPickerBtn: { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: BorderRadius.md, padding: 12, marginBottom: 16 },
  modelPickerEmoji: { fontSize: 32 },
  modelPickerName: { color: Colors.white, fontSize: FontSize.md, fontWeight: '700', maxWidth: 200 },
  modelPickerPrice: { color: Colors.primary, fontSize: FontSize.sm, fontWeight: '700' },
  actionRow: { flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center' },
  actionBtn: { alignItems: 'center', gap: 4 },
  actionBtnText: { color: Colors.white, fontSize: FontSize.xs, fontWeight: '600' },
  captureBtn: { width: 72, height: 72, borderRadius: 36, backgroundColor: Colors.white, alignItems: 'center', justifyContent: 'center' },
  captureBtnInner: { width: 58, height: 58, borderRadius: 29, backgroundColor: Colors.white, borderWidth: 3, borderColor: Colors.textLight },
  modalOverlay: { flex: 1, backgroundColor: Colors.overlay, justifyContent: 'flex-end' },
  modalSheet: { backgroundColor: Colors.card, borderTopLeftRadius: 24, borderTopRightRadius: 24, padding: 20, maxHeight: height * 0.7 },
  modalHandle: { width: 40, height: 4, backgroundColor: Colors.border, borderRadius: 2, alignSelf: 'center', marginBottom: 16 },
  modalTitle: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.textPrimary, marginBottom: 16 },
  modelOption: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: Colors.border },
  modelOptionActive: { backgroundColor: Colors.primary + '10', borderRadius: BorderRadius.md, paddingHorizontal: 8 },
  modelOptionEmoji: { fontSize: 32 },
  modelOptionName: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary, marginBottom: 2 },
  modelOptionCategory: { fontSize: FontSize.xs, color: Colors.textSecondary },
  modelOptionPrice: { fontSize: FontSize.md, fontWeight: '800', color: Colors.primary, marginRight: 8 },
  permissionView: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 40, backgroundColor: Colors.background },
  permEmoji: { fontSize: 72, marginBottom: 16 },
  permTitle: { fontSize: FontSize.xxl, fontWeight: '900', color: Colors.textPrimary, marginBottom: 12, textAlign: 'center' },
  permDesc: { fontSize: FontSize.md, color: Colors.textSecondary, textAlign: 'center', lineHeight: 22, marginBottom: 32 },
  permBtn: { backgroundColor: Colors.primary, paddingHorizontal: 32, paddingVertical: 16, borderRadius: BorderRadius.lg, marginBottom: 12 },
  permBtnText: { color: Colors.white, fontWeight: '800', fontSize: FontSize.lg },
  cancelLink: { color: Colors.textSecondary, fontSize: FontSize.md, fontWeight: '600' },
});

export default ARViewerScreen;
