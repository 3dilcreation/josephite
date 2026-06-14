import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { Colors, FontSize, Spacing, BorderRadius, Shadows } from '../theme';
import { Order, OrderStatus } from '../types';

const mockOrders: Order[] = [
  {
    id: 'ORD001',
    productName: 'Custom Gold Medal — Sports Event',
    status: 'printing',
    placedDate: '2024-01-10',
    estimatedDelivery: '2024-01-14',
    totalAmount: 1499,
    quantity: 10,
    items: [],
    trackingId: '3DIL20240110',
    address: '42 Shivaji Nagar, Pune – 411005',
  },
  {
    id: 'ORD002',
    productName: 'Architectural Model — 3 BHK Villa',
    status: 'quality_check',
    placedDate: '2024-01-08',
    estimatedDelivery: '2024-01-13',
    totalAmount: 3999,
    quantity: 1,
    items: [],
    trackingId: '3DIL20240108',
    address: 'Tech Park, Tower B, Pune – 411045',
  },
  {
    id: 'ORD003',
    productName: 'Ganesh Idol Miniature — Custom Pose',
    status: 'delivered',
    placedDate: '2024-01-02',
    estimatedDelivery: '2024-01-06',
    totalAmount: 899,
    quantity: 1,
    items: [],
    trackingId: '3DIL20240102',
    address: '42 Shivaji Nagar, Pune – 411005',
  },
];

const statusSteps: { key: OrderStatus; label: string; icon: string }[] = [
  { key: 'placed', label: 'Order Placed', icon: '📋' },
  { key: 'designing', label: 'Designing', icon: '✏️' },
  { key: 'printing', label: 'Printing', icon: '🖨️' },
  { key: 'quality_check', label: 'Quality Check', icon: '🔍' },
  { key: 'shipped', label: 'Shipped', icon: '🚚' },
  { key: 'delivered', label: 'Delivered', icon: '✅' },
];

const statusOrder = ['placed', 'designing', 'printing', 'quality_check', 'shipped', 'delivered'];

const statusColors: Record<OrderStatus, string> = {
  placed: Colors.info,
  designing: '#8B5CF6',
  printing: Colors.primary,
  quality_check: Colors.warning,
  shipped: '#0EA5E9',
  delivered: Colors.success,
  cancelled: Colors.error,
};

const OrderTrackingScreen: React.FC = () => {
  const navigation = useNavigation();
  const [selectedOrder, setSelectedOrder] = useState<Order | null>(null);

  const getStatusIndex = (status: OrderStatus) => statusOrder.indexOf(status);

  if (selectedOrder) {
    const currentIndex = getStatusIndex(selectedOrder.status);
    return (
      <View style={styles.container}>
        <LinearGradient colors={[Colors.secondary, '#0F3460']} style={styles.header}>
          <TouchableOpacity style={styles.backBtn} onPress={() => setSelectedOrder(null)}>
            <Ionicons name="arrow-back" size={24} color={Colors.white} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Track Order</Text>
          <Text style={styles.orderId}>#{selectedOrder.id}</Text>
        </LinearGradient>

        <ScrollView showsVerticalScrollIndicator={false}>
          <View style={styles.orderSummaryCard}>
            <Text style={styles.orderProductName}>{selectedOrder.productName}</Text>
            <View style={styles.metaRow}>
              <Text style={styles.metaText}>Tracking: {selectedOrder.trackingId}</Text>
              <View style={[styles.statusBadge, { backgroundColor: statusColors[selectedOrder.status] + '20' }]}>
                <Text style={[styles.statusText, { color: statusColors[selectedOrder.status] }]}>
                  {selectedOrder.status.replace('_', ' ').toUpperCase()}
                </Text>
              </View>
            </View>
            <Text style={styles.deliveryETA}>📅 Expected: {selectedOrder.estimatedDelivery}</Text>
            <Text style={styles.deliveryAddress}>📍 {selectedOrder.address}</Text>
          </View>

          {/* Timeline */}
          <View style={styles.timeline}>
            <Text style={styles.timelineTitle}>Order Timeline</Text>
            {statusSteps.map((step, i) => {
              const isDone = i <= currentIndex && selectedOrder.status !== 'cancelled';
              const isCurrent = i === currentIndex;
              return (
                <View key={step.key} style={styles.timelineStep}>
                  <View style={styles.timelineLeft}>
                    <View style={[
                      styles.stepCircle,
                      isDone ? styles.stepCircleDone : styles.stepCirclePending,
                      isCurrent && styles.stepCircleCurrent,
                    ]}>
                      {isDone ? (
                        <Text style={styles.stepEmoji}>{step.icon}</Text>
                      ) : (
                        <Text style={styles.stepNumber}>{i + 1}</Text>
                      )}
                    </View>
                    {i < statusSteps.length - 1 && (
                      <View style={[styles.connector, isDone && styles.connectorDone]} />
                    )}
                  </View>
                  <View style={styles.stepContent}>
                    <Text style={[styles.stepLabel, isCurrent && styles.stepLabelCurrent, !isDone && styles.stepLabelPending]}>
                      {step.label}
                    </Text>
                    {isCurrent && (
                      <Text style={styles.currentStepNote}>In progress...</Text>
                    )}
                  </View>
                </View>
              );
            })}
          </View>

          {/* Amount */}
          <View style={styles.amountCard}>
            <Text style={styles.amountLabel}>Order Total</Text>
            <Text style={styles.amountValue}>₹{selectedOrder.totalAmount.toLocaleString()}</Text>
          </View>
        </ScrollView>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <LinearGradient colors={[Colors.secondary, '#0F3460']} style={styles.header}>
        <Text style={styles.headerBig}>My Orders</Text>
        <Text style={styles.headerSub}>{mockOrders.length} orders</Text>
      </LinearGradient>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: 12 }}>
        {mockOrders.map(order => (
          <TouchableOpacity
            key={order.id}
            style={styles.orderCard}
            onPress={() => setSelectedOrder(order)}
          >
            <View style={styles.orderCardHeader}>
              <Text style={styles.orderCardId}>#{order.id}</Text>
              <View style={[styles.statusBadge, { backgroundColor: statusColors[order.status] + '20' }]}>
                <Text style={[styles.statusText, { color: statusColors[order.status] }]}>
                  {order.status.replace('_', ' ').toUpperCase()}
                </Text>
              </View>
            </View>
            <Text style={styles.orderCardName}>{order.productName}</Text>
            <View style={styles.orderCardMeta}>
              <Text style={styles.metaText}>📅 {order.placedDate}</Text>
              <Text style={styles.orderAmount}>₹{order.totalAmount.toLocaleString()}</Text>
            </View>
            <View style={styles.progressBar}>
              <View style={[styles.progressFill, {
                width: `${(getStatusIndex(order.status) / (statusOrder.length - 1)) * 100}%`,
                backgroundColor: statusColors[order.status],
              }]} />
            </View>
            <View style={styles.orderCardFooter}>
              <Text style={styles.trackLink}>View Details →</Text>
              <Text style={styles.trackId}>Track: {order.trackingId}</Text>
            </View>
          </TouchableOpacity>
        ))}

        <TouchableOpacity
          style={styles.newOrderBtn}
          onPress={() => navigation.navigate('CustomOrder' as never)}
        >
          <Ionicons name="add-circle-outline" size={20} color={Colors.primary} />
          <Text style={styles.newOrderText}>Place New Order</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { paddingTop: 52, paddingBottom: 20, paddingHorizontal: Spacing.md },
  headerBig: { color: Colors.white, fontSize: FontSize.xxxl, fontWeight: '900', marginBottom: 4 },
  headerSub: { color: 'rgba(255,255,255,0.7)', fontSize: FontSize.md },
  backBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center', marginBottom: 12 },
  headerTitle: { color: Colors.white, fontSize: FontSize.xxl, fontWeight: '900', marginBottom: 4 },
  orderId: { color: 'rgba(255,255,255,0.7)', fontSize: FontSize.md },
  orderCard: { backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 16, marginBottom: 12, ...Shadows.small },
  orderCardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  orderCardId: { fontSize: FontSize.md, fontWeight: '800', color: Colors.textSecondary },
  statusBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12 },
  statusText: { fontSize: FontSize.xs, fontWeight: '800' },
  orderCardName: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary, marginBottom: 8 },
  orderCardMeta: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 10 },
  metaText: { fontSize: FontSize.sm, color: Colors.textSecondary },
  orderAmount: { fontSize: FontSize.md, fontWeight: '800', color: Colors.primary },
  progressBar: { height: 6, backgroundColor: Colors.border, borderRadius: 3, marginBottom: 10, overflow: 'hidden' },
  progressFill: { height: 6, borderRadius: 3 },
  orderCardFooter: { flexDirection: 'row', justifyContent: 'space-between' },
  trackLink: { color: Colors.primary, fontWeight: '700', fontSize: FontSize.sm },
  trackId: { fontSize: FontSize.xs, color: Colors.textLight },
  orderSummaryCard: { margin: 12, backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 16, ...Shadows.small },
  orderProductName: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.textPrimary, marginBottom: 10 },
  metaRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  deliveryETA: { fontSize: FontSize.md, color: Colors.textSecondary, marginBottom: 4 },
  deliveryAddress: { fontSize: FontSize.sm, color: Colors.textSecondary },
  timeline: { margin: 12, backgroundColor: Colors.card, borderRadius: BorderRadius.md, padding: 16, ...Shadows.small },
  timelineTitle: { fontSize: FontSize.xl, fontWeight: '800', color: Colors.textPrimary, marginBottom: 16 },
  timelineStep: { flexDirection: 'row', gap: 12 },
  timelineLeft: { alignItems: 'center' },
  stepCircle: { width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center' },
  stepCircleDone: { backgroundColor: Colors.primary },
  stepCirclePending: { backgroundColor: Colors.border },
  stepCircleCurrent: { backgroundColor: Colors.primary, borderWidth: 3, borderColor: Colors.primary + '40' },
  stepEmoji: { fontSize: 20 },
  stepNumber: { fontSize: FontSize.md, fontWeight: '800', color: Colors.textLight },
  connector: { width: 2, flex: 1, backgroundColor: Colors.border, marginVertical: 4, minHeight: 20 },
  connectorDone: { backgroundColor: Colors.primary },
  stepContent: { flex: 1, paddingBottom: 20, paddingTop: 10 },
  stepLabel: { fontSize: FontSize.md, fontWeight: '700', color: Colors.textPrimary },
  stepLabelCurrent: { color: Colors.primary },
  stepLabelPending: { color: Colors.textLight },
  currentStepNote: { fontSize: FontSize.sm, color: Colors.primary, marginTop: 4 },
  amountCard: { margin: 12, backgroundColor: Colors.secondary, borderRadius: BorderRadius.md, padding: 20, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  amountLabel: { color: Colors.white, fontSize: FontSize.lg, fontWeight: '700' },
  amountValue: { color: Colors.accent, fontSize: FontSize.xxxl, fontWeight: '900' },
  newOrderBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: Colors.card, borderRadius: BorderRadius.lg, paddingVertical: 16, marginTop: 8, borderWidth: 2, borderColor: Colors.primary, borderStyle: 'dashed' },
  newOrderText: { color: Colors.primary, fontWeight: '800', fontSize: FontSize.lg },
});

export default OrderTrackingScreen;
