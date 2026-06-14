import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { OrderStatus } from '../types';
import { Colors, FontSize, BorderRadius } from '../theme';

interface OrderStatusBadgeProps {
  status: OrderStatus;
}

const statusConfig: Record<OrderStatus, { label: string; color: string; bg: string; emoji: string }> = {
  placed: { label: 'Order Placed', color: Colors.info, bg: '#EFF6FF', emoji: '📋' },
  designing: { label: 'Designing', color: '#7C3AED', bg: '#F5F3FF', emoji: '✏️' },
  printing: { label: 'Printing', color: Colors.primary, bg: '#FFF3EE', emoji: '🖨️' },
  quality_check: { label: 'Quality Check', color: Colors.warning, bg: '#FFFBEB', emoji: '🔍' },
  shipped: { label: 'Shipped', color: Colors.success, bg: '#F0FDF4', emoji: '🚚' },
  delivered: { label: 'Delivered', color: '#166534', bg: '#DCFCE7', emoji: '✅' },
  cancelled: { label: 'Cancelled', color: Colors.error, bg: '#FEF2F2', emoji: '❌' },
};

const OrderStatusBadge: React.FC<OrderStatusBadgeProps> = ({ status }) => {
  const config = statusConfig[status];
  return (
    <View style={[styles.badge, { backgroundColor: config.bg }]}>
      <Text style={styles.emoji}>{config.emoji}</Text>
      <Text style={[styles.label, { color: config.color }]}>{config.label}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: BorderRadius.round,
    gap: 4,
    alignSelf: 'flex-start',
  },
  emoji: {
    fontSize: FontSize.sm,
  },
  label: {
    fontSize: FontSize.xs,
    fontWeight: '700',
  },
});

export default OrderStatusBadge;
