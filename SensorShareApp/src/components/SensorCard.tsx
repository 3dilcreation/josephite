import React, { memo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { COLORS } from '../constants';

interface Props {
  title: string;
  icon: string;
  unit: string;
  fields: { label: string; value: number | string | null | undefined }[];
  anomaly?: boolean;
}

const SensorCard = memo(({ title, icon, unit, fields, anomaly }: Props) => {
  return (
    <View style={[styles.card, anomaly && styles.cardAnomaly]}>
      <View style={styles.header}>
        <Text style={styles.icon}>{icon}</Text>
        <Text style={styles.title}>{title}</Text>
        {unit ? <Text style={styles.unit}>{unit}</Text> : null}
        {anomaly && <View style={styles.anomalyDot} />}
      </View>
      <View style={styles.fields}>
        {fields.map((f, i) => (
          <View key={i} style={styles.field}>
            <Text style={styles.fieldLabel}>{f.label}</Text>
            <Text style={[styles.fieldValue, anomaly && styles.fieldValueAnomaly]}>
              {f.value === null || f.value === undefined ? '—' : typeof f.value === 'number' ? f.value.toFixed(3) : f.value}
            </Text>
          </View>
        ))}
      </View>
    </View>
  );
});

const styles = StyleSheet.create({
  card: {
    backgroundColor: COLORS.surface,
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  cardAnomaly: {
    borderColor: COLORS.warning,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  icon: { fontSize: 18, marginRight: 6 },
  title: { color: COLORS.text, fontSize: 14, fontWeight: '700', flex: 1 },
  unit: { color: COLORS.textMuted, fontSize: 11, marginLeft: 4 },
  anomalyDot: {
    width: 8, height: 8, borderRadius: 4,
    backgroundColor: COLORS.warning, marginLeft: 6,
  },
  fields: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  field: { minWidth: 80 },
  fieldLabel: { color: COLORS.textMuted, fontSize: 11, marginBottom: 2 },
  fieldValue: { color: COLORS.primary, fontSize: 15, fontWeight: '600', fontVariant: ['tabular-nums'] },
  fieldValueAnomaly: { color: COLORS.warning },
});

SensorCard.displayName = 'SensorCard';
export default SensorCard;
