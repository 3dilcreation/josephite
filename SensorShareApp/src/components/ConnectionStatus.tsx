import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { ConnectionState } from '../types';
import { COLORS } from '../constants';

interface Props {
  state: ConnectionState;
  mode: string;
  latencyMs?: number;
  slaveCount?: number;
}

const STATE_CONFIG: Record<ConnectionState, { label: string; color: string }> = {
  idle:         { label: 'Idle',         color: COLORS.textMuted },
  connecting:   { label: 'Connecting…',  color: COLORS.warning },
  connected:    { label: 'Connected',    color: COLORS.success },
  error:        { label: 'Error',        color: COLORS.danger },
  disconnected: { label: 'Disconnected', color: COLORS.danger },
};

export default function ConnectionStatus({ state, mode, latencyMs, slaveCount }: Props) {
  const cfg = STATE_CONFIG[state];
  const isConnecting = state === 'connecting';

  return (
    <View style={styles.container}>
      <View style={[styles.dot, { backgroundColor: cfg.color }]} />
      {isConnecting && <ActivityIndicator size="small" color={cfg.color} style={styles.spinner} />}
      <Text style={[styles.label, { color: cfg.color }]}>{cfg.label}</Text>
      <Text style={styles.mode}> · {mode.toUpperCase()}</Text>
      {latencyMs !== undefined && state === 'connected' && (
        <Text style={styles.latency}> · {latencyMs}ms</Text>
      )}
      {slaveCount !== undefined && slaveCount > 0 && (
        <Text style={styles.slaves}> · {slaveCount} slave{slaveCount > 1 ? 's' : ''}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: COLORS.surface,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignSelf: 'flex-start',
  },
  dot: { width: 8, height: 8, borderRadius: 4, marginRight: 6 },
  spinner: { marginRight: 4 },
  label: { fontSize: 13, fontWeight: '600' },
  mode: { color: COLORS.textMuted, fontSize: 12 },
  latency: { color: COLORS.cyan, fontSize: 12 },
  slaves: { color: COLORS.primary, fontSize: 12 },
});
