import React, { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, ScrollView,
  StyleSheet, KeyboardAvoidingView, Platform, Alert,
} from 'react-native';
import { router } from 'expo-router';
import { useMasterStore, useSlaveStore } from '../src/store';
import { COLORS, INNOVATIONS } from '../src/constants';
import { ConnectionMode } from '../src/types';

const MODES: { id: ConnectionMode; label: string; icon: string; desc: string }[] = [
  { id: 'relay',     label: 'Cloud Relay', icon: '☁️', desc: 'Via relay server — works anywhere' },
  { id: 'wifi',      label: 'Local WiFi',  icon: '📡', desc: 'Same network, lowest latency' },
  { id: 'bluetooth', label: 'Bluetooth',   icon: '🔵', desc: 'No internet needed, short range' },
];

export default function HomeScreen() {
  const [userId, setUserId] = useState('');
  const [role, setRole] = useState<'master' | 'slave' | null>(null);
  const [mode, setMode] = useState<ConnectionMode>('relay');
  const [relayUrl, setRelayUrl] = useState('wss://sensorshare-relay.example.com');
  const [showInnovations, setShowInnovations] = useState(false);

  const masterStore = useMasterStore();
  const slaveStore = useSlaveStore();

  function proceed() {
    if (!userId.trim()) {
      Alert.alert('User ID Required', 'Please enter a unique user ID for this device.');
      return;
    }
    if (!role) {
      Alert.alert('Select Role', 'Choose Master (receive) or Slave (transmit).');
      return;
    }

    if (role === 'master') {
      masterStore.setUserId(userId.trim());
      masterStore.setConnectionMode(mode);
      router.push('/master');
    } else {
      slaveStore.setUserId(userId.trim());
      slaveStore.setConnectionMode(mode);
      router.push('/slave');
    }
  }

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">

        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.logoIcon}>📡</Text>
          <Text style={styles.logoTitle}>SensorShare</Text>
          <Text style={styles.logoSubtitle}>Real-time mobile sensor & camera bridge</Text>
        </View>

        {/* User ID */}
        <View style={styles.section}>
          <Text style={styles.label}>Your User ID</Text>
          <TextInput
            style={styles.input}
            value={userId}
            onChangeText={setUserId}
            placeholder="e.g. alice-phone-01"
            placeholderTextColor={COLORS.textMuted}
            autoCapitalize="none"
            autoCorrect={false}
          />
          <Text style={styles.hint}>Slave and Master pair by matching User IDs.</Text>
        </View>

        {/* Role Selection */}
        <View style={styles.section}>
          <Text style={styles.label}>Device Role</Text>
          <View style={styles.roleRow}>
            <TouchableOpacity
              style={[styles.roleCard, role === 'master' && styles.roleCardActive]}
              onPress={() => setRole('master')}
            >
              <Text style={styles.roleIcon}>👑</Text>
              <Text style={styles.roleTitle}>Master</Text>
              <Text style={styles.roleDesc}>Receive & monitor data from slave devices</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.roleCard, role === 'slave' && styles.roleCardActive]}
              onPress={() => setRole('slave')}
            >
              <Text style={styles.roleIcon}>📲</Text>
              <Text style={styles.roleTitle}>Slave</Text>
              <Text style={styles.roleDesc}>Transmit sensors & camera to master</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Connection Mode */}
        <View style={styles.section}>
          <Text style={styles.label}>Connection Mode</Text>
          {MODES.map((m) => (
            <TouchableOpacity
              key={m.id}
              style={[styles.modeRow, mode === m.id && styles.modeRowActive]}
              onPress={() => setMode(m.id)}
            >
              <Text style={styles.modeIcon}>{m.icon}</Text>
              <View style={{ flex: 1 }}>
                <Text style={styles.modeLabel}>{m.label}</Text>
                <Text style={styles.modeDesc}>{m.desc}</Text>
              </View>
              <View style={[styles.radio, mode === m.id && styles.radioActive]} />
            </TouchableOpacity>
          ))}
          {mode === 'relay' && (
            <TextInput
              style={[styles.input, { marginTop: 8 }]}
              value={relayUrl}
              onChangeText={setRelayUrl}
              placeholder="wss://your-relay-server.com"
              placeholderTextColor={COLORS.textMuted}
              autoCapitalize="none"
              autoCorrect={false}
            />
          )}
        </View>

        {/* Launch */}
        <TouchableOpacity style={[styles.btn, !role && styles.btnDisabled]} onPress={proceed}>
          <Text style={styles.btnText}>
            {role === 'master' ? '👑 Open Master Dashboard' : role === 'slave' ? '📲 Start Transmitting' : 'Select a role above'}
          </Text>
        </TouchableOpacity>

        {/* Innovations Toggle */}
        <TouchableOpacity onPress={() => setShowInnovations(!showInnovations)} style={styles.toggleBtn}>
          <Text style={styles.toggleBtnText}>
            {showInnovations ? '▲ Hide' : '▼ Show'} 10 Built-in Innovations
          </Text>
        </TouchableOpacity>

        {showInnovations && (
          <View style={styles.innovationsGrid}>
            {INNOVATIONS.map((inn) => (
              <View key={inn.id} style={styles.innovationCard}>
                <Text style={styles.innIcon}>{inn.icon}</Text>
                <Text style={styles.innId}>#{inn.id}</Text>
                <Text style={styles.innName}>{inn.name}</Text>
                <Text style={styles.innDesc}>{inn.description}</Text>
              </View>
            ))}
          </View>
        )}

        <View style={{ height: 40 }} />
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.bg },
  scroll: { padding: 20, paddingTop: 60 },
  header: { alignItems: 'center', marginBottom: 32 },
  logoIcon: { fontSize: 56, marginBottom: 8 },
  logoTitle: { fontSize: 30, fontWeight: '800', color: COLORS.text, letterSpacing: 1 },
  logoSubtitle: { color: COLORS.textMuted, fontSize: 14, marginTop: 4 },
  section: { marginBottom: 24 },
  label: { color: COLORS.textMuted, fontSize: 12, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 },
  input: {
    backgroundColor: COLORS.surface, borderRadius: 10, borderWidth: 1,
    borderColor: COLORS.border, color: COLORS.text, fontSize: 15,
    paddingHorizontal: 14, paddingVertical: 12,
  },
  hint: { color: COLORS.textMuted, fontSize: 12, marginTop: 6, marginLeft: 2 },
  roleRow: { flexDirection: 'row', gap: 12 },
  roleCard: {
    flex: 1, backgroundColor: COLORS.surface, borderRadius: 12,
    borderWidth: 2, borderColor: COLORS.border, padding: 16, alignItems: 'center',
  },
  roleCardActive: { borderColor: COLORS.primary },
  roleIcon: { fontSize: 32, marginBottom: 8 },
  roleTitle: { color: COLORS.text, fontSize: 16, fontWeight: '700', marginBottom: 4 },
  roleDesc: { color: COLORS.textMuted, fontSize: 12, textAlign: 'center' },
  modeRow: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: COLORS.surface,
    borderRadius: 10, borderWidth: 1, borderColor: COLORS.border,
    padding: 12, marginBottom: 8,
  },
  modeRowActive: { borderColor: COLORS.primary },
  modeIcon: { fontSize: 22, marginRight: 12 },
  modeLabel: { color: COLORS.text, fontSize: 14, fontWeight: '600' },
  modeDesc: { color: COLORS.textMuted, fontSize: 12 },
  radio: { width: 18, height: 18, borderRadius: 9, borderWidth: 2, borderColor: COLORS.border },
  radioActive: { borderColor: COLORS.primary, backgroundColor: COLORS.primary },
  btn: {
    backgroundColor: COLORS.primary, borderRadius: 12,
    padding: 16, alignItems: 'center', marginBottom: 12,
  },
  btnDisabled: { backgroundColor: COLORS.border },
  btnText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  toggleBtn: { alignItems: 'center', padding: 10 },
  toggleBtnText: { color: COLORS.primary, fontSize: 14, fontWeight: '600' },
  innovationsGrid: { gap: 10 },
  innovationCard: {
    backgroundColor: COLORS.surface, borderRadius: 10, borderWidth: 1,
    borderColor: COLORS.border, padding: 14,
  },
  innIcon: { fontSize: 22, marginBottom: 4 },
  innId: { color: COLORS.textMuted, fontSize: 10, fontWeight: '700' },
  innName: { color: COLORS.text, fontSize: 14, fontWeight: '700', marginBottom: 4 },
  innDesc: { color: COLORS.textMuted, fontSize: 12, lineHeight: 17 },
});
