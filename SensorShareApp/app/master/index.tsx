import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  View, Text, ScrollView, StyleSheet, TouchableOpacity,
  TextInput, Switch, Alert, FlatList,
} from 'react-native';
import { router } from 'expo-router';
import { useMasterStore } from '../../src/store';
import { COLORS, INNOVATIONS } from '../../src/constants';
import { RelayService } from '../../src/services/RelayService';
import { detectAnomalies } from '../../src/innovations/AnomalyDetector';
import { computeHealthScore, getScoreColor, getScoreLabel } from '../../src/innovations/HealthScorer';
import { TimeSyncProtocol } from '../../src/innovations/TimeSyncProtocol';
import { MultiSlaveOrchestrator } from '../../src/innovations/MultiSlaveOrchestrator';
import { SensorReplayEngine } from '../../src/innovations/SensorReplayEngine';
import { exportSession } from '../../src/innovations/DataExporter';
import { evaluateAndAlert, alertAnomaly, stopAllAlerts } from '../../src/innovations/VoiceAlert';
import { feedAccelerometer, setGestureCallback } from '../../src/innovations/GestureRecognizer';
import { SensorBundle, CameraFrame, WSMessage, HandshakePayload, CommandPayload, SlaveRecord, GestureEvent } from '../../src/types';
import ConnectionStatus from '../../src/components/ConnectionStatus';
import SensorCard from '../../src/components/SensorCard';
import CameraFeed from '../../src/components/CameraFeed';
import AnomalyAlert from '../../src/components/AnomalyAlert';
import { Accelerometer } from 'expo-sensors';

const TABS = ['Dashboard', 'Camera', 'Anomalies', 'Slaves', 'Replay', 'Innovations'] as const;
type Tab = typeof TABS[number];

export default function MasterScreen() {
  const store = useMasterStore();
  const relayRef = useRef<RelayService | null>(null);
  const timeSyncRef = useRef(new TimeSyncProtocol());
  const orchestratorRef = useRef<MultiSlaveOrchestrator | null>(null);
  const replayRef = useRef(new SensorReplayEngine());
  const masterAccelSub = useRef<{ remove: () => void } | null>(null);
  const [tab, setTab] = useState<Tab>('Dashboard');
  const [targetId, setTargetId] = useState('');
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [replaySpeed, setReplaySpeed] = useState(1.0);
  const [isReplaying, setIsReplaying] = useState(false);
  const [replayBundle, setReplayBundle] = useState<SensorBundle | null>(null);
  const timeSyncTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    startMasterAccelerometer();
    return () => {
      masterAccelSub.current?.remove();
      relayRef.current?.disconnect();
      timeSyncTimer.current && clearInterval(timeSyncTimer.current);
      stopAllAlerts();
    };
  }, []);

  function startMasterAccelerometer() {
    setGestureCallback(handleMasterGesture);
    masterAccelSub.current = Accelerometer.addListener((data) => {
      feedAccelerometer(data);
    });
  }

  function handleMasterGesture(gesture: GestureEvent) {
    if (!store.activeSlaveId || !orchestratorRef.current) return;
    switch (gesture.type) {
      case 'double_tap':
        orchestratorRef.current.sendCommand(store.activeSlaveId, 'SWITCH_CAMERA');
        break;
      case 'shake':
        Alert.alert('Gesture', 'Shake detected — sent SWITCH_CAMERA to slave');
        orchestratorRef.current.sendCommand(store.activeSlaveId, 'SWITCH_CAMERA');
        break;
    }
  }

  async function connect() {
    if (!targetId.trim() && !store.userId) {
      Alert.alert('Enter User ID', 'Enter the slave User ID to subscribe to.');
      return;
    }
    const slaveId = targetId.trim() || store.userId;
    store.setTargetSlaveId(slaveId);
    store.setConnectionState('connecting');

    const orchestrator = new MultiSlaveOrchestrator(
      (targetUserId, type, payload) => {
        relayRef.current?.send(type as any, { ...payload as object, targetUserId });
      },
    );
    orchestratorRef.current = orchestrator;

    const relay = new RelayService(
      `master-${store.userId}`,
      'master',
      handleRelayMessage,
      (state) => {
        const mapped = state === 'connected' ? 'connected'
          : state === 'connecting' ? 'connecting'
          : state === 'error' ? 'error' : 'disconnected';
        store.setConnectionState(mapped);
        if (state === 'connected') {
          relay.send('SUBSCRIBE', { targetUserId: slaveId });
          startTimeSyncLoop(relay);
        }
      },
    );
    relayRef.current = relay;
    relay.connect();
  }

  function startTimeSyncLoop(relay: RelayService) {
    timeSyncTimer.current = setInterval(() => {
      const { requestId, t1 } = timeSyncRef.current.createRequest(`sync-${Date.now()}`);
      relay.send('TIME_SYNC_REQUEST', { requestId, t1 });
    }, 5000);
  }

  const handleRelayMessage = useCallback((msg: WSMessage) => {
    if (msg.type === 'SENSOR_DATA') {
      const bundle = msg.payload as SensorBundle;
      const health = computeHealthScore(bundle);
      const enriched: SensorBundle = {
        ...bundle,
        healthScore: health,
        latencyMs: Date.now() - bundle.timestamp,
      };

      // Anomaly detection
      const anomalies = detectAnomalies(enriched);
      anomalies.forEach((a) => {
        store.addAnomaly(a);
        if (voiceEnabled) alertAnomaly(a);
      });
      enriched.anomalies = anomalies;

      // Orchestrator + store
      orchestratorRef.current?.updateBundle(enriched);
      store.setLatestBundle(enriched);
      store.pushHistory(enriched);

      // Recording for replay
      if (store.isRecording) replayRef.current.record(enriched);

      // Voice rule check
      if (voiceEnabled) evaluateAndAlert(enriched);

      // Update slave record
      if (store.connectedSlaves.find(s => s.userId === bundle.userId)) {
        store.updateSlave(bundle.userId, {
          lastSeen: Date.now(),
          latencyMs: enriched.latencyMs ?? 0,
          healthScore: health,
        });
      }

    } else if (msg.type === 'CAMERA_FRAME') {
      store.setLatestFrame(msg.payload as CameraFrame);

    } else if (msg.type === 'HANDSHAKE') {
      const hs = msg.payload as HandshakePayload;
      if (hs.role === 'slave') {
        const record: SlaveRecord = {
          userId: hs.userId,
          deviceInfo: hs.deviceInfo,
          connectedAt: Date.now(),
          lastSeen: Date.now(),
          latencyMs: 0,
          healthScore: 100,
          isStreaming: true,
        };
        store.addSlave(record);
        orchestratorRef.current?.registerSlave(record);
      }

    } else if (msg.type === 'TIME_SYNC_RESPONSE') {
      const { requestId, t1, t2 } = msg.payload as any;
      const result = timeSyncRef.current.processResponse(requestId, t1, t2);
      if (result) store.updateSlave(msg.senderId, { latencyMs: result.roundTripMs });

    } else if (msg.type === 'SLAVE_LIST') {
      const slaves = msg.payload as SlaveRecord[];
      slaves.forEach(s => {
        store.addSlave(s);
        orchestratorRef.current?.registerSlave(s);
      });
    }
  }, [voiceEnabled, store]);

  function disconnect() {
    timeSyncTimer.current && clearInterval(timeSyncTimer.current);
    relayRef.current?.disconnect();
    store.reset();
    orchestratorRef.current = null;
  }

  function sendCommand(command: CommandPayload['command'], value?: unknown) {
    if (!store.activeSlaveId) return;
    orchestratorRef.current?.sendCommand(store.activeSlaveId, command, value);
  }

  function broadcastCommand(command: CommandPayload['command']) {
    orchestratorRef.current?.broadcastCommand(command);
  }

  function startReplay() {
    const sessions = replayRef.current.sessions_;
    if (sessions.length === 0) { Alert.alert('No sessions', 'Record a session first.'); return; }
    const latest = sessions[0];
    setIsReplaying(true);
    store.setReplaying(true);
    replayRef.current.play(
      latest,
      (bundle, index, total) => {
        setReplayBundle(bundle);
        store.setReplayIndex(index);
      },
      () => {
        setIsReplaying(false);
        store.setReplaying(false);
        setReplayBundle(null);
      },
      replaySpeed,
    );
  }

  function stopReplay() {
    replayRef.current.stop();
    setIsReplaying(false);
    store.setReplaying(false);
    setReplayBundle(null);
  }

  async function handleExport(format: 'csv' | 'json' | 'txt') {
    if (store.sensorHistory.length === 0) {
      Alert.alert('No Data', 'Receive some sensor data first.');
      return;
    }
    try {
      await exportSession(store.sensorHistory, format);
    } catch (e) {
      Alert.alert('Export Error', String(e));
    }
  }

  const displayBundle = isReplaying ? replayBundle : store.latestBundle;
  const anomalySet = new Set((displayBundle?.anomalies ?? []).map(a => `${a.sensor}.${a.field}`));

  return (
    <View style={styles.root}>
      {/* Tab bar */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.tabBar} contentContainerStyle={styles.tabBarContent}>
        {TABS.map(t => (
          <TouchableOpacity key={t} style={[styles.tab, tab === t && styles.tabActive]} onPress={() => setTab(t)}>
            <Text style={[styles.tabText, tab === t && styles.tabTextActive]}>{t}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      <ScrollView style={styles.body} contentContainerStyle={styles.bodyContent}>

        {/* ── DASHBOARD TAB ── */}
        {tab === 'Dashboard' && (
          <>
            <View style={styles.statusRow}>
              <ConnectionStatus state={store.connectionState} mode={store.connectionMode} latencyMs={store.latestBundle?.latencyMs} slaveCount={store.connectedSlaves.length} />
              {store.latestBundle?.healthScore !== undefined && (
                <View style={[styles.healthBadge, { borderColor: getScoreColor(store.latestBundle.healthScore) }]}>
                  <Text style={[styles.healthValue, { color: getScoreColor(store.latestBundle.healthScore) }]}>
                    {store.latestBundle.healthScore}
                  </Text>
                  <Text style={styles.healthLabel}>{getScoreLabel(store.latestBundle.healthScore)}</Text>
                </View>
              )}
            </View>

            {store.connectionState === 'idle' || store.connectionState === 'disconnected' ? (
              <View style={styles.connectPanel}>
                <Text style={styles.connectLabel}>Subscribe to Slave User ID</Text>
                <TextInput
                  style={styles.connectInput}
                  value={targetId}
                  onChangeText={setTargetId}
                  placeholder="e.g. alice-phone-01"
                  placeholderTextColor={COLORS.textMuted}
                  autoCapitalize="none"
                />
                <TouchableOpacity style={styles.connectBtn} onPress={connect}>
                  <Text style={styles.connectBtnText}>🔗 Connect to Slave</Text>
                </TouchableOpacity>
              </View>
            ) : (
              <TouchableOpacity style={styles.disconnectBtn} onPress={disconnect}>
                <Text style={styles.disconnectText}>⬛ Disconnect</Text>
              </TouchableOpacity>
            )}

            {/* Controls */}
            {store.connectionState === 'connected' && (
              <View style={styles.commandGrid}>
                <Text style={styles.sectionTitle}>Remote Commands</Text>
                <View style={styles.cmdRow}>
                  {[
                    { label: '🔄 Flip Camera',   cmd: 'SWITCH_CAMERA' as const },
                    { label: '📸 Start Stream',  cmd: 'START_STREAM' as const },
                    { label: '⬛ Stop Stream',   cmd: 'STOP_STREAM' as const },
                    { label: '⚡ Fast Sensors',  cmd: 'SET_INTERVAL' as const, value: 50 },
                    { label: '🔦 Flash On',      cmd: 'FLASH_ON' as const },
                    { label: '⬛ Flash Off',     cmd: 'FLASH_OFF' as const },
                  ].map(c => (
                    <TouchableOpacity
                      key={c.label}
                      style={styles.cmdBtn}
                      onPress={() => sendCommand(c.cmd, (c as any).value)}
                    >
                      <Text style={styles.cmdBtnText}>{c.label}</Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
            )}

            <View style={styles.toggleRow}>
              <Text style={styles.toggleLabel}>🔊 Voice Alerts</Text>
              <Switch value={voiceEnabled} onValueChange={setVoiceEnabled} trackColor={{ true: COLORS.primary }} />
            </View>

            {/* Time sync */}
            <View style={styles.infoCard}>
              <Text style={styles.infoText}>
                ⏱ Clock offset: {timeSyncRef.current.offset.toFixed(1)}ms · RTT: {timeSyncRef.current.rtt}ms
              </Text>
            </View>

            {/* Recent anomalies */}
            {store.anomalies.slice(0, 3).map((a, i) => (
              <AnomalyAlert key={`${a.timestamp}-${i}`} anomaly={a} />
            ))}

            {displayBundle && (
              <>
                <Text style={styles.sectionTitle}>Live Sensor Data{isReplaying ? ' (REPLAY)' : ''}</Text>
                <SensorCard title="Accelerometer" icon="⚡" unit="m/s²"
                  anomaly={anomalySet.has('accelerometer.x') || anomalySet.has('accelerometer.y') || anomalySet.has('accelerometer.z')}
                  fields={[
                    { label: 'X', value: displayBundle.accelerometer?.x },
                    { label: 'Y', value: displayBundle.accelerometer?.y },
                    { label: 'Z', value: displayBundle.accelerometer?.z },
                  ]} />
                <SensorCard title="Gyroscope" icon="🌀" unit="rad/s"
                  anomaly={anomalySet.has('gyroscope.x')}
                  fields={[
                    { label: 'X', value: displayBundle.gyroscope?.x },
                    { label: 'Y', value: displayBundle.gyroscope?.y },
                    { label: 'Z', value: displayBundle.gyroscope?.z },
                  ]} />
                <SensorCard title="Magnetometer" icon="🧲" unit="μT"
                  fields={[
                    { label: 'X', value: displayBundle.magnetometer?.x },
                    { label: 'Y', value: displayBundle.magnetometer?.y },
                    { label: 'Z', value: displayBundle.magnetometer?.z },
                  ]} />
                <SensorCard title="Barometer" icon="🌡️" unit="hPa"
                  anomaly={anomalySet.has('barometer.pressure')}
                  fields={[
                    { label: 'Pressure', value: displayBundle.barometer?.pressure },
                    { label: 'Alt (m)', value: displayBundle.barometer?.relativeAltitude },
                  ]} />
                <SensorCard title="GPS / Location" icon="📍" unit="°" fields={[
                  { label: 'Lat', value: displayBundle.location?.latitude },
                  { label: 'Lng', value: displayBundle.location?.longitude },
                  { label: 'Alt (m)', value: displayBundle.location?.altitude },
                  { label: 'Speed km/h', value: displayBundle.location?.speed !== null && displayBundle.location?.speed !== undefined ? (displayBundle.location.speed * 3.6) : null },
                  { label: 'Heading°', value: displayBundle.location?.heading },
                  { label: 'Acc (m)', value: displayBundle.location?.accuracy },
                ]} />
                <SensorCard title="Battery" icon="🔋" unit="" fields={[
                  { label: 'Level', value: displayBundle.battery ? `${Math.round(displayBundle.battery.level * 100)}%` : null },
                  { label: 'State', value: displayBundle.battery?.state },
                  { label: 'Low Power', value: displayBundle.battery?.lowPowerMode ? 'Yes' : 'No' },
                ]} />
                <SensorCard title="Network" icon="📶" unit="" fields={[
                  { label: 'Type', value: displayBundle.network?.type },
                  { label: 'Connected', value: displayBundle.network?.isConnected ? 'Yes' : 'No' },
                  { label: 'IP', value: displayBundle.network?.ip },
                ]} />
                <SensorCard title="Pedometer" icon="👟" unit="steps" fields={[
                  { label: 'Steps', value: displayBundle.pedometer?.steps },
                ]} />
                {displayBundle.gesture && (
                  <SensorCard title="Gesture" icon="👋" unit="" fields={[
                    { label: 'Type', value: displayBundle.gesture.type },
                    { label: 'Confidence', value: `${Math.round(displayBundle.gesture.confidence * 100)}%` },
                  ]} />
                )}
              </>
            )}
          </>
        )}

        {/* ── CAMERA TAB ── */}
        {tab === 'Camera' && (
          <>
            <View style={styles.arRow}>
              <Text style={styles.toggleLabel}>🔮 AR Sensor Overlay</Text>
              <Switch
                value={store.showAROverlay}
                onValueChange={() => store.toggleAROverlay()}
                trackColor={{ true: COLORS.accent }}
              />
            </View>
            <CameraFeed
              frame={store.latestFrame}
              bundle={store.latestBundle}
              showAROverlay={store.showAROverlay}
            />
            {store.latestFrame && (
              <View style={styles.infoCard}>
                <Text style={styles.infoText}>
                  Frame #{store.latestFrame.frameIndex} · {new Date(store.latestFrame.timestamp).toLocaleTimeString()}
                </Text>
              </View>
            )}
          </>
        )}

        {/* ── ANOMALIES TAB ── */}
        {tab === 'Anomalies' && (
          <>
            <View style={styles.anomalyHeader}>
              <Text style={styles.sectionTitle}>Anomaly Log ({store.anomalies.length})</Text>
              <TouchableOpacity onPress={() => store.clearAnomalies()}>
                <Text style={styles.clearBtn}>Clear</Text>
              </TouchableOpacity>
            </View>
            {store.anomalies.length === 0 && (
              <Text style={styles.emptyText}>No anomalies detected yet.</Text>
            )}
            {store.anomalies.map((a, i) => (
              <AnomalyAlert key={`${a.timestamp}-${i}`} anomaly={a} />
            ))}
          </>
        )}

        {/* ── SLAVES TAB ── */}
        {tab === 'Slaves' && (
          <>
            <Text style={styles.sectionTitle}>Connected Slaves ({store.connectedSlaves.length} / 8)</Text>
            {store.connectedSlaves.length === 0 && (
              <Text style={styles.emptyText}>No slaves connected. Go to Dashboard and connect.</Text>
            )}
            {store.connectedSlaves.map(slave => (
              <TouchableOpacity
                key={slave.userId}
                style={[styles.slaveCard, store.activeSlaveId === slave.userId && styles.slaveCardActive]}
                onPress={() => store.setActiveSlave(slave.userId)}
              >
                <View style={styles.slaveHeader}>
                  <Text style={styles.slaveId}>{slave.userId}</Text>
                  <View style={[styles.dot, { backgroundColor: Date.now() - slave.lastSeen < 5000 ? COLORS.success : COLORS.danger }]} />
                </View>
                <Text style={styles.slaveMeta}>
                  {slave.deviceInfo.brand} {slave.deviceInfo.modelName} · {slave.deviceInfo.osName}
                </Text>
                <View style={styles.slaveStats}>
                  <Text style={[styles.slaveStat, { color: getScoreColor(slave.healthScore) }]}>Health: {slave.healthScore}</Text>
                  <Text style={styles.slaveStat}>Latency: {slave.latencyMs}ms</Text>
                  <Text style={styles.slaveStat}>Connected: {Math.round((Date.now() - slave.connectedAt) / 1000)}s ago</Text>
                </View>
                {store.activeSlaveId === slave.userId && (
                  <View style={styles.activeLabel}>
                    <Text style={styles.activeLabelText}>ACTIVE</Text>
                  </View>
                )}
              </TouchableOpacity>
            ))}
            {store.connectedSlaves.length > 0 && (
              <View style={styles.infoCard}>
                <Text style={styles.infoText}>
                  Aggregate Health: {orchestratorRef.current?.aggregateHealth ?? 0} · Avg Latency: {orchestratorRef.current?.averageLatency ?? 0}ms
                </Text>
              </View>
            )}
          </>
        )}

        {/* ── REPLAY TAB ── */}
        {tab === 'Replay' && (
          <>
            <Text style={styles.sectionTitle}>Sensor Replay Engine</Text>
            <View style={styles.replayControls}>
              <TouchableOpacity
                style={[styles.replayBtn, store.isRecording && styles.replayBtnRec]}
                onPress={() => {
                  if (store.isRecording) {
                    const session = replayRef.current.stopRecording();
                    store.setRecording(false);
                    store.addReplaySession(session);
                    Alert.alert('Recorded', `${session.frames.length} frames (${(replayRef.current.durationMs / 1000).toFixed(1)}s)`);
                  } else {
                    replayRef.current.startRecording();
                    store.setRecording(true);
                  }
                }}
              >
                <Text style={styles.replayBtnText}>{store.isRecording ? '⏹ Stop Recording' : '⏺ Start Recording'}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.replayBtn, isReplaying && { backgroundColor: COLORS.danger }]} onPress={isReplaying ? stopReplay : startReplay}>
                <Text style={styles.replayBtnText}>{isReplaying ? '⏹ Stop Replay' : '▶ Play Latest'}</Text>
              </TouchableOpacity>
            </View>
            <View style={styles.speedRow}>
              <Text style={styles.speedLabel}>Speed: {replaySpeed}×</Text>
              {[0.25, 0.5, 1, 2, 4].map(s => (
                <TouchableOpacity key={s} style={[styles.speedBtn, replaySpeed === s && styles.speedBtnActive]} onPress={() => setReplaySpeed(s)}>
                  <Text style={styles.speedBtnText}>{s}×</Text>
                </TouchableOpacity>
              ))}
            </View>
            {store.isRecording && (
              <View style={styles.infoCard}>
                <Text style={[styles.infoText, { color: COLORS.danger }]}>
                  ⏺ Recording… {replayRef.current.frameCount} frames
                </Text>
              </View>
            )}
            {isReplaying && replayBundle && (
              <View style={styles.infoCard}>
                <Text style={[styles.infoText, { color: COLORS.warning }]}>
                  ▶ Replaying frame {store.replayIndex} / {replayRef.current.sessions_[0]?.frames.length ?? 0} at {replaySpeed}×
                </Text>
              </View>
            )}
            <View style={styles.exportSection}>
              <Text style={styles.sectionTitle}>Export Session Data</Text>
              <View style={styles.exportRow}>
                {(['csv', 'json', 'txt'] as const).map(f => (
                  <TouchableOpacity key={f} style={styles.exportBtn} onPress={() => handleExport(f)}>
                    <Text style={styles.exportBtnText}>📤 .{f.toUpperCase()}</Text>
                  </TouchableOpacity>
                ))}
              </View>
              <Text style={styles.emptyText}>Exports {store.sensorHistory.length} frames from current session.</Text>
            </View>
          </>
        )}

        {/* ── INNOVATIONS TAB ── */}
        {tab === 'Innovations' && (
          <>
            <Text style={styles.sectionTitle}>10 Built-in Innovations</Text>
            {INNOVATIONS.map(inn => (
              <View key={inn.id} style={styles.innCard}>
                <View style={styles.innHeader}>
                  <Text style={styles.innIcon}>{inn.icon}</Text>
                  <View style={styles.innBadge}><Text style={styles.innBadgeText}>#{inn.id}</Text></View>
                  <Text style={styles.innName}>{inn.name}</Text>
                </View>
                <Text style={styles.innDesc}>{inn.description}</Text>
              </View>
            ))}
          </>
        )}

      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.bg },
  tabBar: { borderBottomWidth: 1, borderBottomColor: COLORS.border, maxHeight: 46 },
  tabBarContent: { paddingHorizontal: 10, alignItems: 'center' },
  tab: { paddingHorizontal: 14, paddingVertical: 12, marginRight: 2 },
  tabActive: { borderBottomWidth: 2, borderBottomColor: COLORS.primary },
  tabText: { color: COLORS.textMuted, fontSize: 13, fontWeight: '600' },
  tabTextActive: { color: COLORS.primary },
  body: { flex: 1 },
  bodyContent: { padding: 16, paddingBottom: 40 },
  statusRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 },
  healthBadge: { alignItems: 'center', borderRadius: 10, borderWidth: 2, padding: 8, minWidth: 70, backgroundColor: COLORS.surface },
  healthValue: { fontSize: 22, fontWeight: '800' },
  healthLabel: { color: COLORS.textMuted, fontSize: 10, fontWeight: '700', marginTop: 2 },
  connectPanel: { backgroundColor: COLORS.surface, borderRadius: 12, padding: 14, borderWidth: 1, borderColor: COLORS.border, marginBottom: 12 },
  connectLabel: { color: COLORS.textMuted, fontSize: 12, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 },
  connectInput: { backgroundColor: COLORS.bg, borderRadius: 8, borderWidth: 1, borderColor: COLORS.border, color: COLORS.text, fontSize: 15, paddingHorizontal: 12, paddingVertical: 10, marginBottom: 10 },
  connectBtn: { backgroundColor: COLORS.primary, borderRadius: 8, padding: 12, alignItems: 'center' },
  connectBtnText: { color: '#fff', fontSize: 14, fontWeight: '700' },
  disconnectBtn: { backgroundColor: COLORS.danger, borderRadius: 10, padding: 12, alignItems: 'center', marginBottom: 12 },
  disconnectText: { color: '#fff', fontSize: 14, fontWeight: '700' },
  commandGrid: { marginBottom: 14 },
  cmdRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  cmdBtn: { backgroundColor: COLORS.surface, borderRadius: 8, borderWidth: 1, borderColor: COLORS.border, paddingHorizontal: 12, paddingVertical: 8 },
  cmdBtnText: { color: COLORS.text, fontSize: 12, fontWeight: '600' },
  toggleRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', backgroundColor: COLORS.surface, borderRadius: 10, padding: 12, marginBottom: 10, borderWidth: 1, borderColor: COLORS.border },
  toggleLabel: { color: COLORS.text, fontSize: 14 },
  infoCard: { backgroundColor: COLORS.surface, borderRadius: 8, padding: 10, marginBottom: 10, borderWidth: 1, borderColor: COLORS.border },
  infoText: { color: COLORS.textMuted, fontSize: 12 },
  sectionTitle: { color: COLORS.textMuted, fontSize: 12, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10, marginTop: 4 },
  arRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  anomalyHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  clearBtn: { color: COLORS.danger, fontSize: 13, fontWeight: '600' },
  emptyText: { color: COLORS.textMuted, fontSize: 14, textAlign: 'center', marginVertical: 20 },
  slaveCard: { backgroundColor: COLORS.surface, borderRadius: 10, borderWidth: 1, borderColor: COLORS.border, padding: 14, marginBottom: 10 },
  slaveCardActive: { borderColor: COLORS.primary },
  slaveHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  slaveId: { color: COLORS.text, fontSize: 14, fontWeight: '700' },
  dot: { width: 8, height: 8, borderRadius: 4 },
  slaveMeta: { color: COLORS.textMuted, fontSize: 12, marginBottom: 6 },
  slaveStats: { flexDirection: 'row', gap: 12 },
  slaveStat: { color: COLORS.text, fontSize: 12 },
  activeLabel: { position: 'absolute', top: 10, right: 10, backgroundColor: COLORS.primary, borderRadius: 4, paddingHorizontal: 6, paddingVertical: 2 },
  activeLabelText: { color: '#fff', fontSize: 9, fontWeight: '800' },
  replayControls: { flexDirection: 'row', gap: 10, marginBottom: 10 },
  replayBtn: { flex: 1, backgroundColor: COLORS.surface, borderRadius: 10, borderWidth: 1, borderColor: COLORS.border, padding: 12, alignItems: 'center' },
  replayBtnRec: { borderColor: COLORS.danger, backgroundColor: 'rgba(231,76,60,0.1)' },
  replayBtnText: { color: COLORS.text, fontSize: 13, fontWeight: '600' },
  speedRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 10 },
  speedLabel: { color: COLORS.textMuted, fontSize: 13, width: 70 },
  speedBtn: { backgroundColor: COLORS.surface, borderRadius: 6, borderWidth: 1, borderColor: COLORS.border, paddingHorizontal: 10, paddingVertical: 6 },
  speedBtnActive: { borderColor: COLORS.primary, backgroundColor: 'rgba(79,142,247,0.15)' },
  speedBtnText: { color: COLORS.text, fontSize: 12, fontWeight: '600' },
  exportSection: { marginTop: 16 },
  exportRow: { flexDirection: 'row', gap: 10, marginBottom: 8 },
  exportBtn: { flex: 1, backgroundColor: COLORS.surface, borderRadius: 8, borderWidth: 1, borderColor: COLORS.border, padding: 12, alignItems: 'center' },
  exportBtnText: { color: COLORS.text, fontSize: 13, fontWeight: '600' },
  innCard: { backgroundColor: COLORS.surface, borderRadius: 10, borderWidth: 1, borderColor: COLORS.border, padding: 14, marginBottom: 10 },
  innHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 8, gap: 8 },
  innIcon: { fontSize: 22 },
  innBadge: { backgroundColor: COLORS.accent, borderRadius: 4, paddingHorizontal: 6, paddingVertical: 2 },
  innBadgeText: { color: '#fff', fontSize: 10, fontWeight: '800' },
  innName: { color: COLORS.text, fontSize: 14, fontWeight: '700', flex: 1 },
  innDesc: { color: COLORS.textMuted, fontSize: 12, lineHeight: 17 },
});
