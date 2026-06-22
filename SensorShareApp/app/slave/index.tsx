import React, { useEffect, useRef, useCallback, useState } from 'react';
import {
  View, Text, ScrollView, StyleSheet, TouchableOpacity, Switch, Alert,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { router } from 'expo-router';
import { useSlaveStore } from '../../src/store';
import { COLORS, CAMERA_FRAME_INTERVAL_MS } from '../../src/constants';
import { SensorService } from '../../src/services/SensorService';
import { RelayService } from '../../src/services/RelayService';
import { AdaptiveCompressor } from '../../src/innovations/AdaptiveCompressor';
import { SensorReplayEngine } from '../../src/innovations/SensorReplayEngine';
import { computeHealthScore } from '../../src/innovations/HealthScorer';
import { feedAccelerometer, setGestureCallback } from '../../src/innovations/GestureRecognizer';
import { evaluateAndAlert } from '../../src/innovations/VoiceAlert';
import { SensorBundle, WSMessage, CommandPayload, GestureEvent } from '../../src/types';
import ConnectionStatus from '../../src/components/ConnectionStatus';
import SensorCard from '../../src/components/SensorCard';

export default function SlaveScreen() {
  const store = useSlaveStore();
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const sensorRef = useRef<SensorService | null>(null);
  const relayRef = useRef<RelayService | null>(null);
  const compressorRef = useRef(new AdaptiveCompressor());
  const replayRef = useRef(new SensorReplayEngine());
  const frameTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const frameIndex = useRef(0);
  const [isRecording, setIsRecording] = useState(false);
  const [facing, setFacing] = useState<'front' | 'back'>('back');
  const [voiceEnabled, setVoiceEnabled] = useState(true);

  useEffect(() => {
    requestCameraPermission();
    setupGestures();
    return () => cleanup();
  }, []);

  function setupGestures() {
    setGestureCallback((gesture: GestureEvent) => {
      store.setGesture(gesture);
      relayRef.current?.send('SENSOR_DATA', {
        userId: store.userId,
        timestamp: Date.now(),
        gesture,
      } as Partial<SensorBundle>);
    });
  }

  async function startTransmitting() {
    if (!store.userId) { Alert.alert('No User ID'); return; }

    store.setConnectionState('connecting');
    store.setTransmitting(true);

    const relay = new RelayService(
      store.userId,
      'slave',
      handleRelayMessage,
      (state) => {
        const mapped = state === 'connected' ? 'connected'
          : state === 'connecting' ? 'connecting'
          : state === 'error' ? 'error' : 'disconnected';
        store.setConnectionState(mapped);
      },
    );
    relayRef.current = relay;
    relay.connect();

    const sensor = new SensorService(
      store.userId,
      (partial) => handleSensorBundle(partial as SensorBundle),
      compressorRef.current.sensorIntervalMs,
    );
    sensorRef.current = sensor;
    await sensor.start();

    if (cameraPermission?.granted) {
      startCameraStream();
    }
  }

  const handleSensorBundle = useCallback((bundle: SensorBundle) => {
    const health = computeHealthScore(bundle);
    const enriched: SensorBundle = { ...bundle, healthScore: health };

    store.setCurrentBundle(enriched);
    store.setHealthScore(health);

    if (enriched.accelerometer) feedAccelerometer(enriched.accelerometer);
    if (voiceEnabled) evaluateAndAlert(enriched);
    if (isRecording) replayRef.current.record(enriched);

    const start = Date.now();
    relayRef.current?.send('SENSOR_DATA', enriched);
    compressorRef.current.recordTransmission(
      JSON.stringify(enriched).length, Date.now() - start,
    );
  }, [isRecording, voiceEnabled]);

  function startCameraStream() {
    store.setCameraActive(true);
    frameTimer.current = setInterval(async () => {
      if (!cameraRef.current) return;
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: compressorRef.current.jpegQuality,
          base64: true,
          skipProcessing: true,
          exif: false,
        });
        if (!photo?.base64) return;
        const frame = {
          userId: store.userId,
          timestamp: Date.now(),
          frame: photo.base64,
          width: photo.width,
          height: photo.height,
          quality: compressorRef.current.jpegQuality,
          frameIndex: frameIndex.current++,
        };
        relayRef.current?.send('CAMERA_FRAME', frame);
        store.incrementSentFrames();
      } catch { /* camera busy */ }
    }, compressorRef.current.cameraIntervalMs);
  }

  function handleRelayMessage(msg: WSMessage) {
    if (msg.type === 'COMMAND') {
      const cmd = msg.payload as CommandPayload;
      switch (cmd.command) {
        case 'SWITCH_CAMERA':
          setFacing(f => f === 'back' ? 'front' : 'back');
          break;
        case 'FLASH_ON':
          // Handled natively by CameraView prop
          break;
        case 'SET_INTERVAL':
          sensorRef.current?.setInterval(cmd.value as number);
          break;
        case 'START_STREAM':
          if (!store.isCameraActive) startCameraStream();
          break;
        case 'STOP_STREAM':
          stopCameraStream();
          break;
        case 'START_REPLAY':
          startRecording();
          break;
        case 'STOP_REPLAY':
          stopRecording();
          break;
      }
    }
    if (msg.type === 'HEARTBEAT_ACK') {
      const latency = Date.now() - msg.timestamp;
      store.setLatency(latency);
      compressorRef.current.updateLatency(latency);
    }
  }

  function stopCameraStream() {
    if (frameTimer.current) clearInterval(frameTimer.current);
    frameTimer.current = null;
    store.setCameraActive(false);
  }

  function startRecording() {
    replayRef.current.startRecording();
    setIsRecording(true);
  }

  function stopRecording() {
    const session = replayRef.current.stopRecording();
    setIsRecording(false);
    Alert.alert('Session Saved', `Recorded ${session.frames.length} frames (${(replayRef.current.durationMs / 1000).toFixed(1)}s)`);
  }

  function stopTransmitting() {
    stopCameraStream();
    sensorRef.current?.stop();
    relayRef.current?.disconnect();
    store.reset();
  }

  function cleanup() {
    stopCameraStream();
    sensorRef.current?.stop();
    relayRef.current?.disconnect();
  }

  const bundle = store.currentBundle;
  const compressor = compressorRef.current;

  return (
    <ScrollView style={styles.root} contentContainerStyle={styles.scroll}>
      {/* Status bar */}
      <View style={styles.statusRow}>
        <ConnectionStatus
          state={store.connectionState}
          mode={store.connectionMode}
          latencyMs={store.latencyMs}
        />
        <View style={styles.healthBadge}>
          <Text style={[styles.healthValue, { color: store.healthScore >= 70 ? COLORS.success : COLORS.warning }]}>
            {store.healthScore}
          </Text>
          <Text style={styles.healthLabel}>Health</Text>
        </View>
      </View>

      {/* Camera preview */}
      {cameraPermission?.granted && (
        <View style={styles.cameraContainer}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={facing}
          />
          <View style={styles.cameraOverlay}>
            <Text style={styles.cameraLabel}>
              {store.isCameraActive ? `🔴 LIVE · ${store.sentFrameCount} frames` : '⬛ Camera Idle'}
            </Text>
            {isRecording && <Text style={styles.recLabel}>⏺ REC {replayRef.current.frameCount}</Text>}
          </View>
        </View>
      )}

      {/* Controls */}
      <View style={styles.controlRow}>
        {!store.isTransmitting ? (
          <TouchableOpacity style={styles.btnStart} onPress={startTransmitting}>
            <Text style={styles.btnText}>▶ Start Transmitting</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity style={styles.btnStop} onPress={stopTransmitting}>
            <Text style={styles.btnText}>⬛ Stop</Text>
          </TouchableOpacity>
        )}
        <TouchableOpacity style={styles.btnSecondary} onPress={() => setFacing(f => f === 'back' ? 'front' : 'back')}>
          <Text style={styles.btnSecText}>🔄</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.btnSecondary, isRecording && { borderColor: COLORS.danger }]}
          onPress={isRecording ? stopRecording : startRecording}
        >
          <Text style={styles.btnSecText}>{isRecording ? '⏹' : '⏺'}</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.toggleRow}>
        <Text style={styles.toggleLabel}>🔊 Voice Alerts</Text>
        <Switch value={voiceEnabled} onValueChange={setVoiceEnabled} trackColor={{ true: COLORS.primary }} />
      </View>

      {/* Compression info */}
      <View style={styles.compressionRow}>
        <Text style={styles.compressionText}>⚙️ {compressor.describe()}</Text>
      </View>

      {/* Sensor grid */}
      {bundle && (
        <View style={styles.sensorSection}>
          <Text style={styles.sectionTitle}>Live Sensor Data</Text>

          <SensorCard title="Accelerometer" icon="⚡" unit="m/s²" fields={[
            { label: 'X', value: bundle.accelerometer?.x },
            { label: 'Y', value: bundle.accelerometer?.y },
            { label: 'Z', value: bundle.accelerometer?.z },
          ]} />
          <SensorCard title="Gyroscope" icon="🌀" unit="rad/s" fields={[
            { label: 'X', value: bundle.gyroscope?.x },
            { label: 'Y', value: bundle.gyroscope?.y },
            { label: 'Z', value: bundle.gyroscope?.z },
          ]} />
          <SensorCard title="Magnetometer" icon="🧲" unit="μT" fields={[
            { label: 'X', value: bundle.magnetometer?.x },
            { label: 'Y', value: bundle.magnetometer?.y },
            { label: 'Z', value: bundle.magnetometer?.z },
          ]} />
          <SensorCard title="Barometer" icon="🌡️" unit="hPa" fields={[
            { label: 'Pressure', value: bundle.barometer?.pressure },
            { label: 'Altitude', value: bundle.barometer?.relativeAltitude },
          ]} />
          <SensorCard title="GPS / Location" icon="📍" unit="°" fields={[
            { label: 'Lat', value: bundle.location?.latitude },
            { label: 'Lng', value: bundle.location?.longitude },
            { label: 'Alt (m)', value: bundle.location?.altitude },
            { label: 'Speed m/s', value: bundle.location?.speed },
            { label: 'Heading°', value: bundle.location?.heading },
            { label: 'Accuracy', value: bundle.location?.accuracy },
          ]} />
          <SensorCard title="Battery" icon="🔋" unit="" fields={[
            { label: 'Level', value: bundle.battery ? `${Math.round(bundle.battery.level * 100)}%` : null },
            { label: 'State', value: bundle.battery?.state },
            { label: 'Low Power', value: bundle.battery?.lowPowerMode ? 'Yes' : 'No' },
          ]} />
          <SensorCard title="Network" icon="📶" unit="" fields={[
            { label: 'Type', value: bundle.network?.type },
            { label: 'Connected', value: bundle.network?.isConnected ? 'Yes' : 'No' },
            { label: 'IP', value: bundle.network?.ip },
          ]} />
          <SensorCard title="Pedometer" icon="👟" unit="steps" fields={[
            { label: 'Steps', value: bundle.pedometer?.steps },
          ]} />

          {store.gesture && (
            <View style={styles.gestureCard}>
              <Text style={styles.gestureText}>
                👋 Gesture: {store.gesture.type} ({Math.round(store.gesture.confidence * 100)}%)
              </Text>
            </View>
          )}
        </View>
      )}

      <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
        <Text style={styles.backText}>← Back to Role Select</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.bg },
  scroll: { padding: 16, paddingTop: 12 },
  statusRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  healthBadge: { alignItems: 'center', backgroundColor: COLORS.surface, borderRadius: 10, padding: 8, borderWidth: 1, borderColor: COLORS.border, minWidth: 60 },
  healthValue: { fontSize: 20, fontWeight: '800' },
  healthLabel: { color: COLORS.textMuted, fontSize: 10, fontWeight: '700' },
  cameraContainer: { height: 220, borderRadius: 12, overflow: 'hidden', marginBottom: 12, position: 'relative' },
  camera: { flex: 1 },
  cameraOverlay: { position: 'absolute', bottom: 0, left: 0, right: 0, flexDirection: 'row', justifyContent: 'space-between', backgroundColor: 'rgba(0,0,0,0.5)', padding: 8 },
  cameraLabel: { color: '#fff', fontSize: 12, fontWeight: '600' },
  recLabel: { color: COLORS.danger, fontSize: 12, fontWeight: '700' },
  controlRow: { flexDirection: 'row', gap: 10, marginBottom: 10 },
  btnStart: { flex: 1, backgroundColor: COLORS.success, borderRadius: 10, padding: 14, alignItems: 'center' },
  btnStop: { flex: 1, backgroundColor: COLORS.danger, borderRadius: 10, padding: 14, alignItems: 'center' },
  btnText: { color: '#fff', fontSize: 14, fontWeight: '700' },
  btnSecondary: { backgroundColor: COLORS.surface, borderRadius: 10, padding: 14, alignItems: 'center', borderWidth: 1, borderColor: COLORS.border, width: 50 },
  btnSecText: { fontSize: 18 },
  toggleRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', backgroundColor: COLORS.surface, borderRadius: 10, padding: 12, marginBottom: 10, borderWidth: 1, borderColor: COLORS.border },
  toggleLabel: { color: COLORS.text, fontSize: 14 },
  compressionRow: { backgroundColor: COLORS.surface, borderRadius: 8, padding: 10, marginBottom: 16, borderWidth: 1, borderColor: COLORS.border },
  compressionText: { color: COLORS.textMuted, fontSize: 12 },
  sensorSection: { marginBottom: 16 },
  sectionTitle: { color: COLORS.textMuted, fontSize: 12, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 },
  gestureCard: { backgroundColor: COLORS.surface, borderRadius: 10, borderWidth: 1, borderColor: COLORS.accent, padding: 12, marginBottom: 10 },
  gestureText: { color: COLORS.text, fontSize: 14, fontWeight: '600' },
  backBtn: { alignItems: 'center', padding: 16 },
  backText: { color: COLORS.textMuted, fontSize: 14 },
});
