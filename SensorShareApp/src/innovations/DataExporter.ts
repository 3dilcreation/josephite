/**
 * Innovation #10 — One-Click Data Exporter
 * Exports session data as CSV or JSON and shares via native share sheet.
 */

import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { SensorBundle } from '../types';
import { format } from 'date-fns';

export type ExportFormat = 'csv' | 'json' | 'txt';

function bundleToCsvRow(b: SensorBundle): string {
  const fields = [
    b.timestamp,
    b.userId,
    b.accelerometer?.x ?? '',
    b.accelerometer?.y ?? '',
    b.accelerometer?.z ?? '',
    b.gyroscope?.x ?? '',
    b.gyroscope?.y ?? '',
    b.gyroscope?.z ?? '',
    b.magnetometer?.x ?? '',
    b.magnetometer?.y ?? '',
    b.magnetometer?.z ?? '',
    b.barometer?.pressure ?? '',
    b.location?.latitude ?? '',
    b.location?.longitude ?? '',
    b.location?.altitude ?? '',
    b.location?.speed ?? '',
    b.location?.heading ?? '',
    (b.battery?.level ?? 0) * 100,
    b.battery?.state ?? '',
    b.network?.type ?? '',
    b.network?.isConnected ? 1 : 0,
    b.pedometer?.steps ?? '',
    b.healthScore ?? '',
    b.latencyMs ?? '',
    b.compressionRatio ?? '',
  ];
  return fields.join(',');
}

const CSV_HEADER =
  'timestamp,userId,' +
  'accel_x,accel_y,accel_z,' +
  'gyro_x,gyro_y,gyro_z,' +
  'mag_x,mag_y,mag_z,' +
  'pressure,' +
  'lat,lng,altitude,speed,heading,' +
  'battery_pct,battery_state,' +
  'net_type,net_connected,' +
  'steps,health_score,latency_ms,compression_ratio';

export async function exportSession(
  bundles: SensorBundle[],
  format: ExportFormat = 'csv',
  sessionLabel?: string,
): Promise<string> {
  const label = sessionLabel ?? format('yyyyMMdd_HHmmss')(new Date());
  const fileName = `SensorShare_${label}.${format}`;
  const filePath = FileSystem.documentDirectory + fileName;

  let content = '';

  if (format === 'csv') {
    const rows = [CSV_HEADER, ...bundles.map(bundleToCsvRow)];
    content = rows.join('\n');
  } else if (format === 'json') {
    content = JSON.stringify({ exportedAt: Date.now(), sessionLabel: label, frames: bundles }, null, 2);
  } else {
    // Plain text summary
    const first = bundles[0];
    const last = bundles[bundles.length - 1];
    const duration = last ? ((last.timestamp - first.timestamp) / 1000).toFixed(1) : '0';
    content = [
      `SensorShare Session Export`,
      `=========================`,
      `User ID   : ${first?.userId ?? '-'}`,
      `Device    : ${first?.deviceInfo?.modelName ?? '-'}`,
      `Duration  : ${duration}s`,
      `Frames    : ${bundles.length}`,
      `Avg Health: ${Math.round(bundles.reduce((s, b) => s + (b.healthScore ?? 0), 0) / bundles.length)}`,
      `Avg Latency: ${Math.round(bundles.reduce((s, b) => s + (b.latencyMs ?? 0), 0) / bundles.length)}ms`,
      ``,
      `First GPS : ${first?.location?.latitude?.toFixed(5)}, ${first?.location?.longitude?.toFixed(5)}`,
      `Last GPS  : ${last?.location?.latitude?.toFixed(5)}, ${last?.location?.longitude?.toFixed(5)}`,
      `Steps     : ${last?.pedometer?.steps ?? 0}`,
    ].join('\n');
  }

  await FileSystem.writeAsStringAsync(filePath, content, { encoding: FileSystem.EncodingType.UTF8 });

  if (await Sharing.isAvailableAsync()) {
    await Sharing.shareAsync(filePath, {
      mimeType: format === 'json' ? 'application/json' : 'text/plain',
      dialogTitle: `Share SensorShare Session`,
    });
  }

  return filePath;
}
