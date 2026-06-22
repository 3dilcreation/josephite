import React, { useMemo } from 'react';
import { View, Image, Text, StyleSheet, Dimensions } from 'react-native';
import Svg, { Line, Circle, Text as SvgText, Rect } from 'react-native-svg';
import { CameraFrame, SensorBundle } from '../types';
import { COLORS } from '../constants';
import { buildAROverlay, OverlayElement } from '../innovations/ARSensorOverlay';

interface Props {
  frame: CameraFrame | null;
  bundle?: SensorBundle | null;
  showAROverlay?: boolean;
  width?: number;
  height?: number;
}

const { width: SCREEN_W } = Dimensions.get('window');
const DEFAULT_W = SCREEN_W - 32;
const DEFAULT_H = (DEFAULT_W * 3) / 4;

export default function CameraFeed({ frame, bundle, showAROverlay = false, width = DEFAULT_W, height = DEFAULT_H }: Props) {
  const overlayElements = useMemo<OverlayElement[]>(() => {
    if (!showAROverlay || !bundle) return [];
    return buildAROverlay(bundle, width, height);
  }, [showAROverlay, bundle, width, height, frame?.timestamp]);

  if (!frame) {
    return (
      <View style={[styles.placeholder, { width, height }]}>
        <Text style={styles.placeholderText}>📷</Text>
        <Text style={styles.placeholderLabel}>Waiting for camera feed…</Text>
      </View>
    );
  }

  return (
    <View style={{ width, height }}>
      <Image
        source={{ uri: `data:image/jpeg;base64,${frame.frame}` }}
        style={{ width, height }}
        resizeMode="cover"
      />

      {showAROverlay && overlayElements.length > 0 && (
        <Svg width={width} height={height} style={StyleSheet.absoluteFill}>
          {overlayElements.map((el, i) => renderOverlayElement(el, i, width, height))}
        </Svg>
      )}

      {/* Frame info bar */}
      <View style={styles.infoBar}>
        <Text style={styles.infoText}>
          #{frame.frameIndex} · {frame.width}×{frame.height} · Q{Math.round(frame.quality * 100)}%
        </Text>
      </View>
    </View>
  );
}

function renderOverlayElement(el: OverlayElement, key: number, w: number, h: number): React.ReactNode {
  const px = el.x * w;
  const py = el.y * h;

  switch (el.type) {
    case 'text':
      return (
        <SvgText
          key={key}
          x={px} y={py}
          fill={el.color ?? COLORS.text}
          fontSize={el.fontSize ?? 14}
          fontWeight="bold"
          stroke="black"
          strokeWidth={0.5}
        >
          {el.value}
        </SvgText>
      );

    case 'bar':
      return (
        <React.Fragment key={key}>
          <Rect x={px} y={py} width={(el.width ?? 0) * w * 0.3} height={6}
            fill={el.color ?? COLORS.primary} rx={3} />
          <SvgText x={px} y={py - 3} fill={COLORS.textMuted} fontSize={9}>{el.label}</SvgText>
        </React.Fragment>
      );

    case 'circle':
      return (
        <React.Fragment key={key}>
          <Circle cx={px} cy={py} r={(el.radius ?? 0.05) * Math.min(w, h)}
            fill="rgba(0,0,0,0.6)" stroke={el.color ?? COLORS.primary} strokeWidth={2} />
          <SvgText x={px} y={py + 5} textAnchor="middle" fill={el.color ?? COLORS.text} fontSize={14} fontWeight="bold">
            {el.value}
          </SvgText>
          <SvgText x={px} y={py + 18} textAnchor="middle" fill={COLORS.textMuted} fontSize={9}>
            {el.label}
          </SvgText>
        </React.Fragment>
      );

    case 'crosshair':
      return (
        <React.Fragment key={key}>
          <Line x1={px - 15} y1={py} x2={px + 15} y2={py} stroke={el.color ?? 'white'} strokeWidth={1} />
          <Line x1={px} y1={py - 15} x2={px} y2={py + 15} stroke={el.color ?? 'white'} strokeWidth={1} />
          <Circle cx={px} cy={py} r={6} stroke={el.color ?? 'white'} strokeWidth={1} fill="none" />
        </React.Fragment>
      );

    default:
      return null;
  }
}

const styles = StyleSheet.create({
  placeholder: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  placeholderText: { fontSize: 48, marginBottom: 8 },
  placeholderLabel: { color: COLORS.textMuted, fontSize: 14 },
  infoBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  infoText: { color: COLORS.textMuted, fontSize: 10 },
});
