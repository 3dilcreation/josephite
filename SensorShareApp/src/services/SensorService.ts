import { Accelerometer, Gyroscope, Magnetometer, Barometer, Pedometer } from 'expo-sensors';
import * as Location from 'expo-location';
import * as Battery from 'expo-battery';
import * as Network from 'expo-network';
import * as Device from 'expo-device';
import {
  AccelerometerData, GyroscopeData, MagnetometerData,
  BarometerData, LocationData, BatteryData, NetworkData,
  PedometerData, DeviceInfo, SensorBundle,
} from '../types';
import { SENSOR_INTERVAL_MS } from '../constants';

type SensorCallback = (bundle: Partial<SensorBundle>) => void;

export class SensorService {
  private subscriptions: { remove: () => void }[] = [];
  private callback: SensorCallback;
  private userId: string;
  private accel: AccelerometerData = { x: 0, y: 0, z: 0 };
  private gyro: GyroscopeData = { x: 0, y: 0, z: 0 };
  private mag: MagnetometerData = { x: 0, y: 0, z: 0 };
  private baro: BarometerData = { pressure: 0 };
  private location: LocationData | undefined;
  private battery: BatteryData = { level: 1, state: 'unknown', lowPowerMode: false };
  private network: NetworkData = { type: 'unknown', isConnected: false };
  private pedometer: PedometerData = { steps: 0 };
  private bundleTimer: ReturnType<typeof setInterval> | null = null;
  private locationSub: Location.LocationSubscription | null = null;
  private deviceInfo: DeviceInfo;
  private intervalMs: number;

  constructor(userId: string, callback: SensorCallback, intervalMs = SENSOR_INTERVAL_MS) {
    this.userId = userId;
    this.callback = callback;
    this.intervalMs = intervalMs;
    this.deviceInfo = {
      brand: Device.brand ?? 'Unknown',
      modelName: Device.modelName ?? 'Unknown',
      osName: Device.osName ?? 'Unknown',
      osVersion: Device.osVersion ?? 'Unknown',
      deviceType: Device.DeviceType[Device.deviceType ?? 0] ?? 'Unknown',
    };
  }

  async requestPermissions(): Promise<boolean> {
    const [locFg, locBg, motion, pedoAvail] = await Promise.all([
      Location.requestForegroundPermissionsAsync(),
      Location.requestBackgroundPermissionsAsync(),
      Pedometer.isAvailableAsync(),
      Pedometer.isAvailableAsync(),
    ]);

    if (locFg.status !== 'granted') {
      console.warn('[SensorService] Location foreground permission denied');
    }
    return true;
  }

  async start(): Promise<void> {
    await this.requestPermissions();

    Accelerometer.setUpdateInterval(this.intervalMs);
    Gyroscope.setUpdateInterval(this.intervalMs);
    Magnetometer.setUpdateInterval(this.intervalMs);
    Barometer.setUpdateInterval(this.intervalMs);

    this.subscriptions.push(
      Accelerometer.addListener((d) => { this.accel = d; }),
      Gyroscope.addListener((d) => { this.gyro = d; }),
      Magnetometer.addListener((d) => { this.mag = d; }),
      Barometer.addListener((d) => { this.baro = { pressure: d.pressure, relativeAltitude: d.relativeAltitude }; }),
    );

    // GPS
    this.locationSub = await Location.watchPositionAsync(
      { accuracy: Location.Accuracy.High, timeInterval: 1000, distanceInterval: 0 },
      (loc) => {
        this.location = {
          latitude: loc.coords.latitude,
          longitude: loc.coords.longitude,
          altitude: loc.coords.altitude,
          speed: loc.coords.speed,
          heading: loc.coords.heading,
          accuracy: loc.coords.accuracy ?? 0,
          timestamp: loc.timestamp,
        };
      },
    );

    // Battery
    const batteryLevel = await Battery.getBatteryLevelAsync();
    const batteryState = await Battery.getBatteryStateAsync();
    const lowPower = await Battery.isLowPowerModeEnabledAsync();
    this.battery = {
      level: batteryLevel,
      state: Battery.BatteryState[batteryState],
      lowPowerMode: lowPower,
    };
    Battery.addBatteryLevelListener(({ batteryLevel: l }) => { this.battery = { ...this.battery, level: l }; });
    Battery.addBatteryStateListener(({ batteryState: s }) => { this.battery = { ...this.battery, state: Battery.BatteryState[s] }; });

    // Network
    const netState = await Network.getNetworkStateAsync();
    const ip = await Network.getIpAddressAsync().catch(() => undefined);
    this.network = { type: netState.type ?? 'unknown', isConnected: netState.isConnected ?? false, ip };

    // Pedometer
    if (await Pedometer.isAvailableAsync()) {
      this.subscriptions.push(
        Pedometer.watchStepCount((r) => { this.pedometer = { steps: r.steps }; }),
      );
    }

    // Bundle emitter
    this.bundleTimer = setInterval(() => this.emit(), this.intervalMs);
  }

  private emit(): void {
    const bundle: Partial<SensorBundle> = {
      userId: this.userId,
      deviceInfo: this.deviceInfo,
      timestamp: Date.now(),
      accelerometer: { ...this.accel },
      gyroscope: { ...this.gyro },
      magnetometer: { ...this.mag },
      barometer: { ...this.baro },
      location: this.location,
      battery: { ...this.battery },
      network: { ...this.network },
      pedometer: { ...this.pedometer },
    };
    this.callback(bundle);
  }

  setInterval(ms: number): void {
    this.intervalMs = ms;
    Accelerometer.setUpdateInterval(ms);
    Gyroscope.setUpdateInterval(ms);
    Magnetometer.setUpdateInterval(ms);
    Barometer.setUpdateInterval(ms);
    if (this.bundleTimer) {
      clearInterval(this.bundleTimer);
      this.bundleTimer = setInterval(() => this.emit(), ms);
    }
  }

  stop(): void {
    this.subscriptions.forEach((s) => s.remove());
    this.subscriptions = [];
    this.locationSub?.remove();
    this.locationSub = null;
    if (this.bundleTimer) clearInterval(this.bundleTimer);
    this.bundleTimer = null;
  }
}
