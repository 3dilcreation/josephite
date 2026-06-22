/**
 * BluetoothService — BLE GATT transport (fallback when WiFi unavailable).
 * Slave: peripheral advertising sensor data via notify characteristic.
 * Master: central scanning for slave UUID and subscribing to notify.
 *
 * NOTE: Requires react-native-ble-plx and a Development Build (not Expo Go).
 */

import { BleManager, Device, Characteristic, BleError, State } from 'react-native-ble-plx';
import { Buffer } from 'buffer';
import { WSMessage } from '../types';

const SERVICE_UUID = '4fafc201-1fb5-459e-8fcc-c5c9c331914b';
const SENSOR_CHAR_UUID = 'beb5483e-36e1-4688-b7f5-ea07361b26a8';
const CAMERA_CHAR_UUID = 'beb5483e-36e1-4688-b7f5-ea07361b26a9';
const MTU = 512;

type BLEMessageHandler = (msg: WSMessage) => void;
type BLEStateHandler = (state: 'idle' | 'scanning' | 'connecting' | 'connected' | 'error') => void;

export class BluetoothService {
  private manager: BleManager;
  private onMessage: BLEMessageHandler;
  private onStateChange: BLEStateHandler;
  private connectedDevice: Device | null = null;
  private destroyed = false;

  constructor(onMessage: BLEMessageHandler, onStateChange: BLEStateHandler) {
    this.manager = new BleManager();
    this.onMessage = onMessage;
    this.onStateChange = onStateChange;
  }

  async waitForBLE(): Promise<boolean> {
    return new Promise((resolve) => {
      const sub = this.manager.onStateChange((state: State) => {
        if (state === State.PoweredOn) {
          sub.remove();
          resolve(true);
        } else if (state === State.Unauthorized || state === State.Unsupported) {
          sub.remove();
          resolve(false);
        }
      }, true);
    });
  }

  // ── Master: scan and connect to slave ────────
  async scanAndConnect(slaveUserId: string, timeoutMs = 15000): Promise<boolean> {
    const ready = await this.waitForBLE();
    if (!ready) { this.onStateChange('error'); return false; }
    this.onStateChange('scanning');

    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.manager.stopDeviceScan();
        resolve(false);
      }, timeoutMs);

      this.manager.startDeviceScan(
        [SERVICE_UUID],
        { allowDuplicates: false },
        async (error: BleError | null, device: Device | null) => {
          if (error || !device) return;
          if (!device.name?.includes(slaveUserId)) return;

          this.manager.stopDeviceScan();
          clearTimeout(timer);
          this.onStateChange('connecting');

          try {
            const connected = await device.connect({ requestMTU: MTU });
            await connected.discoverAllServicesAndCharacteristics();
            this.connectedDevice = connected;
            this.onStateChange('connected');
            this.subscribeToCharacteristics(connected);
            resolve(true);
          } catch {
            this.onStateChange('error');
            resolve(false);
          }
        },
      );
    });
  }

  private subscribeToCharacteristics(device: Device): void {
    device.monitorCharacteristicForService(SERVICE_UUID, SENSOR_CHAR_UUID, (err, char) => {
      if (err || !char?.value) return;
      this.decodeAndDispatch(char);
    });
    device.monitorCharacteristicForService(SERVICE_UUID, CAMERA_CHAR_UUID, (err, char) => {
      if (err || !char?.value) return;
      this.decodeAndDispatch(char);
    });
  }

  private decodeAndDispatch(char: Characteristic): void {
    if (!char.value) return;
    try {
      const json = Buffer.from(char.value, 'base64').toString('utf8');
      const msg: WSMessage = JSON.parse(json);
      this.onMessage(msg);
    } catch (e) {
      console.warn('[BLE] decode error', e);
    }
  }

  // ── Slave: send sensor data chunk via BLE ────
  async sendViaBLE(msg: WSMessage, characteristicUUID = SENSOR_CHAR_UUID): Promise<void> {
    if (!this.connectedDevice) return;
    try {
      const json = JSON.stringify(msg);
      const b64 = Buffer.from(json, 'utf8').toString('base64');
      await this.connectedDevice.writeCharacteristicWithResponseForService(
        SERVICE_UUID, characteristicUUID, b64,
      );
    } catch (e) {
      console.warn('[BLE] write error', e);
    }
  }

  disconnect(): void {
    this.destroyed = true;
    this.connectedDevice?.cancelConnection();
    this.connectedDevice = null;
    this.manager.destroy();
  }
}
