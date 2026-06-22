/**
 * RelayService — WebSocket relay transport.
 * Both Master and Slave connect to the relay server.
 * Slave joins room `slave:<userId>`, Master subscribes to it.
 */

import { WSMessage, WSMessageType, HandshakePayload, DeviceRole } from '../types';
import { RELAY_SERVER_URL, HEARTBEAT_INTERVAL_MS } from '../constants';

type MessageHandler = (msg: WSMessage) => void;
type StateHandler = (state: 'connecting' | 'connected' | 'disconnected' | 'error') => void;

export class RelayService {
  private ws: WebSocket | null = null;
  private userId: string;
  private role: DeviceRole;
  private onMessage: MessageHandler;
  private onStateChange: StateHandler;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private destroyed = false;
  private serverUrl: string;
  private pendingMessages: WSMessage[] = [];

  constructor(
    userId: string,
    role: DeviceRole,
    onMessage: MessageHandler,
    onStateChange: StateHandler,
    serverUrl = RELAY_SERVER_URL,
  ) {
    this.userId = userId;
    this.role = role;
    this.onMessage = onMessage;
    this.onStateChange = onStateChange;
    this.serverUrl = serverUrl;
  }

  connect(): void {
    if (this.destroyed) return;
    this.onStateChange('connecting');
    try {
      this.ws = new WebSocket(this.serverUrl);
    } catch (e) {
      this.onStateChange('error');
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.onStateChange('connected');
      this.sendHandshake();
      this.startHeartbeat();
      this.flushPending();
    };

    this.ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data as string);
        if (msg.type === 'HEARTBEAT_ACK') return;
        this.onMessage(msg);
      } catch (e) {
        console.warn('[RelayService] Parse error', e);
      }
    };

    this.ws.onerror = () => {
      this.onStateChange('error');
    };

    this.ws.onclose = () => {
      this.stopHeartbeat();
      if (!this.destroyed) {
        this.onStateChange('disconnected');
        this.scheduleReconnect();
      }
    };
  }

  private sendHandshake(): void {
    const payload: HandshakePayload = {
      userId: this.userId,
      role: this.role,
      deviceInfo: { brand: '', modelName: '', osName: '', osVersion: '', deviceType: '' },
      capabilities: ['sensors', 'camera', 'bluetooth'],
    };
    this.send('HANDSHAKE', payload);
  }

  send<T>(type: WSMessageType, payload: T): void {
    const msg: WSMessage<T> = {
      type,
      payload,
      timestamp: Date.now(),
      senderId: this.userId,
    };
    const data = JSON.stringify(msg);
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    } else {
      this.pendingMessages.push(msg as WSMessage);
    }
  }

  private flushPending(): void {
    while (this.pendingMessages.length > 0) {
      const msg = this.pendingMessages.shift()!;
      this.ws?.send(JSON.stringify(msg));
    }
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      this.send('HEARTBEAT', { ts: Date.now() });
    }, HEARTBEAT_INTERVAL_MS);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
    this.heartbeatTimer = null;
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
    const delay = Math.min(1000 * 2 ** this.reconnectAttempts, 30000);
    this.reconnectAttempts++;
    this.reconnectTimer = setTimeout(() => this.connect(), delay);
  }

  disconnect(): void {
    this.destroyed = true;
    this.stopHeartbeat();
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
