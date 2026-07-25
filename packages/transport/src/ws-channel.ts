import type { Channel } from './channel.js';

/**
 * Minimal structural type for a WebSocket-like object. Both the browser `WebSocket` and the Node
 * `ws` library satisfy it, so this adapter runs in the headless-browser worker and in Node services
 * without importing a concrete socket library here.
 */
export interface WebSocketLike {
  send(data: string): void;
  close(): void;
  addEventListener?(type: string, listener: (ev: { data?: unknown }) => void): void;
  on?(event: string, listener: (data: unknown) => void): void;
  readyState?: number;
}

/** Wrap any WebSocket-like object as a MoreGPU Channel. */
export class WebSocketChannel implements Channel {
  closed = false;

  constructor(private readonly ws: WebSocketLike) {}

  send(wire: string): void {
    if (this.closed) return;
    this.ws.send(wire);
  }

  onMessage(handler: (wire: string) => void): void {
    const deliver = (data: unknown) => {
      if (typeof data === 'string') handler(data);
      else if (data && typeof (data as { toString?: () => string }).toString === 'function') {
        handler(String(data));
      }
    };
    // Browser style
    if (typeof this.ws.addEventListener === 'function') {
      this.ws.addEventListener('message', (ev) => deliver(ev.data));
    }
    // Node `ws` style
    if (typeof this.ws.on === 'function') {
      this.ws.on('message', (data) => deliver(data));
    }
  }

  close(): void {
    this.closed = true;
    this.ws.close();
  }
}
