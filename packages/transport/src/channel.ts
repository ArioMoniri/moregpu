/**
 * A Channel is a minimal duplex string pipe. Everything above it (dispatch, envelopes, sealing)
 * is transport-agnostic, so the same coordinator/worker logic runs over WebSocket in production and
 * over an in-memory pair in tests.
 */
export interface Channel {
  send(wire: string): void;
  onMessage(handler: (wire: string) => void): void;
  close(): void;
  readonly closed: boolean;
}

/** A pair of in-memory channels wired to each other. Delivery is async (microtask) to mimic a socket. */
export class MemoryChannel implements Channel {
  private handler: ((wire: string) => void) | undefined;
  private peer: MemoryChannel | undefined;
  closed = false;

  /** @internal */
  link(peer: MemoryChannel): void {
    this.peer = peer;
  }

  send(wire: string): void {
    if (this.closed) return;
    const peer = this.peer;
    if (!peer || peer.closed) return;
    queueMicrotask(() => peer.handler?.(wire));
  }

  onMessage(handler: (wire: string) => void): void {
    this.handler = handler;
  }

  close(): void {
    this.closed = true;
  }
}

export const MemoryChannelPair = {
  create(): [MemoryChannel, MemoryChannel] {
    const a = new MemoryChannel();
    const b = new MemoryChannel();
    a.link(b);
    b.link(a);
    return [a, b];
  },
};
