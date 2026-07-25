import { MessageType, makeEnvelope, parseEnvelope, type Envelope } from '@moregpu/protocol';
import type { Channel } from './channel.js';

type Handler = (env: Envelope) => void;
type ErrorHandler = (err: Error, wire: string) => void;

/**
 * Binds a Channel to typed message handling. Incoming wire strings are parsed into validated
 * envelopes and routed by MessageType; malformed frames go to the error handler instead of throwing
 * (a hostile or buggy peer must never crash the loop). Outgoing sends are framed automatically.
 */
export class Dispatcher {
  private readonly handlers = new Map<MessageType, Handler>();
  private errorHandler: ErrorHandler | undefined;

  constructor(private readonly channel: Channel) {
    this.channel.onMessage((wire) => this.dispatch(wire));
  }

  on(type: MessageType, handler: Handler): this {
    this.handlers.set(type, handler);
    return this;
  }

  onError(handler: ErrorHandler): this {
    this.errorHandler = handler;
    return this;
  }

  send<T>(type: MessageType, payload: T): void {
    this.channel.send(JSON.stringify(makeEnvelope(type, payload)));
  }

  private dispatch(wire: string): void {
    let env: Envelope;
    try {
      env = parseEnvelope(wire);
    } catch (err) {
      this.errorHandler?.(err as Error, wire);
      return;
    }
    this.handlers.get(env.type)?.(env);
  }
}
