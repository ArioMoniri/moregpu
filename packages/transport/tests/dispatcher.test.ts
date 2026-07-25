import { describe, it, expect, vi } from 'vitest';
import { MessageType, makeEnvelope } from '@moregpu/protocol';
import { MemoryChannelPair, Dispatcher } from '../src/index.js';

describe('MemoryChannelPair', () => {
  it('delivers a message from one end to the other', async () => {
    const [a, b] = MemoryChannelPair.create();
    const received: string[] = [];
    b.onMessage((wire) => received.push(wire));
    a.send('hello');
    await Promise.resolve();
    expect(received).toEqual(['hello']);
  });

  it('is bidirectional', async () => {
    const [a, b] = MemoryChannelPair.create();
    const gotA: string[] = [];
    a.onMessage((w) => gotA.push(w));
    b.send('from-b');
    await Promise.resolve();
    expect(gotA).toEqual(['from-b']);
  });

  it('stops delivering after close', async () => {
    const [a, b] = MemoryChannelPair.create();
    const got: string[] = [];
    b.onMessage((w) => got.push(w));
    a.close();
    a.send('nope');
    await Promise.resolve();
    expect(got).toEqual([]);
  });
});

describe('Dispatcher', () => {
  it('routes an envelope to the handler registered for its type', async () => {
    const [a, b] = MemoryChannelPair.create();
    const server = new Dispatcher(b);
    const onRegister = vi.fn();
    server.on(MessageType.Register, onRegister);

    a.send(JSON.stringify(makeEnvelope(MessageType.Register, { capability: { nodeId: 'n1' } })));
    await Promise.resolve();

    expect(onRegister).toHaveBeenCalledOnce();
    expect(onRegister.mock.calls[0][0].payload.capability.nodeId).toBe('n1');
  });

  it('ignores malformed wire data without throwing', async () => {
    const [a, b] = MemoryChannelPair.create();
    const server = new Dispatcher(b);
    const onErr = vi.fn();
    server.onError(onErr);

    a.send('not-json');
    await Promise.resolve();

    expect(onErr).toHaveBeenCalledOnce();
  });

  it('does not invoke handlers for other message types', async () => {
    const [a, b] = MemoryChannelPair.create();
    const server = new Dispatcher(b);
    const onHeartbeat = vi.fn();
    server.on(MessageType.Heartbeat, onHeartbeat);

    a.send(JSON.stringify(makeEnvelope(MessageType.Register, {})));
    await Promise.resolve();

    expect(onHeartbeat).not.toHaveBeenCalled();
  });

  it('sends a typed envelope back through the channel', async () => {
    const [a, b] = MemoryChannelPair.create();
    const client = new Dispatcher(a);
    const got: unknown[] = [];
    b.onMessage((w) => got.push(JSON.parse(w)));

    client.send(MessageType.Heartbeat, { load: 0.3 });
    await Promise.resolve();

    expect((got[0] as { type: string }).type).toBe(MessageType.Heartbeat);
    expect((got[0] as { payload: { load: number } }).payload.load).toBe(0.3);
  });
});
