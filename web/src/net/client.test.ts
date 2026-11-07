// A screen's connection to a real room server (in-process): it rejoins by itself, a press made
// while the connection is down goes through once it's back (and only once), and it learns the
// server's clock.
import { afterEach, describe, expect, it } from 'vitest';
import type { AddressInfo } from 'node:net';
import { createAppServer } from '../../../server/src/node/http';
import { Rooms } from '../../../server/src/room/rooms';
import { MemoryStore } from '../../../server/src/store/store';
import { FakeClock, T0 } from '../../../server/src/testing/fakeClock';
import { RoomClient } from './client';

let stop: (() => void)[] = [];
afterEach(() => { for (const s of stop) s(); stop = []; });

async function boot() {
  const clock = new FakeClock();
  const rooms = new Rooms(new MemoryStore(), clock, { secret: 't' });
  const app = createAppServer({ rooms, staticDir: null });
  await new Promise<void>((r) => app.server.listen(0, '127.0.0.1', () => r()));
  const port = (app.server.address() as AddressInfo).port;
  stop.push(() => { app.close(); rooms.close(); });
  return { rooms, url: `ws://127.0.0.1:${port}/ws` };
}

const until = async (cond: () => boolean, ms = 3000) => {
  for (const t0 = Date.now(); !cond(); ) {
    if (Date.now() - t0 > ms) throw new Error('timed out');
    await new Promise((r) => setTimeout(r, 10));
  }
};

describe('RoomClient', () => {
  it('rejoins after a dropped connection; a press made while it was down applies exactly once', async () => {
    const { rooms, url } = await boot();
    const { code, token } = rooms.create({});
    for (const name of ['Ada', 'Bo']) rooms.get(code)!.join({ name });
    const host = new RoomClient(code, token, url).start();
    stop.push(() => host.stop());
    await until(() => host.status === 'open' && !!host.game);
    (host as unknown as { ws: WebSocket }).ws.close();                 // the Wi-Fi blips
    await until(() => host.status !== 'open');
    const pressed = host.send({ type: 'start' });                      // tapped while offline
    await until(() => host.status === 'open');
    expect(await pressed).toEqual({ ok: true });
    await until(() => host.game?.phase === 'main');
    expect(rooms.get(code)!.state.game.phase).toBe('main');
    const before = rooms.get(code)!.state.version;
    expect((await host.send({ type: 'start' })).ok).toBe(false);       // a second Start is refused, not repeated
    expect(rooms.get(code)!.state.version).toBe(before);
  });

  it("learns the server's clock (the offset to this device's)", async () => {
    const { rooms, url } = await boot();
    const { code } = rooms.create({});
    const tv = new RoomClient(code, undefined, url).start();
    stop.push(() => tv.stop());
    await until(() => tv.offset !== 0);
    expect(Math.abs(Date.now() + tv.offset - T0)).toBeLessThan(1000);
  });

  it('a token the room refuses stops the screen trying, with the reason', async () => {
    const { rooms, url } = await boot();
    const { code } = rooms.create({});
    const other = rooms.create({});
    const c = new RoomClient(code, other.token, url).start();
    stop.push(() => c.stop());
    await until(() => !!c.fatal);
    expect(c.fatal).toMatch(/not your room/);
  });
});
