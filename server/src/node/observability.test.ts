// What the server says about itself: JSON log lines, /metrics (this machine or a token only), the
// browsers' error reports, and a crashing press logged without taking the room down.
import { afterEach, describe, expect, it } from 'vitest';
import WebSocket from 'ws';
import type { AddressInfo } from 'node:net';
import { PROTOCOL } from '../../../web/src/net/protocol';
import type { CreateRoomRes, ServerMsg } from '../../../web/src/net/protocol';
import { Rooms } from '../room/rooms';
import { MemoryStore } from '../store/store';
import { FakeClock } from '../testing/fakeClock';
import { clientError, createAppServer, sameSecret } from './http';
import type { ServerOpts } from './http';
import { Metrics, jsonLogger } from './log';

let close: (() => void) | null = null;
afterEach(() => { close?.(); close = null; });

async function boot(opts: Partial<ServerOpts> = {}) {
  const lines: Record<string, unknown>[] = [];
  const rooms = new Rooms(new MemoryStore(), new FakeClock(), { secret: 'test' });
  const app = createAppServer({ rooms, staticDir: null, limits: false, log: jsonLogger((l) => lines.push(JSON.parse(l)), () => 0), ...opts });
  await new Promise<void>((r) => app.server.listen(0, '127.0.0.1', () => r()));
  const base = `http://127.0.0.1:${(app.server.address() as AddressInfo).port}`;
  close = () => { app.close(); rooms.close(); };
  return { base, rooms, lines, app };
}

describe('logs and metrics', () => {
  it('logs one JSON object per line', () => {
    const out: string[] = [];
    jsonLogger((l) => out.push(l), () => 0).info('room.created', { room: 'ABCD' });
    expect(JSON.parse(out[0])).toEqual({ t: '1970-01-01T00:00:00.000Z', level: 'info', event: 'room.created', room: 'ABCD' });
  });

  it('renders counters and gauges in the Prometheus text format', () => {
    const m = new Metrics();
    m.inc('jb_x_total', { kind: 'a"b' });
    m.inc('jb_x_total', { kind: 'a"b' });
    m.gauge('jb_g', () => 3);
    expect(m.render()).toBe('# TYPE jb_x_total counter\njb_x_total{kind="a_b"} 2\n# TYPE jb_g gauge\njb_g 3\n');
  });

  it('/metrics counts rooms and joins for this machine; behind a proxy only with the token', async () => {
    let { base } = await boot();
    const { code } = (await (await fetch(`${base}/api/rooms`, { method: 'POST', body: '{}' })).json()) as CreateRoomRes;
    await fetch(`${base}/api/rooms/${code}/join`, { method: 'POST', body: JSON.stringify({ name: 'Ada' }) });
    const text = await (await fetch(`${base}/metrics`)).text();
    expect(text).toContain('jb_rooms_created_total 1');
    expect(text).toContain('jb_joins_total 1');
    expect(text).toContain('jb_rooms_live 1');
    close?.();
    ({ base } = await boot({ trustProxy: true, metricsToken: 's3cret' }));
    expect((await fetch(`${base}/metrics`)).status).toBe(404);              // everyone looks local through a proxy
    expect((await fetch(`${base}/metrics`, { headers: { Authorization: 'Bearer nope' } })).status).toBe(404);
    expect((await fetch(`${base}/metrics`, { headers: { Authorization: 'Bearer s3cret' } })).status).toBe(200);
    expect(await (await fetch(`${base}/api/info`)).json()).toEqual({ ips: [] });   // no private addresses either
  });

  it('the metrics token is compared whole and in constant time: a near miss of any length is a miss, never an error', async () => {
    expect(sameSecret('Bearer s3cret', 'Bearer s3cret')).toBe(true);
    for (const miss of ['Bearer s3creT', 'Bearer s3cre', 'Bearer s3cret ', 'bearer s3cret', 's3cret', '', 'Bearer s3crét', 'Bearer s3creé', undefined, ['Bearer s3cret'], 7]) {
      expect([miss, sameSecret(miss, 'Bearer s3cret')]).toEqual([miss, false]);
    }
    expect(sameSecret('clé', 'clé')).toBe(true);
    const { base } = await boot({ trustProxy: true, metricsToken: 's3cret' });
    // same length in characters, longer in bytes; shorter; longer; a prefix: all the same 404
    for (const auth of ['Bearer s3crét', 'Bearer s3cre', 'Bearer s3cret0', 'Bearer s', 'Bearer s3creT', 'Basic s3cret']) {
      const res = await fetch(`${base}/metrics`, { headers: { Authorization: Buffer.from(auth, 'utf8').toString('latin1') } });
      expect([auth, res.status]).toEqual([auth, 404]);
    }
    expect((await fetch(`${base}/metrics`, { headers: { Authorization: 'Bearer s3cret' } })).status).toBe(200);
  });

  it("takes a browser's error report: trimmed, counted, logged; never the page's address", async () => {
    const { base, lines } = await boot();
    const r = await fetch(`${base}/api/client-error`, { method: 'POST', body: JSON.stringify({
      page: 'player', message: 'x'.repeat(900), stack: 'at f()', url: 'player.html?room=ABCD&name=Ada' }) });
    expect(r.status).toBe(204);
    const e = lines.find((l) => l.event === 'client.error')!;
    expect(e).toMatchObject({ level: 'warn', page: 'player', stack: 'at f()' });
    expect((e.message as string).length).toBe(500);
    expect(JSON.stringify(e)).not.toContain('ABCD');
    expect(clientError({ page: '<script>' }).page).toBe('unknown');
    expect(await (await fetch(`${base}/metrics`)).text()).toContain('jb_client_errors_total{page="player"} 1');
  });

  it('a press that crashes the engine is refused and logged; the room carries on', async () => {
    const { base, rooms, lines } = await boot();
    const { code, token } = (await (await fetch(`${base}/api/rooms`, { method: 'POST', body: '{}' })).json()) as CreateRoomRes;
    rooms.get(code)!.handle = () => { throw new Error('engine bug'); };
    const ws = new WebSocket(`${base.replace('http', 'ws')}/ws`);
    await new Promise((r) => ws.once('open', r));
    const acks: ServerMsg[] = [];
    ws.on('message', (d) => { const m = JSON.parse(String(d)) as ServerMsg; if (m.t === 'ack') acks.push(m); });
    ws.send(JSON.stringify({ t: 'hello', v: PROTOCOL, room: code, token }));
    ws.send(JSON.stringify({ t: 'intent', id: 'a1', intent: { type: 'start' } }));
    for (let i = 0; i < 50 && !acks.length; i++) await new Promise((r) => setTimeout(r, 10));
    expect(acks[0]).toMatchObject({ ok: false, error: 'Something went wrong. Please try again.' });
    expect(lines.find((l) => l.event === 'intent.crash')).toMatchObject({ level: 'error', room: code, error: 'engine bug' });
    ws.close();
  });
});
