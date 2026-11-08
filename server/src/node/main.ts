// The room server: `npm start` (or `npx vite-node server/src/node/main.ts -- …`).
//   --port 8787  --host 0.0.0.0 (phones on your Wi-Fi)  --db rooms.db | :memory:
//   --static dist (the built app; off with --static none)  --debug-api (tests only, localhost only;
//   also turns off the per-address limits)  --trust-proxy (behind a reverse proxy: client address
//   from X-Forwarded-For; or JB_TRUST_PROXY=1)
// Logs: one JSON object per line on stdout. /metrics answers this machine, or anyone with
// JB_METRICS_TOKEN as a Bearer token.
import { networkInterfaces } from 'node:os';
import { MemoryStore } from '../store/store';
import { SqliteStore } from '../store/sqlite';
import { Rooms } from '../room/rooms';
import { criticSource } from '../critic/source';
import { realClock } from './clock';
import { createAppServer } from './http';
import { jsonLogger } from './log';

const args = process.argv.slice(2);
const flag = (name: string, dflt: string) => { const i = args.indexOf(`--${name}`); return i >= 0 && args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : dflt; };
const port = Number(process.env.PORT || flag('port', '8787'));
const host = flag('host', '0.0.0.0');
const db = flag('db', process.env.JB_DB || 'rooms.db');
const statics = flag('static', 'dist');

const store = db === ':memory:' ? new MemoryStore() : new SqliteStore(db);
const rooms = new Rooms(store, realClock, { secret: process.env.JB_TOKEN_SECRET, critics: criticSource() });
// idle rooms leave memory after 2 hours (forgetting any AI key the host gave); the store keeps them
// and rooms are deleted for good a day after anyone last touched them, or two hours after the game ended
const log = jsonLogger();
const purge = () => { const gone = rooms.purge(); if (gone.length) log.info('rooms.purged', { count: gone.length }); };
setInterval(() => { rooms.sweep(2 * 60 * 60 * 1000); purge(); }, 10 * 60 * 1000).unref();
purge();
const recovered = rooms.recover();
const debugApi = args.includes('--debug-api');
const trustProxy = args.includes('--trust-proxy') || process.env.JB_TRUST_PROXY === '1';
const app = createAppServer({ rooms, staticDir: statics === 'none' ? null : statics, debugApi, trustProxy,
  limits: debugApi ? false : {}, log, metricsToken: process.env.JB_METRICS_TOKEN || undefined });
process.on('uncaughtException', (e) => { log.error('crash', { error: e.message, stack: e.stack?.slice(0, 2000) }); process.exit(1); });
process.on('unhandledRejection', (e) => log.error('unhandled', { error: String((e as Error)?.message ?? e).slice(0, 500) }));

app.server.listen(port, host, () => {
  const lan = Object.values(networkInterfaces()).flat().find((n) => n && n.family === 'IPv4' && !n.internal)?.address;
  const pages = statics === 'none' ? '  (the app: npm run dev serves it on :5173, phones too)'
    : lan ? `  (phones on this Wi-Fi: http://${lan}:${port}/player.html)` : '';
  console.log(`Yuel Duel room server on http://localhost:${port}${pages}`);
  log.info('server.start', { port, db: db === ':memory:' ? 'memory' : 'sqlite', recovered, trustProxy, debugApi });
  console.log(`  rooms: ${db}${recovered ? `, ${recovered} recovered` : ''}${args.includes('--debug-api') ? '  [debug API on]' : ''}`);
});

for (const sig of ['SIGINT', 'SIGTERM'] as const) process.on(sig, () => { app.close(); rooms.close(); process.exit(0); });
