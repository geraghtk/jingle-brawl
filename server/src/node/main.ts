// The room server: `npm start` (or `npx vite-node server/src/node/main.ts -- …`).
//   --port 8787  --host 0.0.0.0 (phones on your Wi-Fi)  --db rooms.db | :memory:
//   --static dist (the built app; off with --static none)  --debug-api (tests only, localhost only)
import { networkInterfaces } from 'node:os';
import { MemoryStore } from '../store/store';
import { SqliteStore } from '../store/sqlite';
import { Rooms } from '../room/rooms';
import { realClock } from './clock';
import { createAppServer } from './http';

const args = process.argv.slice(2);
const flag = (name: string, dflt: string) => { const i = args.indexOf(`--${name}`); return i >= 0 && args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : dflt; };
const port = Number(process.env.PORT || flag('port', '8787'));
const host = flag('host', '0.0.0.0');
const db = flag('db', process.env.JB_DB || 'rooms.db');
const statics = flag('static', 'dist');

const store = db === ':memory:' ? new MemoryStore() : new SqliteStore(db);
const rooms = new Rooms(store, realClock, { secret: process.env.JB_TOKEN_SECRET });
const recovered = rooms.recover();
const app = createAppServer({ rooms, staticDir: statics === 'none' ? null : statics, debugApi: args.includes('--debug-api') });

app.server.listen(port, host, () => {
  const lan = Object.values(networkInterfaces()).flat().find((n) => n && n.family === 'IPv4' && !n.internal)?.address;
  console.log(`Yuel Duel room server on http://localhost:${port}${lan ? `  (phones on this Wi-Fi: http://${lan}:${port}/player.html)` : ''}`);
  console.log(`  rooms: ${db}${recovered ? `, ${recovered} recovered` : ''}${args.includes('--debug-api') ? '  [debug API on]' : ''}`);
});

for (const sig of ['SIGINT', 'SIGTERM'] as const) process.on(sig, () => { app.close(); rooms.close(); process.exit(0); });
