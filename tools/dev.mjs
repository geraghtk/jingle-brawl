// npm run dev: the room server (:8787, rooms in rooms.db; its debug API answers this PC only, for
// the agents and the live simulator) and the app with hot reload (:5173, reachable from phones on
// this Wi-Fi), together. Ctrl+C stops both.
import { spawn } from 'node:child_process';

const kids = [
  spawn('npx', ['vite-node', 'server/src/node/main.ts', '--', '--port', '8787', '--static', 'none', '--debug-api', ...process.argv.slice(2)], { stdio: 'inherit', shell: true }),
  spawn('npx', ['vite', '--config', 'vite.config.mts', '--host'], { stdio: 'inherit', shell: true }),
];
const stop = () => { for (const k of kids) k.kill(); process.exit(0); };
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
for (const k of kids) k.on('exit', (code) => { if (code) stop(); });
