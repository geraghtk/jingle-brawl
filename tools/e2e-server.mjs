// npm run test:e2e:server [-- playwright args]: the Playwright suite with the screens on the room server.
import { spawnSync } from 'node:child_process';

const r = spawnSync('npx', ['playwright', 'test', ...process.argv.slice(2)], { stdio: 'inherit', shell: true, env: { ...process.env, TRANSPORT: 'server' } });
process.exit(r.status ?? 1);
