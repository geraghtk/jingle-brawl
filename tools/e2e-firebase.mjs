// npm run test:e2e:firebase [-- playwright args]: the Playwright suite with the screens on Firebase
// (the emulators), while Firebase is being retired.
import { spawnSync } from 'node:child_process';

const r = spawnSync('npx', ['playwright', 'test', ...process.argv.slice(2)], { stdio: 'inherit', shell: true, env: { ...process.env, TRANSPORT: 'firebase' } });
process.exit(r.status ?? 1);
