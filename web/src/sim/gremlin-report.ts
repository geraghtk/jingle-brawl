// npm run analyze:gremlin [-- --rounds 3000 --players 6]: the Gremlin balance report (gremlin-sim.ts).
import { report } from './gremlin-sim';

const arg = (k: string, d: number) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? Number(process.argv[i + 1]) : d; };
console.log(report(arg('rounds', 3000), arg('players', 6)));
