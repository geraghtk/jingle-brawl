// npm run analyze:mecha [-- --races 300]: the Mecha Santa balance report (mecha-sim.ts).
import { report } from './mecha-sim';

const arg = (k: string, d: number) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? Number(process.argv[i + 1]) : d; };
console.log(report(arg('races', 300)));
