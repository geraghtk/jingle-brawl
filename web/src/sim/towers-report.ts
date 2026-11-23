// npm run analyze:towers [-- --battles 200]: the Elf Tower Wars balance report (towers-sim.ts).
import { report } from './towers-sim';

const arg = (k: string, d: number) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? Number(process.argv[i + 1]) : d; };
console.log(report(arg('battles', 200)));
