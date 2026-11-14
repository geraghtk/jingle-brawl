// The UI asks; the rules decide. Screens send intents (send({type: ...})) and never build plans
// themselves, so the room server can validate every press. This test fails if a screen imports
// an action from the rules engine. Read-only helpers (what to show) stay allowed.
// And the two sides stay apart: the host's pages (host/) and the phone's (player/) never import each
// other; what both draw lives in ui/, what both read in core/.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { describe, expect, it } from 'vitest';

const SRC = resolve(__dirname, '..');
const ACTION_MODULES = ['actions', 'duel', 'admin', 'partyflow', 'duelgame', 'voting', 'drawoff'];
/** Exports of those modules a screen may use: they only describe the game. */
const READ_ONLY = new Set([
  'cleanName', 'voteStatus', 'duelGameStep', 'duelOptions', 'duelPicker', 'duelAppWinner', 'SPIN_MS', 'UNDER',
  'DEFAULT_DUEL_GAMES', 'DRAW_OFF_MS', 'PROMPT_REVEAL_MS', 'duelAppInfo', 'isDrawOff', 'everyoneDrew',
  'whoPlays', 'WHO_PLAYS', 'tableGames',
]);

function screens(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) return screens(p);
    return /\.tsx?$/.test(f) && !/\.test\.tsx?$/.test(f) ? [relative(SRC, p).split(sep).join('/')] : [];
  });
}

/** Cross-imports that are meant (file: what it imports, and why). None today: the host's tab on a phone
 *  (player/SeatTabs.tsx) is the console in a frame, not an import. */
const CROSS_ALLOWED = new Set<string>([]);

/** Every module a file imports or re-exports (also for its side effects, or lazily), as a path under web/src. */
function imports(file: string): string[] {
  const src = readFileSync(join(SRC, file), 'utf8');
  const specs = [...src.matchAll(/(?:\bfrom\s+|\bimport\s*\(?\s*)'(\.[^']*)'/g)].map((m) => m[1]);
  return specs.map((s) => relative(SRC, resolve(dirname(join(SRC, file)), s)).split(sep).join('/'));
}

describe('the host side and the phone side stay apart', () => {
  it.each([['host', 'player'], ['player', 'host']])('no file under %s/ imports from %s/', (from, to) => {
    const offenders: string[] = [];
    for (const file of screens(join(SRC, from))) {
      for (const target of imports(file)) {
        if (target.startsWith(`${to}/`) && !CROSS_ALLOWED.has(`${file}: ${target}`)) offenders.push(`${file}: ${target}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});

describe('screens send intents, never plans', () => {
  it('no screen imports an action from the rules engine', () => {
    const offenders: string[] = [];
    for (const file of [...screens(join(SRC, 'host')), ...screens(join(SRC, 'player')), ...screens(join(SRC, 'ui'))]) {
      const src = readFileSync(join(SRC, file), 'utf8');
      for (const m of src.matchAll(/import\s+(type\s+)?([^;]+?)\s+from\s+'\.\.\/core\/([\w]+)'/g)) {
        if (m[1] || !ACTION_MODULES.includes(m[3])) continue;
        if (/\*\s+as/.test(m[2])) { offenders.push(`${file}: import * from core/${m[3]}`); continue; }
        const names = m[2].replace(/[{}]/g, '').split(',').map((n) => n.trim().replace(/^type\s+/, '')).filter(Boolean);
        for (const n of names) if (!READ_ONLY.has(n.split(/\s+as\s+/)[0]) && !m[2].includes(`type ${n}`)) offenders.push(`${file}: ${n} from core/${m[3]}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
