// The UI asks; the rules decide. Screens send intents (send({type: ...})) and never build plans
// themselves, so the room server can validate every press. This test fails if a screen imports
// an action from the rules engine. Read-only helpers (what to show) stay allowed.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';
import { describe, expect, it } from 'vitest';

const SRC = resolve(__dirname, '..');
const ACTION_MODULES = ['actions', 'duel', 'admin', 'partyflow', 'duelgame', 'voting', 'drawoff'];
/** Exports of those modules a screen may use: they only describe the game. */
const READ_ONLY = new Set([
  'cleanName', 'voteStatus', 'duelGameStep', 'duelOptions', 'duelPicker', 'duelAppWinner', 'SPIN_MS', 'UNDER',
  'DEFAULT_DUEL_GAMES', 'DRAW_OFF_MS', 'PROMPT_REVEAL_MS',
]);
/** Still writing directly until the room server takes them over (plan: S4 create/join, S6 automation). */
const NOT_YET = new Set(['host/App.tsx', 'player/App.tsx', 'host/HostContext.tsx', 'host/useHostAutomation.ts']);

function screens(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) return screens(p);
    return /\.tsx?$/.test(f) && !/\.test\.tsx?$/.test(f) ? [relative(SRC, p).split(sep).join('/')] : [];
  });
}

describe('screens send intents, never plans', () => {
  it('no screen imports an action from the rules engine', () => {
    const offenders: string[] = [];
    for (const file of [...screens(join(SRC, 'host')), ...screens(join(SRC, 'player')), ...screens(join(SRC, 'ui'))]) {
      if (NOT_YET.has(file)) continue;
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
