// Each screen gets its own view of the game: secrets never reach a screen that shouldn't show them.
import { describe, expect, it } from 'vitest';
import * as A from './actions';
import * as D from './duel';
import { HIDDEN_DRAWING, projectGame } from './projection';
import type { Viewer } from './projection';
import { rng } from './rng';
import { apply, ctx, NOW, table } from './testkit';
import type { Game } from './types';
import { Table } from '../sim/table';

function bidding(settings = {}): Game {
  let g = table(4, settings);
  g.hostUid = 'host-account';
  for (const p of Object.values(g.players!)) p.ownerUid = 'secret-uid';
  g = apply(g, A.drawGift(g, 'Socks', 'gA', ctx(3)));
  const opener = g.currentTurn!.openerId!;
  for (const [i, id] of Object.keys(g.players!).filter((x) => x !== opener).entries()) g = apply(g, A.submitBid(g, id, i + 1, NOW));
  return g;
}
const bidders = (g: Game) => Object.keys(g.currentTurn!.bids!);

describe('projectGame', () => {
  it('sealed bids: a phone sees its own amount; others only that a bid is in; the TV and the host see no amounts', () => {
    const g = bidding();
    const [me, other] = bidders(g);
    const phone = projectGame(g, { kind: 'phone', pid: me });
    expect(phone.currentTurn!.bids![me]).toEqual(g.currentTurn!.bids![me]);
    expect(phone.currentTurn!.bids![other]).toMatchObject({ amount: 0, sealed: true });
    for (const v of [{ kind: 'tv' }, { kind: 'host' }] as Viewer[]) {
      const bids = projectGame(g, v).currentTurn!.bids!;
      expect(Object.keys(bids).sort()).toEqual(bidders(g).sort());   // "3/3 bids in" still shows
      expect(Object.values(bids).every((b) => b.amount === 0 && b.sealed)).toBe(true);
    }
  });
  it('after the reveal everyone sees every bid', () => {
    const g0 = bidding();
    const g = apply(g0, A.revealBids(g0));
    expect(projectGame(g, { kind: 'tv' }).currentTurn!.bids).toEqual(g.currentTurn!.bids);
  });
  it('accounts never leave the server; Undo is the host\'s', () => {
    let g = bidding();
    g = apply(g, A.revealBids(g));
    const top = Object.entries(g.currentTurn!.bids!).sort(([, a], [, b]) => b.amount - a.amount)[0][0];
    g = apply(g, A.startDuel(g, top, g.currentTurn!.openerId!));
    g = apply(g, D.resolveDuel(g, top, NOW));
    expect(g.undo).toBeTruthy();
    for (const v of [{ kind: 'tv' }, { kind: 'phone', pid: 'p1' }, { kind: 'host' }] as Viewer[]) {
      const out = JSON.stringify(projectGame(g, v));
      expect(out).not.toContain('secret-uid');
      expect(out).not.toContain('host-account');
    }
    expect(projectGame(g, { kind: 'host' }).undo).toBeTruthy();
    expect(projectGame(g, { kind: 'tv' }).undo).toBeUndefined();
  });
  it('chips: with public chips off, phones and the TV see only their own until the game ends', () => {
    const g = bidding({ publicChips: false });
    const phone = projectGame(g, { kind: 'phone', pid: 'p2' });
    expect(phone.players!.p2.chips).toBe(g.players!.p2.chips);
    expect(phone.players!.p3.chips).toBeUndefined();
    expect(projectGame(g, { kind: 'tv' }).players!.p2.chips).toBeUndefined();
    expect(projectGame(g, { kind: 'host' }).players!.p2.chips).toBe(g.players!.p2.chips);
    expect(projectGame({ ...g, phase: 'ended' }, { kind: 'tv' }).players!.p2.chips).toBe(g.players!.p2.chips);
    expect(projectGame(bidding({ publicChips: true }), { kind: 'tv' }).players!.p2.chips).toBe(10);
  });
  it("Sketch-Off: nobody sees another's drawing while drawing (the count still shows)", () => {
    const g = table(3);
    g.minigame = { type: 'sketch_off', phase: 'playing', seed: 1, participants: ['p1', 'p2', 'p3'], minigameGoal: 'pick_one', startedAt: NOW, phaseEndsAt: NOW + 1,
      submissions: { p1: 'data:image/jpeg;base64,MINE', p2: 'data:image/jpeg;base64,THEIRS' } } as unknown as Game['minigame'];
    const subs = (v: Viewer) => (projectGame(g, v).minigame as { submissions: Record<string, string> }).submissions;
    expect(subs({ kind: 'phone', pid: 'p1' })).toEqual({ p1: 'data:image/jpeg;base64,MINE', p2: HIDDEN_DRAWING });
    expect(subs({ kind: 'tv' })).toEqual({ p1: HIDDEN_DRAWING, p2: HIDDEN_DRAWING });
  });
});

describe('no screen ever learns a secret (whole simulated parties, every step)', () => {
  it('phones and the TV never see another bid amount before the reveal, an account, or Undo', () => {
    for (const seed of [1, 2, 3, 4]) {
      const people = Array.from({ length: 5 }, (_, i) => ({ id: `p${i}`, name: `Guest${i}`, skills: { duel: 0.5, whack: 0.5, sketch: 0.5 } }));
      const t = new Table({ seed, people, settings: { publicChips: seed % 2 === 0 }, autoHost: { paceMs: 4000 } });
      const r = rng(seed);
      let checked = 0;
      for (let tick = 0; tick < 3000 && !t.ended; tick++) {
        for (const p of t.people) {
          const os = t.offers(p.id);
          if (os.length && r() < 0.5) t.act(p.id, os[Math.floor(r() * os.length)]);
        }
        t.advance(3000);
        const g = t.game;
        const sealed = g.currentTurn && ['bidding', 'endgame_bidding'].includes(g.currentTurn.phase);
        for (const v of [{ kind: 'tv' }, ...people.map((p) => ({ kind: 'phone', pid: p.id }))] as Viewer[]) {
          const view = projectGame(g, v);
          expect(view.undo).toBeUndefined();
          expect(Object.values(view.players || {}).some((p) => p.ownerUid)).toBe(false);
          if (sealed) {
            for (const [id, b] of Object.entries(view.currentTurn!.bids || {})) {
              if (v.kind === 'phone' && v.pid === id) continue;
              expect(b.amount).toBe(0);
              checked++;
            }
          }
        }
      }
      expect(t.ended).toBe(true);
      expect(checked).toBeGreaterThan(0);
    }
  });
});
