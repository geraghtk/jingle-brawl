// The table on a virtual clock: whole parties through the real engine, every person pressing
// only what's on their own screen, the host's automation on the same steps as the host page.
import { describe, expect, it } from 'vitest';
import { rng } from '../core/rng';
import { HOST } from '../core/table';
import type { Offer } from '../core/table';
import type { Settings } from '../core/types';
import { buttons, Table } from './table';

const people = (n: number) => Array.from({ length: n }, (_, i) => ({ id: `p${i}`, name: `Guest${i}`, skills: { duel: i / n, whack: i / n, sketch: 1 - i / n } }));

/** Everyone presses one of their buttons at random every few seconds; the scripted host runs it. */
function party(seed: number, settings: Settings = {}, n = 6, until = (_t: Table) => false) {
  const t = new Table({ seed, people: people(n), settings, autoHost: { paceMs: 4000 } });
  const r = rng(seed * 7 + 1);
  for (let tick = 0; tick < 4000 && !t.ended && !until(t); tick++) {
    for (const p of t.people) {
      const os = t.offers(p.id);
      if (os.length && r() < 0.5) {
        const res = t.act(p.id, os[Math.floor(r() * os.length)]);
        expect(res.error ?? 'ok').toBe('ok');            // anything offered can be pressed
      }
    }
    t.advance(3000);
  }
  return t;
}

describe('a whole party on the virtual clock', () => {
  it('random guests and the scripted host get from the lobby to the end; everyone ends with a gift', () => {
    for (const seed of [1, 2, 3]) {
      const t = party(seed);
      expect(t.ended).toBe(true);
      const res = t.result();
      expect(res.seats.every((s) => s.gift)).toBe(true);
      expect(res.seats.every((s) => s.chips >= 0)).toBe(true);
      expect(res.minutes).toBeGreaterThan(2);
    }
  });

  it('the same seed and the same presses give the same party', () => {
    expect(JSON.stringify(party(5).result())).toBe(JSON.stringify(party(5).result()));
  });

  it('a tie is a whole-table party game: people outside the tie play it too', () => {
    let played = 0;
    for (let seed = 1; seed <= 12 && !played; seed++) {
      const t = party(seed, { tiebreakerMethod: 'whack_elf' });
      const whackers = t.people.filter((p) => t.did[p.id]?.whack);
      played = whackers.length;
    }
    expect(played).toBeGreaterThan(2);
  });

  it('nobody can press what isn\'t on their screen', () => {
    const t = new Table({ seed: 1, people: people(4), autoHost: { paceMs: 4000 } });
    expect(t.act('p1', { type: 'bid', amount: 2 }).ok).toBe(false); // still in the lobby
    expect(t.act(HOST, { type: 'reveal' }).ok).toBe(false);
    t.advance(10000);                                               // the host starts and draws
    const opener = t.game.currentTurn!.openerId!;
    expect(t.act(opener, { type: 'bid', amount: 1 }).ok).toBe(false); // not on your own gift
  });
});

describe('what people see', () => {
  it('sealed bids stay sealed: no screen shows an amount before the reveal, the TV counts them', () => {
    const t = new Table({ seed: 3, people: people(4), autoHost: { paceMs: 4000, patienceMs: 10 ** 9 } });
    t.advance(10000);
    const opener = t.game.currentTurn!.openerId!;
    const bidder = t.people.find((p) => p.id !== opener)!.id;
    expect(t.act(bidder, { type: 'bid', amount: 3 }).ok).toBe(true);
    for (const p of t.people) {
      const o = t.observe(p.id);
      const text = JSON.stringify([o.tv, p.id === bidder ? null : o.phone]);
      expect(text).not.toMatch(/bid of 3|Bid sealed: 3|bid 3|3 chips/i);
    }
    expect(t.observe(HOST).tv.lines.join(' ')).toContain('Sealed bids: 1 of 3 in');
    expect(t.observe(bidder).phone!.title).toBe('Bid sealed: 3');
    expect(t.log.filter((e) => !e.private).some((e) => /bid of 3/.test(e.text))).toBe(false);
  });

  it('values are private: a phone carries its own valuations only, and looking never changes the party', () => {
    const a = new Table({ seed: 4, people: people(4), autoHost: { paceMs: 4000 } });
    a.advance(10000);
    const v0 = a.observe('p0').values, v1 = a.observe('p1').values;
    expect(Object.keys(v0)).toEqual(Object.keys(v1));
    const b = new Table({ seed: 4, people: people(4), autoHost: { paceMs: 4000 } });
    b.advance(10000);
    expect(b.observe('p0').values).toEqual(v0);                     // same seed, same tastes
  });
});

describe('the host waits for the table', () => {
  it('bids are revealed once everyone has bid, not before (unless patience runs out)', () => {
    const t = new Table({ seed: 6, people: people(4), autoHost: { paceMs: 4000, patienceMs: 60000 } });
    t.advance(10000);
    const turn = t.game.currentTurn!;
    const bidders = t.people.map((p) => p.id).filter((id) => id !== turn.openerId);
    t.act(bidders[0], { type: 'bid', amount: 1 });
    t.advance(30000);
    expect(t.game.currentTurn!.phase).toBe('bidding');              // two still to bid
    for (const id of bidders.slice(1)) t.act(id, { type: 'bid', amount: 0 });
    t.advance(6000);
    expect(t.game.currentTurn!.phase).not.toBe('bidding');
  });

  it('…and a guest who never bids holds the table up only as long as the host is patient', () => {
    const t = new Table({ seed: 6, people: people(4), autoHost: { paceMs: 4000, patienceMs: 60000 } });
    t.advance(10000);
    t.advance(50000);
    expect(t.game.currentTurn!.phase).toBe('bidding');
    t.advance(20000);
    expect(t.game.currentTurn!.phase).not.toBe('bidding');
  });

  it('a guest who never touches their phone: the host covers their own moves (the Manual Override) and the party ends', () => {
    for (const seed of [1, 2, 3, 4]) {
      const t = new Table({ seed, people: people(5), autoHost: { paceMs: 4000, patienceMs: 30000 } });
      const r = rng(seed);
      for (let tick = 0; tick < 3000 && !t.ended; tick++) {
        for (const p of t.people.slice(1)) {                        // p0 is away
          const os = t.offers(p.id);
          if (os.length && r() < 0.5) t.act(p.id, os[Math.floor(r() * os.length)]);
        }
        t.advance(3000);
      }
      expect(t.ended).toBe(true);
      expect(t.did.p0).toBeUndefined();
    }
  });

  it('the duel winner waits for the spectators to call it (vote window on)', () => {
    const t = party(8, { voteWindow: 30, duelPick: 'off' }, 4, (x) => x.game.currentTurn?.phase === 'duel' && !x.game.currentTurn.isTiebreak);
    expect(t.game.currentTurn?.phase).toBe('duel');
    const spectators = t.people.filter((p) => t.offers(p.id).some((o: Offer) => o.type === 'call_duel'));
    expect(spectators.length).toBeGreaterThan(0);
    expect(t.offers(HOST).some((o) => o.type === 'duel_winner')).toBe(false);
  });

  it("a phone's buttons name people by name, even where the app's player ids aren't theirs (the real app)", () => {
    // the live app's players are keyed by its own ids; offers name people by theirs
    const game = { phase: 'main', players: { '-Nx1': { name: 'Ada', chips: 10 }, '-Nx2': { name: 'Bo', chips: 10 } } } as never;
    const ppl = [{ id: 'ada', name: 'Ada' }, { id: 'bo', name: 'Bo' }];
    expect(buttons([{ type: 'call_duel', pick: 'ada' } as Offer, { type: 'vote_sketch', pick: 'bo' } as Offer], game, ppl))
      .toEqual(['backs Ada', "votes for Bo's drawing"]);
  });
});
