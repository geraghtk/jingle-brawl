// Everything the host does on its own: phase timers, the AI Draw Off pipeline, starting
// mini-game tiebreakers, opening the duel vote, and the mascots.
// Timers are re-armed from state, so a refreshed host picks up where it left off; every
// step is a claimed plan, so a timer racing a button can't apply it twice.
import { useEffect, useRef } from 'react';
import * as A from '../core/actions';
import * as DO from '../core/drawoff';
import type { DrawOff, DrawOffScore } from '../core/drawoff';
import { everyoneDone, GAMES } from '../core/party';
import type { PartyGame } from '../core/party';
import { advanceParty } from '../core/partyflow';
import { openVoting } from '../core/voting';
import { duelAppWinner, duelGameStep, SPIN_MS, spinDuelGame } from '../core/duelgame';
import { resolveDuel } from '../core/duel';
import { isMinigameMethod } from '../core/settings';
import { useServerOffset } from '../firebase/game';
import { observe as mascotsObserve } from '../ui/mascots';
import { critique, openaiKeyFor, wittyVerdict } from './critic';
import { freshRng, useHost } from './HostContext';

export function useHostAutomation() {
  const { game, room, act } = useHost();
  const offset = useServerOffset();
  const t = game.currentTurn;
  const mg = game.minigame;

  useEffect(() => { mascotsObserve(game); }, [game]);

  // --- party games: one timer per phase; move on early when everyone's done --------------------
  const party = mg && (mg as PartyGame).type in GAMES ? (mg as PartyGame) : null;
  useEffect(() => {
    if (!party) return;
    const from = party.phase;
    if (everyoneDone(party)) { act((g) => advanceParty(g, from, Date.now() + offset), { quiet: true }); return; }
    const grace = from === 'playing' && party.type === 'whack_elf' ? 1500 : 250; // last scores land
    const wait = Math.max(0, party.phaseEndsAt - (Date.now() + offset)) + grace;
    const id = setTimeout(() => act((g) => advanceParty(g, from, Date.now() + offset), { quiet: true }), wait);
    return () => clearTimeout(id);
  }, [party?.seed, party?.phase, party && everyoneDone(party)]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- duel voting opens as soon as a duel starts (server time, so every phone shows one clock) ----
  // …once the table knows which duel it is (after the wheel stops)
  const step = duelGameStep(game);
  const duelKey = t?.phase === 'duel' ? `${t.challengerId}|${t.defenderId}|${t.giftId}|${!!t.isTiebreak}` : null;
  const voteKey = duelKey && !t!.isTiebreak && t!.voteEndsAt == null && !step ? duelKey : null;
  const spunAt = typeof t?.wheel?.spunAt === 'number' ? t.wheel.spunAt : null;
  useEffect(() => {
    if (!voteKey) return;
    const wait = spunAt == null ? 0 : Math.max(0, spunAt + SPIN_MS - (Date.now() + offset));
    const id = setTimeout(() => act((g) => openVoting(g, Date.now() + offset), { quiet: true }), wait);
    return () => clearTimeout(id);
  }, [voteKey, spunAt]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- the wheel picks the duel when nobody is really defending (or the house always spins) -------
  const spinKey = step === 'wheel' ? duelKey : null;
  useEffect(() => {
    if (spinKey) act((g) => spinDuelGame(g, { now: Date.now() + offset, rng: freshRng() }), { quiet: true });
  }, [spinKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- an in-app duel game decided it: settle the duel with that winner ---------------------------
  const appWinner = duelAppWinner(t);
  useEffect(() => {
    if (appWinner) act((g) => resolveDuel(g, appWinner, Date.now() + offset), { quiet: true });
  }, [appWinner, t?.duelAppStartedAt]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- mini-game tiebreakers start themselves (once per turn) -------------------------------------
  const tieKey = t?.phase === 'tiebreaker_pending' ? `${t.giftId}|${t.openerId}` : null;
  useEffect(() => {
    if (!tieKey || !t || !isMinigameMethod(config(game))) return;
    act((g) => A.startTiebreaker(g, t.tiebreakerC1!, t.tiebreakerC2!, { now: Date.now() + offset, rng: freshRng() }));
  }, [tieKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- AI Draw Off -----------------------------------------------------------------------------------
  const drawOff = mg && (mg as DrawOff).type === 'draw_off' ? (mg as DrawOff) : null;
  const needsDrawOff = t?.phase === 'minigame_draw_off' && !drawOff;
  useEffect(() => {
    if (needsDrawOff) act((g) => DO.startDrawOff(g, { now: Date.now(), rng: freshRng() }));
  }, [needsDrawOff]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!drawOff) return;
    if (drawOff.phase === 'prompt_reveal') {
      const id = setTimeout(() => act((g) => DO.startDrawing(g, Date.now()), { quiet: true }), Math.max(0, drawOff.promptRevealEndTime - Date.now()));
      return () => clearTimeout(id);
    }
    if (drawOff.phase === 'drawing') {
      const id = setTimeout(() => act(DO.startJudging, { quiet: true }), Math.max(0, (drawOff.drawingEndTime || 0) - Date.now()));
      return () => clearTimeout(id);
    }
  }, [drawOff?.startedAt, drawOff?.phase]); // eslint-disable-line react-hooks/exhaustive-deps
  const judging = useRef<number | null>(null);
  useEffect(() => {
    if (drawOff?.phase !== 'judging' || judging.current === drawOff.startedAt) return;
    judging.current = drawOff.startedAt;
    (async () => {
      const url = game.settings?.aiCriticUrl || 'http://localhost:5050';
      const key = game.settings?.aiCriticBackend === 'openai' ? openaiKeyFor(room) : undefined;
      const scores: Record<string, DrawOffScore> = {};
      for (const pid of drawOff.participants) {
        const img = drawOff.submissions?.[pid];
        if (!img) { scores[pid] = { rating: 0, description: 'No submission', critique: 'Did not submit in time' }; continue; }
        try { scores[pid] = await critique(url, img, drawOff.prompt, key); }
        catch (e) { scores[pid] = { rating: 0, description: 'Error', critique: (e as Error).message }; }
      }
      const named = Object.fromEntries(Object.entries(scores).map(([pid, s]) => [pid, { ...s, name: game.players?.[pid]?.name || 'Unknown' }]));
      const verdict = await wittyVerdict(url, drawOff.prompt, named, key);
      act((g) => DO.judged(g, scores, verdict));
    })();
  }, [drawOff?.phase, drawOff?.startedAt]); // eslint-disable-line react-hooks/exhaustive-deps
  // (the duel verdict — chips flying loser → winner — lives in DuelVerdict.tsx)
}

const config = (game: ReturnType<typeof useHost>['game']) => {
  const m = game.settings?.tiebreakerMethod || 'random';
  return m === 'minigame' ? 'sketch_off' : m;
};
