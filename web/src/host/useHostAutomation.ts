// Everything the host does on its own: the steps in core/hostauto.ts (phase timers, starting
// tie party games, opening the duel vote, the wheel), the AI Draw Off pipeline, and the mascots.
// Timers are re-armed from state, so a refreshed host picks up where it left off; every
// step is a claimed plan, so a timer racing a button can't apply it twice.
import { useEffect, useRef } from 'react';
import * as DO from '../core/drawoff';
import type { DrawOff, DrawOffScore } from '../core/drawoff';
import { hostAuto } from '../core/hostauto';
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

  // --- the host's own steps (core/hostauto.ts): party phases, opening the duel vote, the wheel,
  // in-app duel winners, ties. One timer per step, armed once per key (a changed key re-arms)
  const steps = hostAuto(game);
  const armed = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  useEffect(() => {
    const live = new Set(steps.map((s) => s.key));
    for (const [k, id] of armed.current) if (!live.has(k)) { clearTimeout(id); armed.current.delete(k); }
    for (const s of steps) {
      if (armed.current.has(s.key)) continue;
      const wait = s.at === 0 ? 0 : Math.max(0, s.at - (Date.now() + offset));
      armed.current.set(s.key, setTimeout(() => act((g) => s.run(g, { now: Date.now() + offset, rng: freshRng() }), { quiet: s.quiet }), wait));
    }
  }, [steps.map((s) => s.key).join(',')]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => { for (const id of armed.current.values()) clearTimeout(id); armed.current.clear(); }, []);

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

