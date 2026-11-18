// The Game Lab's picker (the lobby's controls): which mini-game, who plays, how (the modes this game
// supports with that many players), and the two duelists or captains where the mode has them.
// Practice only: nothing is won or lost (core/minigames/lab.ts).
import { useEffect, useRef, useState } from 'react';
import type { MatchMode } from '../core/minigames/contracts';
import { defaultLabMode, isPartyGame, labGames, labModes, NEEDS_DUELISTS, partyFits } from '../core/minigames/lab';
import { getDef } from '../core/minigames/registry';
import { Glyphs } from '../ui/icons';
import { useHost } from './HostContext';

const MODE_NAMES: Record<MatchMode, string> = {
  direct_duel: 'Head to head (two duelists)',
  party_with_duelists: 'Everyone plays, two duelists',
  free_for_all: 'Free for all (practice, no duel)',
  captain_teams: 'Two captains, two teams',
  cooperative: 'Everyone together',
};
const field = { width: '100%', padding: '0.6rem', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-card-hover)' };

export function LabModal({ onClose }: { onClose: () => void }) {
  const { game, send } = useHost();
  const games = labGames();
  const everyone = Object.entries(game.players || {}).filter(([, p]) => p.connected !== false);
  const [gameId, setGameId] = useState(games.find((d) => d.id === 'reindeer_ring_toss')?.id ?? games[0]?.id ?? '');
  const [picked, setPlayers] = useState<string[]>(everyone.map(([id]) => id));
  // someone who arrives while this is open (a bot just added, a phone joining) is in; someone who left is out
  const known = useRef(new Set(everyone.map(([id]) => id)));
  const here = everyone.map(([id]) => id).join(',');
  useEffect(() => {
    const fresh = here.split(',').filter((id) => id && !known.current.has(id));
    fresh.forEach((id) => known.current.add(id));
    if (fresh.length) setPlayers((ps) => [...ps, ...fresh]);
  }, [here]);
  const players = picked.filter((id) => everyone.some(([e]) => e === id));
  const bots = everyone.filter(([, p]) => p.isBot).length;
  const info = games.find((d) => d.id === gameId);
  const party = isPartyGame(gameId);
  const def = party ? undefined : getDef(gameId);
  const modes = def ? labModes(def, players.length) : [];
  const [modePick, setMode] = useState<MatchMode | ''>('');
  const mode = modePick && modes.includes(modePick) ? modePick : def ? defaultLabMode(def, players.length) : undefined;
  const [duelPick, setDuel] = useState<string[] | null>(null);
  const duelists = duelPick && duelPick.every((id) => players.includes(id)) ? duelPick : [players[0], players[1]];   // default: the first two
  const needsDuel = !!mode && NEEDS_DUELISTS.includes(mode);
  const toggle = (id: string) => setPlayers((ps) => (ps.includes(id) ? ps.filter((x) => x !== id) : [...ps, id]));
  const ok = party ? isPartyGame(gameId) && partyFits(gameId, players.length) : !!def && !!mode && (!needsDuel || (duelists[0] !== duelists[1] && duelists.every(Boolean)));
  const start = async () => {
    if (!ok) return;
    if (await send(party ? { type: 'lab_start', gameId, players } : { type: 'lab_start', gameId, players, mode, ...(needsDuel ? { duelists } : {}) })) onClose();
  };
  return (
    <div className="lab-modal" data-testid="lab-modal" style={{ textAlign: 'left' }}>
      <h2><Glyphs>🧪 Game Lab</Glyphs></h2>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Try a mini-game on the real phones and the TV. Practice only: no gifts, chips or points change hands.</p>
      <label style={{ display: 'block', margin: '0.75rem 0 0.35rem', fontWeight: 700 }} htmlFor="lab-game">Game</label>
      <select id="lab-game" data-testid="lab-game" value={gameId} onChange={(e) => setGameId(e.target.value)} style={field}>
        {games.map((d) => <option key={d.id} value={d.id}>{`${d.emoji} ${d.label}`}</option>)}
      </select>
      {info && <div className="hint" style={{ marginTop: 4 }}>{info.rules}</div>}

      <div style={{ margin: '0.75rem 0 0.35rem', fontWeight: 700 }}>Who plays</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 1rem' }}>
        {everyone.map(([id, p]) => (
          <label key={id} style={{ display: 'flex', gap: 6, alignItems: 'center', cursor: 'pointer' }}>
            <input type="checkbox" data-testid={`lab-player-${p.name}`} checked={players.includes(id)} onChange={() => toggle(id)} />{p.name}
          </label>
        ))}
        {!everyone.length && <span className="hint">Nobody has joined yet. Add a bot or two to watch a game on the TV.</span>}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center', marginTop: '0.5rem' }}>
        <button type="button" className="btn btn-secondary" data-testid="lab-add-bot" style={{ padding: '0.4rem 0.8rem' }} onClick={() => send({ type: 'lab_add_bot' })}><Glyphs>🤖 Add a practice bot</Glyphs></button>
        {bots > 0 && <button type="button" className="btn btn-secondary" data-testid="lab-remove-bots" style={{ padding: '0.4rem 0.8rem' }} onClick={() => send({ type: 'lab_remove_bots' })}>Remove the bots</button>}
        <span className="hint" style={{ flex: '1 1 12rem' }}>Bots play like a so-so guest, so you can try a game on your own. They leave when the real game starts.</span>
      </div>

      <label style={{ display: 'block', margin: '0.75rem 0 0.35rem', fontWeight: 700 }} htmlFor="lab-mode">How</label>
      {party ? (
        isPartyGame(gameId) && partyFits(gameId, players.length)
          ? <div className="hint">Everyone picked plays, each on their own phone.</div>
          : <div className="hint" role="alert">{info?.label} needs more players (add a bot).</div>
      ) : modes.length ? (
        <select id="lab-mode" data-testid="lab-mode" value={mode} onChange={(e) => setMode(e.target.value as MatchMode)} style={field}>
          {modes.map((m) => <option key={m} value={m}>{MODE_NAMES[m]}</option>)}
        </select>
      ) : <div className="hint" role="alert">{info?.label} can't be played with {players.length} {players.length === 1 ? 'player' : 'players'}.</div>}

      {needsDuel && (
        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem' }}>
          {[0, 1].map((i) => (
            <label key={i} style={{ flex: 1 }}>
              <span style={{ display: 'block', fontWeight: 700, marginBottom: 4 }}>{mode === 'captain_teams' ? `Captain ${i ? 'B' : 'A'}` : `Duelist ${i + 1}`}</span>
              <select data-testid={`lab-duelist-${i}`} value={duelists[i]} onChange={(e) => { const n = [...duelists]; n[i] = e.target.value; setDuel(n); }} style={field}>
                {players.map((id) => <option key={id} value={id}>{game.players?.[id]?.name}</option>)}
              </select>
            </label>
          ))}
        </div>
      )}
      {needsDuel && duelists[0] === duelists[1] && <div className="hint" role="alert">Pick two different players.</div>}

      <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1.25rem' }}>
        <button className="btn btn-primary" data-testid="lab-start" disabled={!ok} onClick={start}>▶ Start practice</button>
        <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
      </div>
    </div>
  );
}
