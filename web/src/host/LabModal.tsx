// The Game Lab's picker (the lobby's controls): which mini-game, who plays, how (the modes this game
// supports with that many players), and the two duelists or captains where the mode has them.
// Practice only: nothing is won or lost (core/minigames/lab.ts).
import { useState } from 'react';
import type { MatchMode } from '../core/minigames/contracts';
import { defaultLabMode, labModes, NEEDS_DUELISTS } from '../core/minigames/lab';
import { registered } from '../core/minigames/registry';
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
  const games = registered();
  const everyone = Object.entries(game.players || {}).filter(([, p]) => p.connected !== false);
  const [gameId, setGameId] = useState(games.find((d) => d.id === 'reindeer_ring_toss')?.id ?? games[0]?.id ?? '');
  const [players, setPlayers] = useState<string[]>(everyone.map(([id]) => id));
  const def = games.find((d) => d.id === gameId);
  const modes = def ? labModes(def, players.length) : [];
  const [modePick, setMode] = useState<MatchMode | ''>('');
  const mode = modePick && modes.includes(modePick) ? modePick : def ? defaultLabMode(def, players.length) : undefined;
  const [duelPick, setDuel] = useState<string[] | null>(null);
  const duelists = duelPick && duelPick.every((id) => players.includes(id)) ? duelPick : [players[0], players[1]];   // default: the first two
  const needsDuel = !!mode && NEEDS_DUELISTS.includes(mode);
  const toggle = (id: string) => setPlayers((ps) => (ps.includes(id) ? ps.filter((x) => x !== id) : [...ps, id]));
  const ok = !!def && !!mode && (!needsDuel || (duelists[0] !== duelists[1] && duelists.every(Boolean)));
  const start = async () => {
    if (!ok) return;
    if (await send({ type: 'lab_start', gameId, players, mode, ...(needsDuel ? { duelists } : {}) })) onClose();
  };
  return (
    <div className="lab-modal" data-testid="lab-modal" style={{ textAlign: 'left' }}>
      <h2><Glyphs>🧪 Game Lab</Glyphs></h2>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Try a mini-game on the real phones and the TV. Practice only: no gifts, chips or points change hands.</p>
      <label style={{ display: 'block', margin: '0.75rem 0 0.35rem', fontWeight: 700 }} htmlFor="lab-game">Game</label>
      <select id="lab-game" data-testid="lab-game" value={gameId} onChange={(e) => setGameId(e.target.value)} style={field}>
        {games.map((d) => <option key={d.id} value={d.id}>{`${d.emoji} ${d.label}`}</option>)}
      </select>
      {def && <div className="hint" style={{ marginTop: 4 }}>{def.rules}</div>}

      <div style={{ margin: '0.75rem 0 0.35rem', fontWeight: 700 }}>Who plays</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 1rem' }}>
        {everyone.map(([id, p]) => (
          <label key={id} style={{ display: 'flex', gap: 6, alignItems: 'center', cursor: 'pointer' }}>
            <input type="checkbox" data-testid={`lab-player-${p.name}`} checked={players.includes(id)} onChange={() => toggle(id)} />{p.name}
          </label>
        ))}
      </div>

      <label style={{ display: 'block', margin: '0.75rem 0 0.35rem', fontWeight: 700 }} htmlFor="lab-mode">How</label>
      {modes.length ? (
        <select id="lab-mode" data-testid="lab-mode" value={mode} onChange={(e) => setMode(e.target.value as MatchMode)} style={field}>
          {modes.map((m) => <option key={m} value={m}>{MODE_NAMES[m]}</option>)}
        </select>
      ) : <div className="hint" role="alert">{def?.label} can't be played with {players.length} {players.length === 1 ? 'player' : 'players'}.</div>}

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
