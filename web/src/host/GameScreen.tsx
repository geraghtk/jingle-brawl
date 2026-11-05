// The TV screen: top strip, turn panel, players / gifts / records, and the sidebar.
import { useState } from 'react';
import type { ReactNode } from 'react';
import { giftless, wrappedGifts } from '../core/rules';
import { minCost } from '../core/settings';
import type { Game } from '../core/types';
import { Avatar, avatarColor } from '../ui/avatar';
import { MvpReveal } from '../ui/MvpReveal';
import { DuelModal } from './DuelModal';
import { DuelVerdict } from './DuelVerdict';
import { useHost } from './HostContext';
import { Modal, ModalCtx } from './Modal';
import { Sidebar } from './Sidebar';
import { TurnPanel } from './TurnPanel';
import { useHostAutomation } from './useHostAutomation';
import { Icon } from '../ui/icons';

export function GameScreen({ onExit }: { onExit: () => void }) {
  const [modal, setModal] = useState<ReactNode | null>(null);
  const [mvpSeen, setMvpSeen] = useState(false);
  const { game } = useHost();
  useHostAutomation();
  return (
    <ModalCtx.Provider value={setModal}>
      <div id="game-screen">
        <Strip />
        <div className="container">
          <div className="main-area">
            <TurnPanel />
            <PlayersCard />
            <GiftsCard />
            <Records />
          </div>
          <Sidebar onExit={onExit} />
        </div>
      </div>
      <Modal content={modal} onClose={() => setModal(null)} />
      <DuelModal />
      <DuelVerdict />
      {game.phase === 'ended' && !mvpSeen && <MvpReveal game={game} variant="tv" onClose={() => setMvpSeen(true)} />}
      <div className="coin-animation-container" id="coin-animation-container" />
    </ModalCtx.Provider>
  );
}

function Strip() {
  const { game, room } = useHost();
  const raw = (game.currentTurn?.phase || game.phase || 'lobby').replace(/_/g, ' ');
  const pot = game.currentTurn?.pot || 0;
  const inDuel = (game.currentTurn?.phase || '').includes('duel');
  return (
    <div className="f6-strip host-strip">
      <div className="cell brand-cell"><img src="ydlogo2.png" alt="Yuel Duel" /><span className="brand-name f6-disp">Yuel Duel</span></div>
      <div className="cell"><span className="k">Phase</span><span className="v f6-num" id="strip-phase">{raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase()}</span></div>
      <div className="cell"><span className="k">Players</span><span className="v f6-num"><span id="connected-count">{Object.keys(game.players || {}).length}</span></span></div>
      <div className="cell"><span className="k">Gifts left</span><span className="v f6-num" id="strip-gifts">{wrappedGifts(game)}</span></div>
      <div className={`cell pot-cell${inDuel && pot > 0 ? '' : ' hidden'}`} id="strip-pot-cell"><span className="k">Pot</span><span className="v f6-num" id="strip-pot">{pot} ◆</span></div>
      <div className="cell room-cell"><span className="k">Room code</span><span className="v f6-num" id="strip-room">{room}</span></div>
    </div>
  );
}

function PlayersCard() {
  const { game, cfg } = useHost();
  if (game.phase === 'lobby') return null;
  const players = Object.entries(game.players || {});
  return (
    <div className="card" id="card-players">
      <div className="card-title">Players</div>
      <div className="players-grid" id="players-grid">
        {players.length === 0
          ? <div style={{ color: 'var(--text-muted)', gridColumn: '1/-1', textAlign: 'center', padding: '2rem' }}>Waiting for players to join...</div>
          : players.map(([id, p]) => (
            <div key={id} className={`player-card ${game.currentTurn?.activePlayerId === id ? 'active' : ''}`} data-player-id={id}>
              <Avatar name={p.name} avatar={p.avatar} size={34} />
              <div className="who"><div className="nm">{p.name}</div><div className={`sub${p.giftId ? ' gift' : ''}`}>{p.giftId ? 'Has gift' : 'In bag'}</div></div>
              {cfg.publicChips && <span className="roster-chips"><span className="f6-coin" /> {p.chips}</span>}
            </div>
          ))}
      </div>
    </div>
  );
}

function GiftsCard() {
  const { game, cfg } = useHost();
  const gifts = Object.entries(game.gifts || {}).sort(([, a], [, b]) => {
    const am = a.ownerId === 'misfit', bm = b.ownerId === 'misfit';
    if (am !== bm) return am ? 1 : -1;
    return am ? (a.misfitOrder || 0) - (b.misfitOrder || 0) : 0;
  });
  if (!gifts.length) return null;
  return (
    <div className="card" id="card-gifts">
      <div className="card-title">Gifts</div>
      <div className="gifts-grid" id="gifts-grid">
        {gifts.map(([id, g]) => {
          const misfit = g.ownerId === 'misfit';
          const lvl = Math.max(0, Math.min(5, g.naughtyLevel ?? 0));
          const cost = minCost(cfg, g.naughtyLevel);
          return (
            <div key={id} className={`gift-card ${misfit ? 'misfit' : ''} ${game.currentTurn?.giftId === id ? 'contested' : ''}`}>
              <div className="f6-gift mini" style={{ ['--gift-color' as string]: avatarColor(g.name) }}><span className="ribbon-v" /><span className="ribbon-h" /><span className="bow" /></div>
              <div className="gift-lines">
                <div className="gift-name">{g.name}</div>
                <div className="gift-owner">{misfit ? <><Icon name="teddy" /> Misfit pile</> : game.players?.[g.ownerId]?.name || 'Unknown'} · min {cost} chip{cost > 1 ? 's' : ''}</div>
              </div>
              <span className="f6-chip naughty-meter" title={`Naughty ${lvl}`}>{'●'.repeat(lvl)}{'○'.repeat(5 - lvl)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Duel records (who's good at duelling) + the betting leaderboard (who calls duels right). */
function Records() {
  const { game } = useHost();
  if (game.phase === 'lobby') return null;
  const duels = rows(game, 'duelWins', 'duelLosses').sort((a, b) => (Math.abs(b.ratio - a.ratio) > 0.1 ? b.ratio - a.ratio : b.total - a.total));
  const votes = rows(game, 'votingWins', 'votingLosses').map((r) => ({ ...r, weighted: r.ratio * (Math.log(r.total + 1) / Math.log(10)) + (r.w - r.l) * 0.1 }))
    .sort((a, b) => (Math.abs(b.weighted - a.weighted) > 0.01 ? b.weighted - a.weighted : Math.abs(b.ratio - a.ratio) > 0.1 ? b.ratio - a.ratio : b.total - a.total));
  if (!duels.length && !votes.length) return null;
  const list = (rs: ReturnType<typeof rows>, sub: (r: ReturnType<typeof rows>[0]) => string, label: string, crown = false) => (
    <div className="f6-rows">
      {rs.map((r, i) => (
        <div key={r.id} className={`f6-row${i === 0 ? ' top' : ''}`}>
          <span className="rank f6-num">{String(i + 1).padStart(2, '0')}</span>
          <Avatar name={r.name} avatar={game.players?.[r.id]?.avatar} />
          <div className="who"><div className="nm">{r.name}{crown && i === 0 && <> <Icon name="crown" /></>}</div><div className="sub">{sub(r)}</div></div>
          <div className="stat"><div className="big" style={{ color: r.ratio >= 50 ? 'var(--accent2)' : 'var(--accent)' }}>{r.ratio}%</div><div className="small">{label}</div></div>
        </div>
      ))}
    </div>
  );
  return (
    <div className="card-duo" id="card-duo">
      {duels.length > 0 && (
        <div className="card" id="card-duel-records">
          <div className="card-title">Duel records</div>
          <div className="f6-cap" style={{ marginBottom: 10 }}>Use this when voting on duel winners</div>
          <div id="player-performance-container">{list(duels, (r) => `${r.w}W, ${r.l}L · ${r.total} duel${r.total > 1 ? 's' : ''}`, 'win rate')}</div>
        </div>
      )}
      {votes.length > 0 && (
        <div className="card" id="card-betting">
          <div className="card-title">Betting leaderboard</div>
          <div id="voting-leaderboard-container">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 10 }}>
              <span className="f6-chip tint"><Icon name="medal" /> Each correct call = 1 Party Score point</span>
              <span className="f6-cap" style={{ fontSize: 9 }}>win % × participation</span>
            </div>
            {list(votes, (r) => `${r.w}–${r.l} record · ${r.total} vote${r.total > 1 ? 's' : ''}`, 'correct', true)}
          </div>
        </div>
      )}
    </div>
  );
}

function rows(game: Game, winKey: 'duelWins' | 'votingWins', lossKey: 'duelLosses' | 'votingLosses') {
  return Object.entries(game.players || {}).map(([id, p]) => {
    const w = p[winKey] || 0, l = p[lossKey] || 0, total = w + l;
    return { id, name: p.name, w, l, total, ratio: total ? parseFloat(((w / total) * 100).toFixed(1)) : 0 };
  }).filter((r) => r.total > 0);
}

export { giftless };
