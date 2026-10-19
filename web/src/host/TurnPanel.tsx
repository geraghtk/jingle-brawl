// The TV's turn panel: what's happening now, the bids/party/Draw Off area, and the host's buttons.
import qrcode from 'qrcode-generator';
import { DuelGameStage, useDuelGame } from './DuelGameStage';
import { useVoteGate } from './VoteGate';
import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import * as A from '../core/actions';
import * as D from '../core/duel';
import * as DO from '../core/drawoff';
import type { DrawOff } from '../core/drawoff';
import { GAMES } from '../core/party';
import type { PartyGame } from '../core/party';
import { advanceParty } from '../core/partyflow';
import { gambitTargets, giftless, rankedBids, reprisalTargets, topTiedBidders, wrappedGifts } from '../core/rules';
import { isMinigameMethod, minCost } from '../core/settings';
import { partyMvps, partyScore } from '../core/scoring';
import type { Game, Plan, Turn } from '../core/types';
import { usingEmulators } from '../firebase/app';
import { Avatar, avatarColor } from '../ui/avatar';
import { freshRng, useHost } from './HostContext';
import { OptionList, useModal } from './Modal';
import { PartyPanel } from './PartyPanel';

const nm = (game: Game, id?: string | null) => (id && game.players?.[id]?.name) || 'Someone';

export interface Panel {
  badge?: string; badgeClass?: string; action?: ReactNode; details?: ReactNode;
  areaTitle?: string; area?: ReactNode; buttons?: ReactNode; lobby?: boolean;
}

export function TurnPanel() {
  const host = useHost();
  const setModal = useModal();
  const p = usePanel(host, setModal);
  return (
    <div className="card turn-panel" id="turn-panel">
      {!p.lobby && <span className={`phase-badge ${p.badgeClass ?? ''}`} id="phase-badge">{p.badge}</span>}
      {!p.lobby && <div className="current-action" id="current-action">{p.action}</div>}
      {!p.lobby && <div className="action-details" id="action-details">{p.details}</div>}
      {p.lobby && <LobbyHero />}
      <div id="bids-container" className={p.area ? '' : 'hidden'}>
        <div className="card-title" id="bids-container-title">{p.areaTitle ?? 'Bids'}</div>
        <div className="bids-grid" id="bids-grid">{p.area}</div>
      </div>
      <div className="action-buttons" id="action-buttons">{p.buttons}</div>
    </div>
  );
}

// --- the lobby ------------------------------------------------------------------------------

function LobbyHero() {
  const { game, room, act, cfg } = useHost();
  const players = Object.entries(game.players || {});
  const url = `${location.origin}${location.pathname.replace('host.html', 'player.html')}?room=${room}`;
  const svg = useMemo(() => {
    const qr = qrcode(0, 'M');
    qr.addData(url);
    qr.make();
    return qr.createSvgTag({ cellSize: 4, margin: 4, scalable: true });
  }, [url]);
  return (
    <div id="qr-container" className="lobby-hero">
      <div className="lobby-hero-left">
        <span className="f6-chip tint2">● Live lobby</span>
        <div className="f6-cap lobby-kicker">Join the room</div>
        <h1 className="lobby-hero-title f6-disp fz-riso">Scan,<br />join,<br /><span className="hero-accent">brawl.</span></h1>
        <div className="lobby-hero-tag">High stakes. Low blows. Questionable gifts.</div>
        <div className="lobby-join-card f6-card f6-rise">
          <span className="fz-sticker yel lobby-sticker">{cfg.startingChips ?? 10} CHIPS EACH</span>
          <div id="qr-code" style={{ display: 'block' }} dangerouslySetInnerHTML={{ __html: svg }} />
          <div className="join-mid">
            <div className="f6-cap">Room code</div>
            <div className="room-code" id="room-code-display">{room}</div>
          </div>
          <button className="btn btn-primary lobby-start-btn" id="btn-start-game" aria-label="Start Game" disabled={players.length < 2}
            onClick={() => act(A.startGame)}>Start →</button>
        </div>
        <div className="join-url" id="join-url">{url}</div>
        <p className="lobby-start-hint" id="lobby-start-hint">{players.length < 2 ? 'Need at least 2 players to start' : 'Everyone in? Hit Start.'}</p>
        {usingEmulators && <p className="lobby-start-hint" style={{ fontSize: 12 }}>Local test mode (emulators)</p>}
      </div>
      <div className="lobby-roster f6-card f6-rise">
        <div className="lobby-roster-head">
          <span className="roster-title">Players</span>
          <span className="f6-chip tint2" id="roster-count">{players.length} joined</span>
        </div>
        <div className="lobby-roster-list" id="lobby-roster-list">
          {players.length === 0 ? <div className="roster-empty">Waiting for players to join…</div> : players.map(([id, p]) => (
            <div className="roster-row" key={id}>
              <Avatar name={p.name} size={32} />
              <div className="who"><div className="nm">{p.name}</div><div className="sub">Player</div></div>
              {cfg.publicChips && <span className="roster-chips"><span className="f6-coin" /> {p.chips}</span>}
              <span className="roster-ready">✓</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// --- bids ---------------------------------------------------------------------------------

function SealedBids({ t }: { t: Turn }) {
  const { game, cfg } = useHost();
  const sitter = t.openerId || t.activePlayerId;
  const gift = t.giftId ? game.gifts?.[t.giftId] : undefined;
  const naughty = Math.max(0, Math.min(5, gift?.naughtyLevel ?? 0));
  const eligible = Object.keys(game.players || {}).filter((id) => id !== sitter);
  const submitted = Object.keys(t.bids || {}).length;
  const pct = eligible.length ? Math.round((submitted / eligible.length) * 100) : 0;
  return (
    <div className="sealed-grid">
      <div className="sealed-gift-card">
        <div className="fz-halftone sealed-halftone" />
        <span className="f6-chip tint sealed-chip">Contested</span>
        <div className="sealed-gift-title f6-disp">{gift?.name || t.misfitName || 'Mystery gift'}</div>
        <div className="sealed-gift-wrap">
          <div className="f6-gift" style={{ ['--gift-color' as string]: avatarColor(gift?.name || 'gift') }}>
            <span className="ribbon-v" /><span className="ribbon-h" /><span className="bow" />
          </div>
          <span className="fz-sticker ink naughty-sticker">NAUGHTY {'●'.repeat(naughty) + '○'.repeat(5 - naughty)}</span>
        </div>
        <div className="sealed-gift-help">Everyone but <strong>{nm(game, sitter) || 'the opener'}</strong> bids. Highest earns the right to throw hands.</div>
      </div>
      <div className="sealed-progress">
        <div className="sealed-progress-head">
          <span className="sealed-heading f6-disp">Locking in…</span>
          <span className="sealed-count f6-disp fz-riso f6-num">{submitted}<span className="of">of {eligible.length}</span></span>
        </div>
        <div className="sealed-bar"><div className="sealed-bar-fill" style={{ width: `${pct}%` }} /></div>
        <div className="sealed-tiles">
          {Object.entries(game.players || {}).map(([id, p]) => {
            const sealed = !!t.bids?.[id];
            return (
              <div key={id} className={`sealed-tile${sealed ? ' is-sealed' : ''}`}>
                <Avatar name={p.name} size={28} />
                <div className="tile-who"><div className="tile-nm">{p.name}</div>{cfg.publicChips && <div className="tile-sub">{p.chips} chips</div>}</div>
                {id === sitter ? <span className="st opener">Opener</span> : sealed ? <span className="st sealed">✓ Sealed</span> : <span className="st bidding">…bidding</span>}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/** Bars rise one at a time; the outcome text waits for them. */
function RevealChart({ t }: { t: Turn }) {
  const { game } = useHost();
  const entries = Object.entries(t.bids || {}).sort(([, a], [, b]) => b.amount - a.amount);
  const [shown, setShown] = useState(0);
  const key = `${t.giftId}|${t.openerId}|${entries.length}`;
  useEffect(() => {
    setShown(0);
    const timers = entries.map((_, i) => setTimeout(() => setShown(i + 1), 400 + i * 600));
    return () => timers.forEach(clearTimeout);
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!entries.length) return <div style={{ gridColumn: '1 / -1', color: 'var(--text-muted)' }}>No bids received</div>;
  const top = entries[0][1].amount;
  const max = Math.max(1, top);
  return (
    <div className="reveal-chart">
      {entries.map(([pid, bid], i) => {
        const isTop = bid.amount > 0 && bid.amount === top;
        const runner = !isTop && i === 1 && bid.amount > 0;
        return (
          <div key={pid} className={`reveal-col${i < shown ? ' shown' : ''}`} style={{ ['--h' as string]: `${((bid.amount / max) * 70 + 14).toFixed(1)}%` }}>
            {isTop && i === 0 ? <span className="fz-sticker reveal-sticker">👑 CHALLENGER 1</span>
              : isTop ? <span className="fz-sticker reveal-sticker">👑 TIED</span>
              : runner ? <span className="fz-sticker yel reveal-sticker">CHALLENGER 2</span>
              : <span className="reveal-sticker-spacer" />}
            <div className={`reveal-bar ${isTop ? 'winner' : runner ? 'runner' : ''}`}><span className="reveal-amount f6-disp f6-num">{bid.amount > 0 ? bid.amount : '–'}</span></div>
            <div className="reveal-meta">
              <Avatar name={nm(game, pid)} />
              <div className="reveal-name">{nm(game, pid)}</div>
              <div className="reveal-sub f6-cap"><span className="when-sealed">sealed</span><span className="when-shown">{bid.amount > 0 ? `${bid.amount} chip${bid.amount > 1 ? 's' : ''}` : 'pass'}</span></div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

const Waiting = ({ who, what, override }: { who: string; what: string; override: ReactNode }) => (
  <>
    <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '1rem', background: 'var(--bg-card-hover)', borderRadius: 8, marginBottom: '0.75rem' }}>
      <div style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>📱 Waiting for {who === 'Opener' ? 'Opener' : 'Player'}...</div>
      <div style={{ fontSize: '0.85rem' }}>{what}</div>
    </div>
    <details style={{ marginTop: '0.5rem' }}>
      <summary style={{ cursor: 'pointer', color: 'var(--text-muted)', fontSize: '0.85rem' }}>🔧 Manual Override (if {who === 'Opener' ? 'opener' : 'player'} can't use phone)</summary>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.75rem' }}>{override}</div>
    </details>
  </>
);

const OVERHEARD = ['“Oh no. A scented candle. Again.”', '“I can hear it ticking. That\'s normal, right?”',
  '“Whoever wrapped this used an entire roll of tape.”', '“It\'s either socks or a lawsuit.”', '“Re-gifted? That\'s MY handwriting on the tag.”'];

// --- which panel for which moment -------------------------------------------------------------

function usePanel(host: ReturnType<typeof useHost>, setModal: ReturnType<typeof useModal>): Panel {
  const { game, act, cfg } = host;
  const t = game.currentTurn;
  const close = () => setModal(null);

  if (!t) {
    if (game.phase === 'lobby') return { lobby: true };
    if (game.phase === 'main') {
      const wrapped = wrappedGifts(game);
      const bag = giftless(game).length;
      if (wrapped > 0 && bag > 0) {
        return {
          badge: 'READY', badgeClass: 'bidding', action: '🎅 Ready to draw!',
          details: `${bag} player${bag > 1 ? 's' : ''} in the bag, ${wrapped} gift${wrapped > 1 ? 's' : ''} to open.`,
          areaTitle: 'Next up',
          area: (
            <div className="unwrap-tease">
              <div className="sealed-gift-wrap">
                <div className="f6-gift" style={{ ['--gift-color' as string]: 'var(--accent2)' }}><span className="ribbon-v" /><span className="ribbon-h" /><span className="bow" /></div>
                <span className="fz-sticker unwrap-sticker f6-wiggle">UNWRAP ME</span>
              </div>
              <div className="overheard-bar"><span className="fz-sticker yel">OVERHEARD</span><em>{OVERHEARD[(wrapped + bag) % OVERHEARD.length]}</em></div>
            </div>
          ),
        };
      }
      if (wrapped === 0 && bag > 0) return { badge: 'READY', badgeClass: 'bidding', action: '🎰 All gifts opened!', details: `${bag} player${bag > 1 ? 's' : ''} still need gifts. Start the Endgame!` };
      return { badge: 'READY', badgeClass: 'bidding', action: '✅ Everyone has a gift!', details: 'The game is complete, or you can continue with more rounds.' };
    }
    if (game.phase === 'endgame') return { badge: 'ENDGAME', badgeClass: 'duel', action: '🎰 Misfit Lottery', details: 'Giftless players compete for misfit gifts!' };
    return finalTally(game);
  }

  switch (t.phase) {
    case 'bidding':
      return {
        badge: '🎯 SEALED BIDS', badgeClass: 'bidding',
        action: <><span className="f6-chip tint" style={{ verticalAlign: 'middle', marginRight: 10 }}>Drawn from the sack</span>{nm(game, t.activePlayerId)} opened a gift!</>,
        area: <SealedBids t={t} />,
        buttons: <>
          <button className="btn btn-gold" data-testid="reveal-bids" onClick={() => act(A.revealBids)}>👁️ Reveal All Bids</button>
          <button className="btn btn-secondary" onClick={() => act(A.skipBidding)}>⏭️ No Bids (Skip)</button>
        </>,
      };

    case 'reveal': {
      const base: Panel = {
        badge: '✉ REVEAL', badgeClass: 'reveal',
        action: <><span className="f6-chip tint" style={{ verticalAlign: 'middle', marginRight: 10 }}>✉ Envelope, please</span>The bids, <span style={{ color: 'var(--accent)' }}>revealed.</span></>,
        area: <RevealChart t={t} />,
      };
      const bids = rankedBids(t);
      const opener = nm(game, t.openerId);
      if (!bids.length) {
        return {
          ...base,
          details: <>No bids!<br /><span style={{ color: 'var(--accent-green)' }}>⏳ Waiting for {opener} to choose: Keep or Gambit?</span></>,
          buttons: <Waiting who="Opener" what={`${opener} is choosing on their device`} override={<>
            <button className="btn btn-primary" onClick={() => act(A.keepGift)}>✅ Opener Keeps Gift</button>
            <button className="btn btn-gold" onClick={() => {
              const targets = gambitTargets(game, t.giftId, t.openerId!);
              if (!targets.length) return alert('No other gifts to challenge!');
              setModal(<OptionList title="😈 Grinch's Gambit" intro={`${opener} can challenge anyone's gift!`} onCancel={close}
                options={targets.map(([id, g]) => {
                  const cost = minCost(cfg, g.naughtyLevel);
                  return { key: id, title: g.name, desc: `Owner: ${nm(game, g.ownerId)} • Cost: ${cost} chip${cost > 1 ? 's' : ''}`, onPick: () => { close(); act((x) => A.gambit(x, id)); } };
                })} />);
            }}>😈 Grinch's Gambit</button>
          </>} />,
        };
      }
      const tied = topTiedBidders(t);
      const [[c1, top], second] = bids;
      if (tied.length >= 2) {
        const mini = isMinigameMethod(cfg.tiebreakerMethod);
        const hidden = mini ? `${tied.length === 2 ? '2' : tied.length} players (hidden until tiebreaker concludes)` : null;
        return {
          ...base,
          details: <>
            <strong style={{ color: 'var(--accent-gold)' }}>{tied.length === 2 ? 'TIE!' : `${tied.length}-WAY TIE`}</strong>{' '}
            {tied.length === 2 ? <>{hidden ?? `${nm(game, tied[0])} & ${nm(game, tied[1])}`} both bid {top}.</> : <>at {top} chips: {hidden ?? tied.map((id) => nm(game, id)).join(', ')}</>}<br />
            <span style={{ color: 'var(--accent-green)' }}>⏳ Waiting for {opener} to choose: Fight or Yield?</span>
          </>,
          buttons: <Waiting who="Opener" what={`${opener} is choosing on their device`} override={<>
            <button className="btn btn-primary" onClick={() => act((g) => A.hostTieFight(g, { now: Date.now(), rng: freshRng() }))}>⚔️ Opener Fights → {mini ? 'Mini-Game to Pick 1' : tied.length === 2 ? 'Tiebreaker Duel' : 'Random Tiebreaker'}</button>
            {tied.length === 2
              ? <button className="btn btn-gold" onClick={() => act((g) => A.startYieldDuel(g, tied[0], tied[1], false, { now: Date.now(), rng: freshRng() }))}>🏃 Opener Yields → {mini ? '2 players fight' : `${nm(game, tied[0])} vs ${nm(game, tied[1])}`}</button>
              : <button className="btn btn-gold" onClick={() => act((g) => A.hostTieYield(g, { now: Date.now(), rng: freshRng() }))}>🏃 Opener Yields → {mini ? 'Mini-Game to Pick 2' : 'Random Selection'}</button>}
          </>} />,
        };
      }
      if (second) {
        return {
          ...base,
          details: <>Highest bid: <strong>{nm(game, c1)}</strong> ({top} chips). Second: {nm(game, second[0])}<br /><span style={{ color: 'var(--accent-green)' }}>⏳ Waiting for {opener} to choose: Fight or Yield?</span></>,
          buttons: <Waiting who="Opener" what={`${opener} is choosing on their device`} override={<>
            <button className="btn btn-primary" onClick={() => act((g) => A.startDuel(g, c1, t.openerId!))}>⚔️ Duel: {nm(game, c1)} vs Opener</button>
            <button className="btn btn-secondary" onClick={() => act((g) => A.startYieldDuel(g, c1, second[0], false, { now: Date.now(), rng: freshRng() }))}>🏃 Opener Yields</button>
          </>} />,
        };
      }
      return {
        ...base, details: `Highest bid: ${nm(game, c1)} (${top} chips)`,
        buttons: <button className="btn btn-primary" data-testid="start-duel" onClick={() => act((g) => A.startDuel(g, c1, t.openerId!))}>⚔️ Start Duel</button>,
      };
    }

    case 'yield_choice': {
      const a = nm(game, t.selectedC1), b = nm(game, t.selectedC2);
      return {
        badge: '🎲 SELECTED', badgeClass: 'reveal', action: `${a} & ${b} Selected!`,
        details: <>Both bid {t.bids?.[t.selectedC1!]?.amount || 0} chips.<br /><span style={{ color: 'var(--accent-green)' }}>⏳ Waiting for {nm(game, t.openerId)} to choose: Fight or Yield?</span></>,
        buttons: <Waiting who="Opener" what={`${nm(game, t.openerId)} is choosing on their device`} override={<>
          <button className="btn btn-primary" onClick={() => act((g) => A.startTiebreaker(g, t.selectedC1!, t.selectedC2!, { now: Date.now(), rng: freshRng() }))}>⚔️ Tiebreaker → Winner vs Opener</button>
          <button className="btn btn-gold" onClick={() => act((g) => A.startYieldDuel(g, t.selectedC1!, t.selectedC2!, true, { now: Date.now(), rng: freshRng() }))}>🏃 Opener Yields → {a} vs {b}</button>
        </>} />,
      };
    }

    case 'duel_pending':
      return {
        badge: '⚔️ DUEL', badgeClass: 'duel',
        action: <span className="f6-chip tint" style={{ verticalAlign: 'middle', marginRight: 10 }}>Opener chose to fight</span>,
        details: <div className="duel-pending-row">
          <span className="dp-side f6-disp" style={{ color: 'var(--accent)' }}>{nm(game, t.challengerId).toUpperCase()}</span>
          <span className="dp-vs f6-disp fz-riso f6-wiggle">VS</span>
          <span className="dp-side f6-disp" style={{ color: 'var(--accent2)' }}>{nm(game, t.openerId).toUpperCase()}</span>
        </div>,
        buttons: <button className="btn btn-primary" data-testid="start-duel" onClick={() => act((g) => A.startDuel(g, t.challengerId!, t.openerId!))}>⚔️ Start Duel!</button>,
      };

    case 'yield_pending':
      return {
        badge: '🏃 YIELD', badgeClass: 'reveal', action: `${nm(game, t.openerId)} chose to YIELD!`,
        details: <>{nm(game, t.yieldC1)} vs {nm(game, t.yieldC2)} will fight for the gift directly.<br /><span style={{ color: 'var(--text-muted)' }}>Click to start the yield duel (both pay half bids).</span></>,
        buttons: <button className="btn btn-gold" data-testid="start-yield-duel" onClick={() => act((g) => A.startYieldDuel(g, t.yieldC1!, t.yieldC2!, false, { now: Date.now(), rng: freshRng() }))}>🏃 Start Yield Duel!</button>,
      };

    case 'tiebreaker_pending': {
      const method = cfg.tiebreakerMethod;
      if (isMinigameMethod(method)) {
        // Mini-game tiebreakers start themselves (useHostAutomation)
        const label = method === 'draw_off' ? 'AI Draw Off' : GAMES[method as keyof typeof GAMES].label;
        return { badge: '⚖️ TIEBREAKER', badgeClass: 'duel', action: `🎨 Starting ${label}...`, details: 'Everyone plays — the best-placed tied bidder wins the tie.' };
      }
      return {
        badge: '⚖️ TIEBREAKER', badgeClass: 'duel', action: 'Opener chose to FIGHT!',
        details: <>{nm(game, t.tiebreakerC1)} vs {nm(game, t.tiebreakerC2)} for the right to challenge the opener.<br /><span style={{ color: 'var(--text-muted)' }}>Click to start the tiebreaker duel.</span></>,
        buttons: <button className="btn btn-gold" data-testid="start-tiebreaker" onClick={() => act((g) => A.startTiebreaker(g, t.tiebreakerC1!, t.tiebreakerC2!, { now: Date.now(), rng: freshRng() }))}>⚔️ Start Tiebreaker Duel!</button>,
      };
    }

    case 'duel': {
      const gift = t.giftId ? game.gifts?.[t.giftId] : undefined;
      const record = t.isTiebreak ? A.recordTiebreakWinner : (g: Game, id: string) => D.resolveDuel(g, id, Date.now());
      return {
        badge: t.isTiebreak ? '⚖️ TIEBREAK' : '⚔️ DUEL', badgeClass: 'duel',
        action: `${nm(game, t.challengerId)} vs ${nm(game, t.defenderId)}`,
        details: <>
          {t.isTiebreak ? <span style={{ color: 'var(--accent-gold)' }}>Winner becomes the Challenger!</span> : <>Fighting for: "<strong>{gift?.name}</strong>"</>}
          {t.isTiebreak && <DuelGameStage compact />}
        </>,
        buttons: <DuelWinnerButtons t={t} record={record} />,
      };
    }

    case 'reprisal': {
      const who = nm(game, t.reprisalPlayerId);
      return {
        badge: '🦌 REPRISAL', badgeClass: 'duel', action: `${who} has a Reindeer Reprisal!`,
        details: <span style={{ color: 'var(--accent-green)' }}>⏳ Waiting for {who} to choose on their device...</span>,
        buttons: <Waiting who="Player" what={`${who} is choosing a target on their device`} override={<>
          <button className="btn btn-gold" onClick={() => setModal(<OptionList title="🦌 Choose Reprisal Target" intro="Select a gift to challenge (costs Min Cost to Bank)" onCancel={close}
            options={reprisalTargets(game, t.excludedGiftId, t.reprisalPlayerId!).map(([id, g]) => {
              const cost = minCost(cfg, g.naughtyLevel);
              return { key: id, title: g.name, desc: `Owner: ${g.ownerId === 'misfit' ? 'Misfit Pile' : nm(game, g.ownerId)} • Cost: ${cost} chip${cost > 1 ? 's' : ''}`, onPick: () => { close(); act((x) => D.chooseReprisal(x, id, Date.now())); } };
            })} />)}>🎯 Choose Target for Them</button>
          <button className="btn btn-secondary" onClick={() => act(D.skipReprisal)}>⏭️ Skip Reprisal</button>
        </>} />,
      };
    }

    case 'endgame_choice': {
      const who = nm(game, t.activePlayerId);
      return {
        badge: '🎰 ENDGAME', badgeClass: 'duel', action: `${who}'s Misfit Lottery Turn`,
        details: <>Target: "{t.misfitName}"<br /><span style={{ color: 'var(--accent-green)' }}>⏳ Waiting for {who} to choose path...</span></>,
        buttons: <Waiting who="Player" what={`${who} is choosing on their device`} override={<>
          <button className="btn btn-primary" onClick={() => stealModal()}>🎯 Path A: Steal a Gift</button>
          <button className="btn btn-gold" onClick={() => act(D.startAuction)}>🎰 Path B: Auction for Misfit</button>
        </>} />,
      };
    }
    case 'endgame_steal_select': {
      const who = nm(game, t.activePlayerId);
      return {
        badge: '🎯 STEAL', badgeClass: 'duel', action: `${who} is Stealing!`,
        details: <span style={{ color: 'var(--accent-green)' }}>⏳ Waiting for {who} to choose target...</span>,
        buttons: <Waiting who="Player" what={`${who} is selecting a target on their device`} override={<button className="btn btn-gold" onClick={() => stealModal()}>🎯 Choose Target for Them</button>} />,
      };
    }
    case 'endgame_nobids': {
      const who = nm(game, t.activePlayerId);
      return {
        badge: '🎰 NO BIDS', badgeClass: 'reveal', action: `No Bids! ${who} Chooses Defender`,
        details: <span style={{ color: 'var(--accent-green)' }}>⏳ Waiting for {who} to choose...</span>,
        buttons: <Waiting who="Player" what={`${who} is choosing a defender on their device`} override={<button className="btn btn-gold" onClick={() => defenderModal()}>👤 Choose Defender for Them</button>} />,
      };
    }
    case 'endgame_bidding':
      return {
        badge: '🎰 AUCTION', badgeClass: 'bidding', action: `Auction for "${t.misfitName}"`,
        details: 'Everyone can bid! Active player duels highest bidder.', area: <SealedBids t={t} />,
        buttons: <>
          <button className="btn btn-gold" data-testid="reveal-auction" onClick={() => act(D.revealAuction)}>👁️ Reveal Bids</button>
          <button className="btn btn-secondary" onClick={() => act(D.skipAuction)}>⏭️ No Bids</button>
        </>,
      };
    case 'endgame_reveal': {
      const bids = rankedBids(t);
      return {
        badge: '👁️ REVEAL', badgeClass: 'reveal', action: 'Auction Bids Revealed!', area: <RevealChart t={t} />,
        details: bids.length ? `Highest: ${nm(game, bids[0][0])} (${bids[0][1]} chips)` : 'No bids! Active player chooses a defender.',
        buttons: bids.length
          ? <button className="btn btn-primary" data-testid="auction-duel" onClick={() => act((g) => D.auctionDuel(g, bids[0][0]))}>⚔️ Duel Highest Bidder</button>
          : <button className="btn btn-primary" onClick={() => defenderModal()}>🎯 Choose Defender</button>,
      };
    }

    case 'minigame_party': {
      const mg = game.minigame as PartyGame | undefined;
      const g = (mg && GAMES[mg.type]) || { emoji: '🎉', label: 'Party game' };
      return {
        badge: `${g.emoji} ${g.label.toUpperCase()}`, badgeClass: 'reveal', action: `${g.emoji} ${g.label} — everyone plays!`,
        details: 'Tied bidders play for the tie; everyone else for party points.',
        areaTitle: 'Party game', area: mg ? <PartyPanel mg={mg} /> : null,
        buttons: mg && <button className="btn btn-secondary" data-testid="skip-party-phase" onClick={() => act((x) => advanceParty(x, mg.phase, Date.now()), { quiet: true })}>
          {mg.phase === 'results' ? '⚔️ Continue to the duel' : '⏭ Skip ahead'}
        </button>,
      };
    }
    case 'minigame_draw_off': return drawOffPanel(game, act);
    default: return { badge: t.phase, action: 'Waiting…' };
  }

  function stealModal() {
    const targets = Object.entries(game.gifts || {}).filter(([, g]) => g.ownerId !== 'misfit' && g.ownerId !== t!.activePlayerId);
    if (!targets.length) return alert('No player gifts to steal! Use Path B.');
    setModal(<OptionList title="🎯 Path A: Steal a Gift" intro={`${nm(game, t!.activePlayerId)} challenges a player. Loser takes the misfit.`} onCancel={close}
      options={targets.map(([id, g]) => {
        const cost = minCost(cfg, g.naughtyLevel);
        return { key: id, title: g.name, desc: `Owner: ${nm(game, g.ownerId)} • Cost: ${cost} chip${cost > 1 ? 's' : ''}`, onPick: () => { close(); act((x) => D.executeSteal(x, id)); } };
      })} />);
  }
  function defenderModal() {
    setModal(<OptionList title="🎯 Choose Your Defender" intro={`${nm(game, t!.activePlayerId)} picks someone to duel. Win = free misfit. Lose = pay 1 chip toll.`} onCancel={close}
      options={Object.entries(game.players || {}).filter(([id]) => id !== t!.activePlayerId)
        .map(([id, p]) => ({ key: id, title: p.name, onPick: () => { close(); act((x) => D.noBidsDuel(x, id)); } }))} />);
  }
}

function finalTally(game: Game): Panel {
  const ranked = Object.entries(game.players || {})
    .map(([id, p]) => ({ id, name: p.name, chips: p.chips || 0, score: partyScore(p), gift: p.giftId ? game.gifts?.[p.giftId]?.name || 'Mystery gift' : 'No gift' }))
    .sort((a, b) => b.chips - a.chips);
  const mvps = partyMvps(game.players);
  return {
    badge: 'FINAL TALLY', badgeClass: '',
    action: <><span className="fz-sticker" style={{ marginRight: 12, verticalAlign: 'middle' }}>FINAL VERDICT</span> Who got <span style={{ color: 'var(--accent)' }}>what.</span></>,
    details: 'Thanks for playing Yuel Duel!', areaTitle: 'Standings',
    area: ranked.length ? <>
      {mvps.length > 0 && (
        <div className="party-mvp" style={{ gridColumn: '1 / -1', textAlign: 'center', fontWeight: 800, fontSize: '1.15rem', marginBottom: 8 }}>
          🏅 Party MVP: {mvps.map((id) => game.players![id].name).join(' & ')}
          <span className="f6-cap" style={{ display: 'block', marginTop: 2 }}>Party Score = party-game points + correct duel calls</span>
        </div>
      )}
      <div className="f6-rows tally-grid" style={{ gridColumn: '1 / -1' }}>
        {ranked.map((p, i) => (
          <div key={p.id} className={`f6-row${i === 0 ? ' top' : ''}`}>
            <span className="rank f6-num">{String(i + 1).padStart(2, '0')}</span>
            <Avatar name={p.name} size={38} />
            <div className="who"><div className="nm">{p.name}{i === 0 ? ' 🏆' : ''}</div><div className="sub">{p.gift}</div></div>
            <div className="stat">
              <div className="big"><span className="f6-coin" /> {p.chips}</div>
              {p.score > 0 && <div className="sub">🏅 {p.score} party score</div>}
            </div>
          </div>
        ))}
      </div>
    </> : null,
  };
}

function drawOffPanel(game: Game, act: ReturnType<typeof useHost>['act']): Panel {
  const mg = game.minigame as DrawOff | undefined;
  if (!mg || mg.type !== 'draw_off') return { badge: '🎨 DRAW OFF', badgeClass: 'reveal', action: 'AI Draw Off', details: 'Loading...' };
  const players = game.players || {};
  const card = (pid: string, children: ReactNode, winner = false) => (
    <div key={pid} className={`drawoff-card${winner ? ' winner' : ''}`}>
      {winner && <span className="fz-sticker mint drawoff-winner-sticker">🏆 WINNER</span>}
      <div className="drawoff-who"><Avatar name={players[pid]?.name || ''} size={24} /><span>{players[pid]?.name || 'Unknown'}</span></div>
      {children}
    </div>
  );
  switch (mg.phase) {
    case 'prompt_reveal':
      return { badge: '📋 PROMPT', badgeClass: 'reveal', action: <><span className="f6-chip critic-chip">🤖 AI critic</span> Draw-off!</>,
        details: <><div className="drawoff-prompt f6-disp">“{mg.prompt}”</div><div style={{ color: 'var(--muted)' }}>Players see the prompt. Drawing starts in 10 seconds…</div></> };
    case 'drawing': {
      const subs = mg.submissions || {};
      return {
        badge: '✏️ DRAWING', badgeClass: 'reveal', action: <><span className="f6-chip critic-chip">🤖 AI critic</span> <DrawOffTimer endsAt={mg.drawingEndTime} /></>,
        details: <>
          <div className="drawoff-prompt f6-disp">“{mg.prompt}”</div>
          <div style={{ color: 'var(--muted)', marginBottom: '1rem' }}>{mg.participants.length} players drawing on their devices · {Object.keys(subs).length} / {mg.participants.length} submitted</div>
          <div className="drawoff-grid">{mg.participants.map((pid) => card(pid, <div className="drawoff-well">{subs[pid] ? <img src={subs[pid]} alt={players[pid]?.name} /> : <span className="drawoff-waiting">Drawing…</span>}</div>))}</div>
        </>,
      };
    }
    case 'judging':
      return { badge: '🤖 JUDGING', badgeClass: 'reveal', action: 'AI Art Critic is judging...',
        details: <><div style={{ fontSize: '1.1rem', color: 'var(--accent-gold)' }}>Draw: "{mg.prompt}"</div><div style={{ color: 'var(--text-secondary)', marginTop: '1rem' }}>AI is scoring each drawing. Please wait...</div></> };
    case 'results': {
      const winners = new Set(mg.winners || []);
      return {
        badge: '🏆 RESULTS', badgeClass: 'reveal', action: 'Review drawings & critiques, then continue to duel',
        details: <>
          <div className="drawoff-prompt f6-disp">“{mg.prompt}”</div>
          {mg.wittyVerdict && <div style={{ background: 'linear-gradient(135deg, rgba(255, 197, 61,0.12) 0%, var(--bg-card-hover) 100%)', borderLeft: '4px solid var(--accent-gold)', padding: '1rem', borderRadius: 8, marginBottom: '1rem', fontStyle: 'italic' }}>🎭 <strong>Art Critic's Verdict:</strong> {mg.wittyVerdict}</div>}
          {!!mg.winners?.length && <div style={{ fontSize: '0.9rem', marginBottom: '1rem', padding: '0.75rem', background: 'rgba(24, 184, 138, 0.15)', borderRadius: 8, borderLeft: '4px solid var(--accent-green)' }}>
            🏆 <strong>{mg.winners.length === 1 ? 'Winner (Challenger)' : 'Winners (will duel)'}:</strong> {mg.winners.map((id) => players[id]?.name).join(' & ')} — selected by {mg.winnerSource}
          </div>}
          <div className="drawoff-grid">{(mg.rankings || []).map((pid) => {
            const s = mg.scores?.[pid];
            return card(pid, <>
              <div className="drawoff-well">{mg.submissions?.[pid] ? <img src={mg.submissions[pid]} alt={players[pid]?.name} /> : <span className="drawoff-waiting">No submission</span>}</div>
              <div className="drawoff-score f6-disp">Score {s?.rating || 0}<span className="of">/100</span></div>
              {s?.critique && <div className="drawoff-critique">“{s.critique}”</div>}
            </>, winners.has(pid));
          })}</div>
        </>,
        buttons: <button className="btn btn-gold" onClick={() => act(DO.completeDrawOff)} style={{ fontSize: '1.1rem', padding: '1rem 2rem' }}>⚔️ Continue to Duel</button>,
      };
    }
  }
}

function DrawOffTimer({ endsAt }: { endsAt?: number }) {
  const [, tick] = useState(0);
  useEffect(() => { const id = setInterval(() => tick((n) => n + 1), 500); return () => clearInterval(id); }, []);
  return <span className="drawoff-timer f6-disp f6-num">{Math.max(0, Math.ceil(((endsAt || 0) - Date.now()) / 1000))}s</span>;
}

/** "X Wins" in the turn panel (tiebreaks have no modal). A component, so the wheel's and the
 *  vote's timers re-render it — the lock lifts the moment the wheel stops / the votes are in. */
function DuelWinnerButtons({ t, record }: { t: Turn; record: (g: Game, id: string) => Plan }) {
  const { game, act, takeSnapshot } = useHost();
  const { ready, choice } = useDuelGame();
  const { locked: votesPending } = useVoteGate();
  const locked = !ready || (!t.isTiebreak && votesPending) || choice?.kind === 'app';
  return (
    <>
      <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
        {!ready ? '🎡 First: which duel?' : locked ? '🗳️ Waiting for the table to vote' : '👆 Click the winner to continue'}
      </div>
      <div style={{ display: 'flex', gap: '1rem' }}>
        {[t.challengerId!, t.defenderId!].map((id) => (
          <button key={id} className="btn btn-duel" disabled={locked} data-testid={`${t.isTiebreak ? 'tiebreak' : 'duel'}-winner-${id}`}
            onClick={() => { if (!t.isTiebreak) takeSnapshot(); act((g) => record(g, id)); }}>🏆 {nm(game, id)} Wins</button>
        ))}
      </div>
    </>
  );
}
