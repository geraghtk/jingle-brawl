// The TV's main game, a scene per moment (host/TvStage.tsx picks one): the lobby, between turns, sealed
// bids, the reveal, who faces whom, a player deciding, AI Draw Off. Each fills the picture's middle;
// nothing here has a control, and everything comes from the public view the TV is sent.
import qrcode from 'qrcode-generator';
import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type { DrawOff } from '../core/drawoff';
import { misfitPile, rankedBids, topTiedBidders, wrappedGifts, giftless } from '../core/rules';
import { partyStandings } from '../core/scoring';
import { minCost } from '../core/settings';
import type { Gift, PlayerId, Turn } from '../core/types';
import { useJoinOrigin } from '../net/lan';
import { useServerNow } from '../net/hooks';
import { secsLeft } from '../ui/clock';
import { Icon } from '../ui/icons';
import { StakesLines } from '../ui/Stakes';
import { bestGrid } from '../ui/tvLayout';
import { DuelGameStage, useDuelGame } from './DuelGameStage';
import { useHost } from './HostContext';
import { Av, chipWord, Fit, GiftBox, GiftTiles, Naughty, nameOf, u, vars, Wait } from './TvParts';
import { joinPath, OVERHEARD, roomSeed } from './TurnPanel';

// --- the lobby ------------------------------------------------------------------------------------

export function LobbyScene() {
  const { game, room, cfg } = useHost();
  const joinOrigin = useJoinOrigin();
  const players = Object.entries(game.players || {});
  const real = players.filter(([, p]) => !p.isBot).length;
  const url = `${joinOrigin()}${joinPath(location.pathname)}?room=${room}`;
  const svg = useMemo(() => {
    const qr = qrcode(0, 'M');
    qr.addData(url);
    qr.make();
    return qr.createSvgTag({ cellSize: 4, margin: 2, scalable: true });   // the library's own drawing of the address: no user text in it
  }, [url]);
  const address = url.replace(/^https?:\/\//, '').replace(/\?.*$/, '');
  const grid = bestGrid(players.length, 71, 66, { ratio: 1.08, gap: 1.6, maxTile: 21 });
  return (
    <div className="tvs-lobby" id="qr-container">
      <section className="tvs-join">
        <h1 className="tvs-join-title f6-disp fz-riso mo-slam">Scan, join, <span>duel.</span></h1>
        <div className="tvs-join-card mo-up">
          <span className="fz-sticker yel tvs-sticker tvs-join-sticker">{cfg.startingChips ?? 10} CHIPS EACH</span>
          <div id="qr-code" className="tvs-qr" dangerouslySetInnerHTML={{ __html: svg }} />
          <div className="tvs-join-how">
            <div className="tvs-k">Room code</div>
            <div className="tvs-code f6-disp" id="room-code-display">{room}</div>
            <div className="tvs-k">Point your camera at the code, or open</div>
            <Fit text={address} width={38} max={3.4} min={1.5} lines={2} className="tvs-url" />
            <span id="join-url" hidden>{url}</span>
          </div>
        </div>
        <p className="tvs-lobby-hint" id="lobby-start-hint">{real < 2 ? 'Need at least 2 players to start' : 'Everyone in? The host starts the game.'}</p>
      </section>
      <section className="tvs-roster">
        <header className="tvs-roster-head">
          <span className="tvs-roster-title f6-disp">Players</span>
          <span className="tvs-pill mint" id="roster-count">{players.length} joined</span>
        </header>
        <div className="tvs-roster-grid" id="lobby-roster-list" style={vars({ '--cols': grid.cols, '--tile': u(grid.tile) })}>
          {players.length === 0 ? <div className="tvs-roster-empty">Waiting for the first phone…</div> : players.map(([id, p]) => (
            <div className="roster-row tvs-roster-tile mo-pop" key={id} data-player-id={id}>
              <Av id={id} size={grid.tile * 0.58} />
              <Fit text={p.name} width={grid.tile - 1} max={Math.min(3.4, grid.tile * 0.17)} min={1.3} lines={2} className="tvs-roster-name" />
              {p.isBot && <div className="tvs-roster-sub">Practice bot</div>}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

// --- between turns ----------------------------------------------------------------------------------

function ScoreBoard() {
  const { game } = useHost();
  const table = partyStandings(game.players);
  if (!table.length || table[0].score <= 0) return null;
  return (
    <aside className="tvs-scores mo-left" id="card-party-score">
      <div className="tvs-scores-title f6-disp">Party Score</div>
      {table.slice(0, 5).map((r, i) => (
        <div key={r.id} className={`tvs-score-row${r.place === 1 ? ' top' : ''} mo-up`} style={vars({ '--i': i + 1 })} data-player-id={r.id}>
          <span className="tvs-score-rank f6-num">{r.place}</span>
          <Av id={r.id} size={5.4} />
          <Fit text={r.name} width={21} max={2.9} min={1.5} lines={2} className="tvs-score-name" />
          {r.place === 1 && <Icon name="crown" />}
          <span className="tvs-score-n f6-disp f6-num">{r.score}</span>
        </div>
      ))}
      <div className="tvs-scores-cap">Party games, correct duel calls and duels fought</div>
    </aside>
  );
}

export function BetweenScene() {
  const { game, room } = useHost();
  const wrapped = wrappedGifts(game);
  const bag = giftless(game).length;
  const draw = wrapped > 0 && bag > 0;
  const [title, sub] = draw ? ['Ready to draw!', `${bag} player${bag > 1 ? 's' : ''} in the bag, ${wrapped} gift${wrapped > 1 ? 's' : ''} to open.`]
    : bag > 0 ? ['All gifts opened!', `${bag} player${bag > 1 ? 's' : ''} still need a gift. The Misfit Lottery is next.`]
    : ['Everyone has a gift!', 'That is every gift opened and every player holding one.'];
  return (
    <div className="tvs-between">
      <div className="tvs-next">
        {draw && (
          <div className="tvs-next-gift mo-pop">
            <GiftBox size={30} color="var(--accent2)" />
            <span className="fz-sticker tvs-sticker big f6-wiggle tvs-unwrap">UNWRAP ME</span>
          </div>
        )}
        <div className="tvs-next-txt">
          <span className="tvs-pill tint mo-down">{draw ? 'Next up' : 'Ready'}</span>
          <h1 className="tvs-h1 f6-disp mo-up">{title}</h1>
          <p className="tvs-sub mo-up" style={vars({ '--i': 1 })}>{sub}</p>
          {draw && <div className="tvs-overheard mo-up" style={vars({ '--i': 2 })}><span className="fz-sticker yel tvs-sticker">OVERHEARD</span><em>{OVERHEARD[(roomSeed(room) + wrapped + bag) % OVERHEARD.length]}</em></div>}
        </div>
      </div>
      <ScoreBoard />
    </div>
  );
}

export function EndgameIdleScene() {
  const { game } = useHost();
  const pile = misfitPile(game);
  const waiting = giftless(game);
  return (
    <div className="tvs-spot">
      <div className="tvs-spot-txt wide">
        <span className="tvs-pill tint mo-down">Endgame</span>
        <h1 className="tvs-h1 f6-disp mo-up">The Misfit Lottery</h1>
        <p className="tvs-sub mo-up" style={vars({ '--i': 1 })}>
          {waiting.length ? `${waiting.length} player${waiting.length > 1 ? 's' : ''} without a gift play for the Misfit pile.` : 'The Misfit pile is up for grabs.'}
        </p>
        {waiting.length > 0 && <div className="tvs-faces">{waiting.slice(0, 10).map((id, i) => <span key={id} className="mo-pop" style={vars({ '--i': i })}><Av id={id} size={9} /></span>)}</div>}
        <GiftTiles gifts={pile} />
      </div>
    </div>
  );
}

export function EndedScene() {
  return (
    <div className="tvs-spot">
      <div className="tvs-spot-txt wide center">
        <span className="fz-sticker tvs-sticker big">FINAL VERDICT</span>
        <h1 className="tvs-h1 f6-disp mo-slam">That's the night!</h1>
        <p className="tvs-sub">Thanks for playing YuleDuel.</p>
      </div>
    </div>
  );
}

// --- the gift on the table ----------------------------------------------------------------------------

/** The gift this turn is about (or the Misfit up for auction), and whose it is. */
function GiftCard({ t, small = false, help }: { t: Turn; small?: boolean; help?: ReactNode }) {
  const { game, cfg } = useHost();
  const gift: Gift | undefined = t.giftId ? game.gifts?.[t.giftId] : t.misfitId ? game.gifts?.[t.misfitId] : undefined;
  const name = gift?.name || t.misfitName || 'Mystery gift';
  const sitter = t.openerId || t.activePlayerId;
  const cost = minCost(cfg, gift?.naughtyLevel ?? 0);
  const w = small ? 40 : 54;
  return (
    <section className={`tvs-giftcard${small ? ' small' : ''} mo-right`}>
      <div className="fz-halftone tvs-halftone" />
      <span className="fz-sticker tvs-sticker tvs-giftcard-sticker">{t.misfitName && !t.giftId ? 'THE MISFIT' : 'CONTESTED'}</span>
      <div className="tvs-giftcard-box"><GiftBox name={name} size={small ? 17 : 24} /></div>
      <Fit key={name} text={name} width={w} max={small ? 5 : 6.4} min={2.4} lines={3} className="tvs-giftcard-name f6-disp mo-slam" testid="tv-gift-name" />
      <div className="tvs-giftcard-meta"><Naughty level={gift?.naughtyLevel} /><span className="tvs-min">min bid {cost}</span></div>
      {sitter && (
        <div className="tvs-giftcard-by">
          <Av id={sitter} size={small ? 5 : 6} />
          <Fit text={`${t.giftId ? 'Opened by' : 'Drawn by'} ${nameOf(game, sitter)}`} width={w - 9} max={small ? 2.6 : 3} min={1.6} lines={2} />
        </div>
      )}
      {help && <p className="tvs-giftcard-help">{help}</p>}
    </section>
  );
}

// --- sealed bids ------------------------------------------------------------------------------------------

export function BiddingScene({ t }: { t: Turn }) {
  const { game } = useHost();
  const sitter = t.openerId || t.activePlayerId;
  const bidders = Object.keys(game.players || {}).filter((id) => id !== sitter);
  const sealed = bidders.filter((id) => !!t.bids?.[id]);
  const pct = bidders.length ? Math.round((sealed.length / bidders.length) * 100) : 0;
  const grid = bestGrid(bidders.length, 104, 37, { ratio: 1.12, gap: 1.4, maxTile: 20 });
  const auction = t.phase === 'endgame_bidding';
  return (
    <div className="tvs-bidding">
      <GiftCard t={t} help={auction ? <>Everyone can bid. <strong>{nameOf(game, sitter)}</strong> duels the highest bidder.</>
        : <>Everyone but <strong>{nameOf(game, sitter)}</strong> bids. Highest earns the right to throw hands.</>} />
      <section className="tvs-sealed">
        <header className="tvs-sealed-head">
          <div>
            <span className="tvs-pill tint mo-down">Bid on your phone</span>
            <h1 className="tvs-h2 f6-disp">{sealed.length === bidders.length && bidders.length ? 'All bids are in!' : 'Locking in…'}</h1>
          </div>
          <div className="tvs-sealed-count f6-disp fz-riso f6-num" data-testid="tv-sealed-count"><b key={sealed.length} className="mo-bump">{sealed.length}</b><span>of {bidders.length}</span></div>
        </header>
        <div className="tvs-bar"><div className="tvs-bar-fill" style={{ width: `${pct}%` }} /></div>
        <div className="tvs-bidders" style={vars({ '--cols': grid.cols, '--tile': u(grid.tile) })}>
          {bidders.map((id, i) => {
            const isIn = !!t.bids?.[id];
            return (
              <div key={id} className={`tvs-bidder${isIn ? ' in' : ''} mo-pop`} style={vars({ '--i': i })} data-player-id={id}>
                <Av id={id} size={grid.tile * 0.5} />
                <Fit text={nameOf(game, id)} width={grid.tile - 1.4} max={Math.min(3, grid.tile * 0.17)} min={1.3} lines={2} className="tvs-bidder-name" />
                {/* never an amount: the TV is sent sealed bids as sealed */}
                {isIn ? <span key="in" className="tvs-bidder-state in mo-pop" data-testid={`sealed-${id}`}>✓ Sealed</span>
                  : <span key="out" className="tvs-bidder-state">…bidding</span>}
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}

// --- the reveal -----------------------------------------------------------------------------------------

/** How long after the reveal the last bar is up (the outcome waits for it). */
const BAR_FIRST_MS = 400, BAR_STEP_MS = 600, MAX_BARS = 8;

export function RevealScene({ t }: { t: Turn }) {
  const { game } = useHost();
  const entries = Object.entries(t.bids || {}).sort(([, a], [, b]) => b.amount - a.amount);
  // a bar each for the best bids; a big table's lower bids and its passes go in a row underneath
  const bars = entries.filter(([, b]) => b.amount > 0).slice(0, MAX_BARS);
  const rest = entries.filter((e) => !bars.includes(e));
  const [shown, setShown] = useState(0);
  const key = `${t.giftId}|${t.misfitId}|${t.openerId}|${bars.length}`;
  useEffect(() => {
    setShown(0);
    const timers = bars.map((_, i) => setTimeout(() => setShown(i + 1), BAR_FIRST_MS + i * BAR_STEP_MS));
    return () => timers.forEach(clearTimeout);
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  const done = shown >= bars.length;
  const top = bars[0]?.[1].amount ?? 0;
  const colW = Math.min(24, (118 - 1.6 * Math.max(0, bars.length - 1)) / Math.max(1, bars.length));
  return (
    <div className="tvs-reveal">
      <GiftCard t={t} small />
      <section className="tvs-reveal-main">
        <header className="tvs-reveal-head">
          <span className="tvs-pill tint mo-down"><Icon name="envelope" /> Envelope, please</span>
          <h1 className="tvs-h2 f6-disp mo-up">The bids, <span className="tvs-accent">revealed.</span></h1>
        </header>
        {bars.length ? (
          <div className="tvs-chart" data-testid="tv-reveal-chart" style={vars({ '--col': u(colW) })}>
            {bars.map(([pid, bid], i) => {
              const isTop = bid.amount > 0 && bid.amount === top;
              const runner = !isTop && i === 1 && bid.amount > 0;
              const up = i < shown;
              return (
                <div key={pid} className={`tvs-col${up ? ' shown' : ''}${isTop ? ' top' : runner ? ' runner' : ''}`} style={vars({ '--h': `${((bid.amount / Math.max(1, top)) * 78 + 22).toFixed(1)}%` })}>
                  <div className="tvs-col-well">
                    {up && (isTop || runner) && <span className={`fz-sticker tvs-sticker tvs-col-sticker${runner ? ' yel' : ''}`}>{isTop ? (bars.filter(([, b]) => b.amount === top).length > 1 ? 'TIED' : 'TOP BID') : 'SECOND'}</span>}
                    <div className="tvs-colbar"><span className="tvs-col-amount f6-disp f6-num">{up ? (bid.amount > 0 ? bid.amount : 'pass') : '?'}</span></div>
                  </div>
                  <Av id={pid} size={Math.min(6.5, colW * 0.42)} />
                  <Fit text={nameOf(game, pid)} width={colW - 0.6} max={Math.min(2.8, colW * 0.2)} min={1.25} lines={2} className="tvs-col-name" />
                </div>
              );
            })}
          </div>
        ) : <div className="tvs-nobids mo-pop"><span className="fz-sticker tvs-sticker big ink">NO BIDS</span></div>}
        <div className="tvs-rest">
          {done && rest.length > 0 && <>
            <span className="tvs-k mo-fade">{bars.length ? 'Also in' : 'Everyone passed'}</span>
            {rest.map(([pid, bid], i) => (
              <span key={pid} className="tvs-rest-item mo-pop" style={vars({ '--i': i })} title={nameOf(game, pid)}><Av id={pid} size={3.8} /><b className="f6-num">{bid.amount > 0 ? bid.amount : 'pass'}</b></span>
            ))}
          </>}
        </div>
        <div className="tvs-outcome">{done && <Outcome t={t} />}</div>
      </section>
    </div>
  );
}

/** What the bids mean, once they are all up: who is top, a tie, and who the table waits for. */
function Outcome({ t }: { t: Turn }) {
  const { game } = useHost();
  const bids = rankedBids(t);
  const auction = t.phase === 'endgame_reveal';
  const opener = nameOf(game, auction ? t.activePlayerId : t.openerId);
  if (!bids.length) {
    return auction ? <><p className="tvs-outcome-line mo-up">Nobody bid for the Misfit.</p><Wait>{opener} picks someone to duel</Wait></>
      : <><p className="tvs-outcome-line mo-up">Nobody bid.</p><Wait>{opener} chooses: keep it, or play Grump's Gambit</Wait></>;
  }
  const [[c1, top], second] = bids;
  if (auction) return <p className="tvs-outcome-line mo-up">Highest bid: <strong>{nameOf(game, c1)}</strong> ({chipWord(top)}). Next: the duel with {opener}.</p>;
  const tied = topTiedBidders(t);
  if (tied.length >= 2) {
    return <>
      <p className="tvs-outcome-line mo-up"><strong className="tvs-gold">{tied.length === 2 ? 'A tie!' : `A ${tied.length}-way tie!`}</strong> {tied.length} players bid {chipWord(top)}. A party game settles it.</p>
      <Wait>{opener} chooses: fight or yield</Wait>
    </>;
  }
  if (second) {
    return <>
      <p className="tvs-outcome-line mo-up">Highest bid: <strong>{nameOf(game, c1)}</strong> ({chipWord(top)}). Second: {nameOf(game, second[0])}.</p>
      <Wait>{opener} chooses: fight or yield</Wait>
    </>;
  }
  return <p className="tvs-outcome-line mo-up">Highest bid: <strong>{nameOf(game, c1)}</strong> ({chipWord(top)}). Next: the duel with {opener}.</p>;
}

// --- who faces whom ------------------------------------------------------------------------------------------

export function VersusScene({ a, b, kicker, title, foot, mid = 'VS', roles, children }: {
  a?: PlayerId; b?: PlayerId; kicker: string; title?: ReactNode; foot?: ReactNode; mid?: string; roles?: [string, string]; children?: ReactNode;
}) {
  const { game } = useHost();
  const side = (id: PlayerId | undefined, cls: string, role?: string) => (
    <div className={`tvs-side ${cls} ${cls === 'red' ? 'mo-right' : 'mo-left'}`} data-player-id={id}>
      <div className="fz-halftone tvs-halftone" />
      <Av id={id} size={children ? 15 : 25} />
      <Fit text={nameOf(game, id)} width={58} max={children ? 6 : 8.5} min={3} lines={2} upper className="tvs-side-name f6-disp" />
      {role && <div className="tvs-side-role">{role}</div>}
    </div>
  );
  return (
    <div className={`tvs-versus${children ? ' with-stage' : ''}`}>
      <header className="tvs-versus-head">
        <span className="fz-sticker tvs-sticker big ink">{kicker}</span>
        {title && <h1 className="tvs-h2 f6-disp mo-up">{title}</h1>}
      </header>
      <div className="tvs-versus-row">
        {side(a, 'red', roles?.[0])}
        <span className="tvs-vs f6-disp fz-riso f6-wiggle">{mid}</span>
        {side(b, 'green', roles?.[1])}
      </div>
      {children && <div className="tvs-versus-stage">{children}</div>}
      {foot && <div className="tvs-versus-foot mo-up" style={vars({ '--i': 2 })}>{foot}</div>}
    </div>
  );
}

export function TiebreakDuelScene({ t }: { t: Turn }) {
  const { step, choice } = useDuelGame();          // the picker, the wheel or the duel's name: only when duel picking is on
  return (
    <VersusScene a={t.challengerId} b={t.defenderId} kicker="TIEBREAK" foot={<span className="tvs-gold">The winner becomes the Challenger!</span>}>
      {(step || choice) ? <DuelGameStage compact /> : null}
    </VersusScene>
  );
}

// --- one player deciding -------------------------------------------------------------------------------------

export function SpotScene({ who, kicker, title, sub, wait, children }: { who?: PlayerId; kicker: string; title: string; sub?: ReactNode; wait?: ReactNode; children?: ReactNode }) {
  return (
    <div className="tvs-spot">
      {who && <div className="tvs-spot-face mo-pop"><Av id={who} size={30} /></div>}
      <div className="tvs-spot-txt">
        <span className="fz-sticker tvs-sticker big ink">{kicker}</span>
        <Fit text={title} width={who ? 112 : 155} max={children ? 6.6 : 8.5} min={3.4} lines={2} className="tvs-h1 f6-disp mo-up" />
        {sub && <p className="tvs-sub mo-up" style={vars({ '--i': 1 })}>{sub}</p>}
        {children}
        {wait && <Wait>{wait}</Wait>}
      </div>
    </div>
  );
}

/** The two ways a Misfit Lottery turn can go. */
export function Paths({ who }: { who: string }) {
  return (
    <div className="tvs-paths">
      <div className="tvs-path mo-up" style={vars({ '--i': 1 })}><b><Icon name="target" /> Steal a gift</b><span>Duel a player for theirs. The loser takes the Misfit.</span></div>
      <div className="tvs-path mo-up" style={vars({ '--i': 2 })}><b><Icon name="bid" /> Auction the Misfit</b><span>Everyone bids. {who} duels the highest bidder.</span></div>
    </div>
  );
}

// --- AI Draw Off (the two duellists draw, an AI critic picks) ---------------------------------------------

export function DrawOffScene() {
  const { game } = useHost();
  const now = useServerNow(250);
  const mg = game.minigame as DrawOff | undefined;
  if (!mg || mg.type !== 'draw_off') return <SpotScene kicker="AI DRAW OFF" title="Sharpening the pencils…" />;
  const subs = mg.submissions || {};
  const winners = new Set(mg.winners || []);
  const results = mg.phase === 'results';
  const order = results ? (mg.rankings || mg.participants) : mg.participants;
  const head = mg.phase === 'prompt_reveal' ? 'Drawing starts in a moment…'
    : mg.phase === 'drawing' ? `${Object.keys(subs).length} of ${mg.participants.length} done`
    : mg.phase === 'judging' ? 'The AI Art Critic is judging…'
    : mg.winners?.length ? `${nameOf(game, mg.winners[0])} wins the Draw Off!` : 'The critic could not decide: the host picks';
  return (
    <div className={`tvs-drawoff ${mg.phase}`}>
      <header className="tvs-drawoff-head">
        <span className="tvs-pill ink"><Icon name="critic" /> AI critic</span>
        <Fit text={`“${mg.prompt}”`} width={150} max={results ? 4.6 : 7} min={2.6} lines={2} className="tvs-drawoff-prompt f6-disp mo-slam" />
        <div className="tvs-drawoff-status">
          <span data-testid="tv-drawoff-status">{head}</span>
          {mg.phase === 'drawing' && mg.drawingEndTime != null && <span className="tvs-clock f6-disp f6-num">{secsLeft(mg.drawingEndTime, now)}s</span>}
        </div>
      </header>
      <div className="tvs-drawoff-cards">
        {order.map((pid, i) => {
          const s = mg.scores?.[pid];
          const img = results ? mg.submissions?.[pid] : undefined;
          return (
            <div key={pid} className={`tvs-drawcard${winners.has(pid) ? ' winner' : ''} mo-up`} style={vars({ '--i': i })}>
              {winners.has(pid) && <span className="fz-sticker mint tvs-sticker big tvs-drawcard-sticker"><Icon name="trophy" /> WINNER</span>}
              <div className="tvs-drawcard-who"><Av id={pid} size={5.5} /><Fit text={nameOf(game, pid)} width={results ? 34 : 60} max={3.4} min={1.8} className="f6-disp" /></div>
              <div className="tvs-drawcard-well">
                {img && img !== 'data:,' ? <img src={img} alt={`${nameOf(game, pid)}'s drawing`} />
                  : <span className="tvs-drawcard-wait">{results ? 'No drawing' : mg.phase === 'judging' ? 'With the critic' : subs[pid] ? <><Icon name="check" /> Done! Hidden till judging</> : mg.phase === 'drawing' ? 'Drawing…' : 'Get ready'}</span>}
              </div>
              {results && (s || mg.failed) && (
                <div className="tvs-drawcard-score">
                  {s && <div className="tvs-drawcard-n f6-disp f6-num">{s.rating}<span>/100</span></div>}
                  {s?.critique && <p className="tvs-drawcard-critique">“{s.critique}”</p>}
                </div>
              )}
            </div>
          );
        })}
      </div>
      {/* the one announcement: what the winner takes (then the duel is settled with no verdict after it) */}
      {results && !!mg.winners?.length && <StakesLines stakes={mg.stakes} game={game} big />}
      {results && mg.wittyVerdict && <p className="tvs-drawoff-verdict mo-up"><Icon name="masks" /> {mg.wittyVerdict}</p>}
      {results && !mg.winners?.length && <p className="tvs-drawoff-verdict mo-up" data-testid="drawoff-no-contest">The critic could not decide ({mg.failed}). Next, the host picks the winner.</p>}
    </div>
  );
}
