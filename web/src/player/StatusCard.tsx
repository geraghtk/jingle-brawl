// The phone's status card: what's happening and — when it's my move — my options.
import { useState } from 'react';
import type { ReactNode } from 'react';
import * as A from '../core/actions';
import * as D from '../core/duel';
import { rng } from '../core/rng';
import { gambitTargets, rankedBids, stealTargets, topTiedBidders } from '../core/rules';
import { minCost } from '../core/settings';
import { partyScore } from '../core/scoring';
import type { Turn } from '../core/types';
import { Avatar } from '../ui/avatar';
import { dividendNote, reprisalNote, spectatorLoserNote, voteStakeNote } from '../ui/notes';
import { voteStatus } from '../core/voting';
import { chooseDuelGame, duelGameStep, duelOptions, duelPicker, SPIN_MS, spinDuelGame } from '../core/duelgame';
import { GAMES } from '../core/party';
import { DuelWheel } from '../ui/DuelWheel';
import { useServerOffset } from '../firebase/game';
import { useSecondsLeft } from '../ui/PartyResults';
import { usePlayer } from './PlayerContext';
import { Glyphs, Icon } from '../ui/icons';

const r = () => rng(Date.now() ^ Math.floor(Math.random() * 1e9));

function Card({ tone, children }: { tone?: 'your-turn' | 'active' | 'callit'; children: ReactNode }) {
  return <div className={`card status-card${tone ? ` ${tone}` : ''}`} id="status-card" data-testid="status-card">{children}</div>;
}

const Simple = ({ icon, title, desc, tone }: { icon: string; title: ReactNode; desc?: ReactNode; tone?: 'your-turn' | 'active' }) => (
  <Card tone={tone}>
    <div className="status-icon"><Glyphs>{icon}</Glyphs></div>
    <div className="status-title">{title}</div>
    {desc != null && <div className="status-desc">{desc}</div>}
  </Card>
);

const ordinal = (n: number) => {
  const s = ['th', 'st', 'nd', 'rd'], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
};

export function StatusCard() {
  const { game, me } = usePlayer();
  const t = game.currentTurn;
  if (!t) {
    if (game.phase === 'lobby') return <Lobby />;
    if (game.phase === 'main') return <Idle />;
    if (game.phase === 'endgame') return <EndgameIdle />;
    return <Final />;
  }
  switch (t.phase) {
    case 'bidding': return t.openerId === me ? <OpenerWaiting t={t} /> : null;
    case 'reveal': return <Reveal t={t} />;
    case 'yield_choice': return <YieldChoice t={t} />;
    case 'duel': return <Duel t={t} />;
    case 'reprisal': return <ReprisalWatch t={t} />;
    case 'endgame_choice': return <EndgameChoice t={t} />;
    case 'endgame_steal_select': return <StealSelect t={t} />;
    case 'endgame_nobids': return <NoBids t={t} />;
    case 'endgame_bidding': case 'endgame_reveal': return <EndgameAuction t={t} />;
    case 'tiebreaker_pending': return <Simple icon="⚖️" title="Tiebreaker Pending" desc="Waiting for tiebreaker to start..." />;
    default: return <Simple icon="⏳" title="Waiting..." desc="Watch the main screen." />;
  }
}

function Lobby() {
  const { game, me, room, player } = usePlayer();
  const entries = Object.entries(game.players || {});
  return (
    <Card>
      <div className="f6-cap">Room {room} · you're in</div>
      <div className="lobby-title f6-disp fz-riso">Yuel <span className="lobby-accent">Duel</span></div>
      <div className="lobby-tag">High stakes. Low blows. Questionable gifts.</div>
      <div className="final-stats">
        <div className="fstat"><div className="fstat-v f6-disp f6-num"><span className="f6-coin" /> {player.chips}</div><div className="fstat-k">your chips</div></div>
        <div className="fstat accent"><div className="fstat-v f6-disp f6-num">{entries.length}</div><div className="fstat-k">in the lobby</div></div>
      </div>
      <div className="f6-cap lineup-label">Lineup</div>
      <div className="lineup">
        {entries.length ? entries.map(([id, p]) => (
          <div className="lineup-row" key={id}>
            <Avatar name={p.name} avatar={p.avatar} />
            <span className="lineup-nm">{p.name}</span>
            <span className={`f6-chip lineup-chip${id === me ? ' you' : ''}`}>{id === me ? 'You' : 'Ready'}</span>
          </div>
        )) : <div className="lineup-empty">Just you so far…</div>}
      </div>
      <div className="status-desc" style={{ marginTop: 12 }}>Waiting for the host to start…</div>
    </Card>
  );
}

function Idle() {
  const { player, cfg } = usePlayer();
  if (player.giftId) {
    return (
      <Card>
        <div className="status-icon"><Icon name="check" /></div>
        <div className="status-title">You have a gift!</div>
        <div className="status-desc">Watch the main screen - someone might challenge you!</div>
        <div className="help-box success">
          <div className="help-title"><Icon name="shield" /> Defending is FREE</div>
          <div className="help-content">
            If challenged, you <strong>don't pay anything</strong> to defend.<br />
            If you lose: {dividendNote(cfg, true)} and a <strong>Reprisal</strong> chance!
          </div>
        </div>
      </Card>
    );
  }
  return (
    <Card tone="active">
      <div className="status-icon"><Icon name="santa" /></div>
      <div className="status-title">You're in the bag!</div>
      <div className="status-desc">Waiting to be drawn to open a gift...</div>
      <div className="help-box">
        <div className="help-title"><Icon name="idea" /> What happens next?</div>
        <div className="help-content">
          When drawn, you'll open a gift. Others can <strong>bid chips</strong> to challenge you for it.<br />
          No bids? Use <strong>Grinch's Gambit</strong> to steal someone else's gift!
        </div>
      </div>
    </Card>
  );
}

function EndgameIdle() {
  const { player } = usePlayer();
  return (
    <Card tone="active">
      <div className="status-icon"><Icon name="bid" /></div>
      <div className="status-title">Misfit Lottery!</div>
      <div className="status-desc">{player.giftId ? 'You have a gift! Watch for auction opportunities.' : 'You need a gift - watch for your turn!'}</div>
      <div className="help-box warning">
        <div className="help-title"><Icon name="tent" /> Endgame Rules</div>
        <div className="help-content">
          <strong>Path A:</strong> Giftless player can steal any player's gift (loser takes misfit)<br />
          <strong>Path B:</strong> Auction for misfit - anyone can bid!
        </div>
      </div>
    </Card>
  );
}

function Final() {
  const { game, me, player } = usePlayer();
  const standings = Object.entries(game.players || {}).map(([id, p]) => ({ id, chips: p.chips || 0 })).sort((a, b) => b.chips - a.chips);
  const rank = standings.findIndex((e) => e.id === me) + 1;
  const gift = player.giftId ? game.gifts?.[player.giftId] : null;
  const score = partyScore(player);
  return (
    <Card>
      <div className="f6-cap">Game over · your night</div>
      <div className="final-place f6-disp fz-riso">{rank > 0 ? `${ordinal(rank)} place.` : 'Game over.'}</div>
      <div className="final-gift-wrap">
        <div className={`final-gift-tile${gift ? '' : ' none'}`}>{gift ? <Icon name="gift" /> : '🫥'}</div>
        {rank === 1 && <span className="fz-sticker yel final-champ"><Icon name="trophy" /> CHAMP</span>}
      </div>
      <div className="final-gift-name">{gift ? gift.name : 'No gift this year'}</div>
      <div className="final-stats">
        <div className="fstat"><div className="fstat-v f6-disp f6-num">{player.chips}</div><div className="fstat-k">chips</div></div>
        <div className="fstat accent"><div className="fstat-v f6-disp f6-num">{gift ? <><Icon name="naughty" /> {gift.naughtyLevel ?? 0}</> : 'none'}</div><div className="fstat-k">gift naughty</div></div>
        {score > 0 && <div className="fstat"><div className="fstat-v f6-disp f6-num"><Icon name="medal" /> {score}</div><div className="fstat-k">party score</div></div>}
      </div>
      <div className="status-desc" style={{ marginTop: 10 }}>Thanks for playing Yuel Duel!</div>
    </Card>
  );
}

function OpenerWaiting({ t }: { t: Turn }) {
  const { game } = usePlayer();
  const gift = t.giftId ? game.gifts?.[t.giftId] : undefined;
  return (
    <Card tone="your-turn">
      <div className="status-icon"><Icon name="gift" /></div>
      <div className="status-title">You opened "{gift?.name || 'a gift'}"!</div>
      <div className="status-desc">Waiting for others to bid...</div>
      <div className="help-box warning">
        <div className="help-title"><Icon name="wait" /> What's happening?</div>
        <div className="help-content">
          Others are deciding if they want to <strong>challenge you</strong> for this gift.<br /><br />
          <strong>If someone bids:</strong> Highest bidder fights you in a duel.<br />
          <strong>If no one bids:</strong> You keep it OR use <strong>Grinch's Gambit</strong> to steal someone else's gift!
        </div>
      </div>
    </Card>
  );
}

const nm = (game: ReturnType<typeof usePlayer>['game'], id?: string | null) => (id && game.players?.[id]?.name) || 'Unknown';

function Reveal({ t }: { t: Turn }) {
  const { game, me, act, cfg } = usePlayer();
  const [picking, setPicking] = useState(false);
  const bids = rankedBids(t);
  const myBid = t.bids?.[me]?.amount || 0;
  if (t.openerId !== me) {
    return (
      <Card tone="active">
        <div className="status-icon"><Icon name="eye" /></div>
        <div className="status-title">Bids Revealed!</div>
        <div className="status-desc">
          {bids.length === 0 ? 'No bids! Opener decides next...' : `${bids.length} bid${bids.length > 1 ? 's' : ''} placed. ${myBid > 0 ? `Your bid: ${myBid}` : ''}`}
        </div>
        {myBid > 0 && (
          <div className="help-box">
            <div className="help-title"><Icon name="idea" /> What's Next?</div>
            <div className="help-content">If you're the highest bidder, you'll duel the opener!<br />If there's a tie, there may be a tiebreaker first.</div>
          </div>
        )}
      </Card>
    );
  }
  if (bids.length === 0) {
    if (picking) return <GambitPicker t={t} onCancel={() => setPicking(false)} />;
    const canGambit = gambitTargets(game, t.giftId, me).length > 0;
    return (
      <Card tone="your-turn">
        <div className="status-icon"><Icon name="gift" /></div>
        <div className="status-title">No Challengers!</div>
        <div className="status-desc">Nobody bid. You can keep your gift or use Grinch's Gambit!</div>
        <div className="help-box success">
          <div className="help-title">Your Options</div>
          <div className="help-content">
            <strong>Keep Gift:</strong> It's yours! Move on to next player.<br />
            <strong>Grinch's Gambit:</strong> Pay the Minimum Cost to challenge anyone else's gift!
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
          <button className="btn btn-primary" data-testid="keep-gift" onClick={() => act(A.keepGift)}><Icon name="gift" /> Keep My Gift</button>
          {canGambit
            ? <button className="btn btn-gold" data-testid="gambit" onClick={() => setPicking(true)}><Icon name="gambit" /> Grinch's Gambit</button>
            : <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center' }}>No other gifts to challenge yet</div>}
        </div>
      </Card>
    );
  }
  if (bids.length === 1) {
    const [cid, amount] = bids[0];
    return (
      <Card tone="your-turn">
        <div className="status-icon"><Icon name="duel" /></div>
        <div className="status-title">Duel Time!</div>
        <div className="status-desc">{nm(game, cid)} bid {amount} chips to challenge you!</div>
        <div className="help-box">
          <div className="help-title"><Icon name="shield" /> Get Ready to Defend!</div>
          <div className="help-content">
            <strong>Defending is FREE</strong> - you don't pay anything!<br />
            <strong>If you win:</strong> Keep your gift + win their bid (minus Santa Tax).<br />
            <strong>If you lose:</strong> {dividendNote(cfg, true)}{reprisalNote(cfg, t)}
          </div>
        </div>
      </Card>
    );
  }
  const tied = topTiedBidders(t);
  const [[c1, top], [c2]] = bids;
  const fight = (a: string | null, b: string | null, tie: boolean) => act((g) => A.openerFight(g, a, b, tie, r()));
  const yieldTo = (a: string | null, b: string | null) => act((g) => A.openerYield(g, a, b, r()));
  const choice = (title: ReactNode, desc: ReactNode, fightHelp: ReactNode, yieldHelp: ReactNode, fightBtn: ReactNode, yieldBtn: ReactNode, onFight: () => void, onYield: () => void, icon = '⚖️') => (
    <Card tone="your-turn">
      <div className="status-icon"><Glyphs>{icon}</Glyphs></div>
      <div className="status-title">{title}</div>
      <div className="status-desc">{desc}</div>
      <div className="help-box warning">
        <div className="help-title"><Icon name="run" /> Fight or Yield?</div>
        <div className="help-content"><strong>Fight:</strong> {fightHelp}<br /><strong>Yield:</strong> {yieldHelp}</div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
        <button className="btn btn-primary" data-testid="fight" onClick={onFight}>{fightBtn}</button>
        <button className="btn btn-gold" data-testid="yield" onClick={onYield}>{yieldBtn}</button>
      </div>
    </Card>
  );
  if (tied.length >= 3) {
    return choice(`${tied.length}-WAY TIE!`, `${tied.map((id) => nm(game, id)).join(', ')} all bid ${top}. Two will be drawn at random. You decide!`,
      'The two drawn duel for the right to challenge YOU.', 'The two drawn fight each other for your gift directly.',
      '⚔️ Fight! (Tiebreaker first)', '🏃 Yield! (two drawn at random)', () => fight(null, null, true), () => yieldTo(null, null));
  }
  if (tied.length === 2) {
    return choice(`TIE! ${nm(game, c1)} & ${nm(game, c2)}`, `Both bid ${top} chips. You decide!`,
      'They duel for the right to challenge YOU.', 'They fight each other for your gift directly.',
      '⚔️ Fight! (Tiebreaker first)', `🏃 Yield! (${nm(game, c1)} vs ${nm(game, c2)})`, () => fight(c1, c2, true), () => yieldTo(c1, c2));
  }
  return choice(`${nm(game, c1)} wants your gift!`, `Bid: ${top} chips. Second place: ${nm(game, c2)}`,
    `Duel ${nm(game, c1)}!`, `${nm(game, c1)} fights ${nm(game, c2)} instead.`,
    `⚔️ Fight ${nm(game, c1)}!`, `🏃 Yield! (${nm(game, c1)} vs ${nm(game, c2)})`, () => fight(c1, c2, false), () => yieldTo(c1, c2), '⚔️');
}

function GambitPicker({ t, onCancel }: { t: Turn; onCancel: () => void }) {
  const { game, me, player, cfg, act } = usePlayer();
  return (
    <Card tone="your-turn">
      <div className="status-icon"><Icon name="naughty" /></div>
      <div className="status-title">Grinch's Gambit!</div>
      <div className="status-desc">Choose a gift to challenge:</div>
      <div style={{ maxHeight: 300, overflowY: 'auto', margin: '1rem 0' }}>
        {gambitTargets(game, t.giftId, me).map(([id, gift]) => {
          const cost = minCost(cfg, gift.naughtyLevel || 0);
          return <TargetOption key={id} testid={`gambit-target-${id}`} title={gift.name} sub={`Owner: ${nm(game, gift.ownerId)}`}
            naughty={gift.naughtyLevel} cost={cost} afford={player.chips >= cost} onPick={() => act((g) => A.gambit(g, id))} />;
        })}
      </div>
      <button className="btn btn-secondary" data-testid="cancel-gambit" onClick={onCancel} style={{ width: '100%' }}>← Cancel</button>
    </Card>
  );
}

/** A gift you could pay to challenge for (Gambit / Steal). */
export function TargetOption({ title, sub, naughty, cost, afford, onPick, testid }: {
  title: string; sub: string; naughty: number; cost: number; afford: boolean; onPick: () => void; testid: string;
}) {
  return (
    <div className={`action-option ${afford ? '' : 'disabled'}`} data-testid={testid} onClick={afford ? onPick : undefined}
      style={{ background: 'var(--bg-card-hover)', padding: '1rem', borderRadius: 8, marginBottom: '0.5rem', cursor: afford ? 'pointer' : 'not-allowed', opacity: afford ? 1 : 0.5 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontWeight: 600 }}>{title}</div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{sub}</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          {naughty > 0 && <span style={{ background: 'var(--accent-red)', padding: '2px 6px', borderRadius: 4, fontSize: '0.75rem' }}><Icon name="naughty" /> {naughty}</span>}
          <div style={{ fontSize: '0.85rem', color: afford ? 'var(--accent-gold)' : 'var(--accent-red)' }}><Icon name="moneybag" /> {cost} chip{cost > 1 ? 's' : ''}</div>
        </div>
      </div>
    </div>
  );
}

function YieldChoice({ t }: { t: Turn }) {
  const { game, me, act } = usePlayer();
  const a = nm(game, t.selectedC1), b = nm(game, t.selectedC2);
  const amount = t.bids?.[t.selectedC1!]?.amount || 0;
  if (t.openerId !== me) return <Simple icon="🎲" tone="active" title={`${a} & ${b} Selected!`} desc={`Waiting for ${nm(game, t.openerId)} to choose: Fight or Yield?`} />;
  return (
    <Card tone="your-turn">
      <div className="status-icon"><Icon name="dice" /></div>
      <div className="status-title">{a} & {b} Selected!</div>
      <div className="status-desc">Both bid {amount} chips. Your choice!</div>
      <div className="help-box warning">
        <div className="help-title"><Icon name="run" /> Fight or Yield?</div>
        <div className="help-content"><strong>Fight:</strong> They duel for the right to challenge YOU.<br /><strong>Yield:</strong> They fight each other for your gift directly.</div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
        <button className="btn btn-primary" data-testid="fight" onClick={() => act((g) => A.openerFight(g, t.selectedC1!, t.selectedC2!, true, r()))}><Icon name="duel" /> Fight! (Tiebreaker first)</button>
        <button className="btn btn-gold" data-testid="yield" onClick={() => act((g) => A.openerYield(g, t.selectedC1!, t.selectedC2!, r()))}><Icon name="run" /> Yield! ({a} vs {b})</button>
      </div>
    </Card>
  );
}

function Duel({ t }: { t: Turn }) {
  const { game, me, act, cfg, player, toast } = usePlayer();
  const offset = useServerOffset();
  const secondsLeft = useSecondsLeft(t.voteEndsAt, offset); // re-renders while the vote clock runs
  const votes = voteStatus(game, Date.now() + offset);
  const gift = t.giftId ? game.gifts?.[t.giftId] : undefined;
  const mine = t.challengerId === me || t.defenderId === me;
  const step = duelGameStep(game);
  const spunAt = typeof t.wheel?.spunAt === 'number' ? t.wheel.spunAt : null;
  const spinLeft = useSecondsLeft(spunAt != null ? spunAt + SPIN_MS : undefined, offset);
  const spinning = spunAt != null && spinLeft > 0;
  const picking = !!step || spinning;
  const picker = duelPicker(game);
  const appName = (o: { kind: string; app?: keyof typeof GAMES; name: string }) => (o.kind === 'app' && o.app ? `${GAMES[o.app].emoji} ${o.name}` : o.name);
  const duelGameLine = t.wheel && spinning
    ? <DuelWheel spin={t.wheel} offset={offset} size={230} />
    : t.duelGame
      ? <div className="duel-game-line" data-testid="phone-duel-game"><span>The duel</span><strong><Glyphs>{appName(t.duelGame)}</Glyphs></strong>{t.duelGame.kind === 'app' && <small>on your phone · everyone plays</small>}</div>
      : step === 'picker' && picker !== me ? <div className="duel-game-line pending"><Icon name="target" /> {nm(game, picker)} is picking the duel…</div>
      : step === 'wheel' ? <div className="duel-game-line pending"><Icon name="wheel" /> Spinning the wheel…</div> : null;

  // The picker (the defender, or an under-16 duellist) chooses what gets played — or hands it to the wheel
  if (step === 'picker' && picker === me) {
    const defending = t.defenderId === me;
    const rival = nm(game, defending ? t.challengerId : t.defenderId);
    return (
      <Card tone="your-turn">
        <div className="f6-cap">{(player.age ?? 99) < 16 ? "You're under 16, so you pick" : "You're defending"}</div>
        <div className="status-title">Pick the duel</div>
        <div className="status-desc">{defending ? `${rival} challenged you${gift ? ` for the ${gift.name}` : ''}.` : `You're up against ${rival}.`} You choose how you fight.</div>
        <div className="duel-picker" role="list">
          {duelOptions(game).map((o, i) => (
            <button key={o.name} role="listitem" className="duel-pick" data-testid={`duel-game-${i}`} onClick={() => act((g) => chooseDuelGame(g, me, i))}>
              <span className="duel-pick-name"><Glyphs>{appName(o)}</Glyphs></span>
              {o.kind === 'app' && <small>played on phones · everyone joins in</small>}
            </button>
          ))}
        </div>
        <button className="btn btn-secondary" data-testid="duel-game-spin" style={{ width: '100%', marginTop: 10 }}
          onClick={() => act((g) => spinDuelGame(g, { now: Date.now() + offset, rng: r() }, me))}><Icon name="wheel" /> Let the wheel pick</button>
      </Card>
    );
  }
  const tally = !picking && votes.eligible.length > 0 && (
    <div className="callit-tally" data-testid="vote-tally">
      {votes.blocking
        ? <>The table is calling it · <strong>{votes.voted.length} of {votes.eligible.length}</strong> in{votes.endsAt != null && <> · {secondsLeft}s</>}</>
        : <>{votes.voted.length} of {votes.eligible.length} called it</>}
    </div>
  );

  if (mine) {
    const defending = t.defenderId === me;
    const opponent = nm(game, defending ? t.challengerId : t.defenderId);
    return (
      <Card tone="your-turn">
        <div className="f6-cap">Duel</div>
        <div className="duel-you-row">
          <span className="duel-side f6-disp" style={{ color: 'var(--accent)' }}>YOU</span>
          <span className="duel-vs-mini f6-disp fz-riso f6-wiggle">vs</span>
          <span className="duel-side f6-disp" style={{ color: 'var(--accent2)' }}>{opponent.toUpperCase()}</span>
        </div>
        {!!t.pot && (
          <div className="final-stats" style={{ margin: '4px 0 10px' }}>
            <div className="fstat"><div className="fstat-v f6-disp f6-num">{t.pot} <span className="f6-coin" /></div><div className="fstat-k">to the winner</div></div>
          </div>
        )}
        {duelGameLine}
        <div className="status-desc">{defending ? `You're defending against ${opponent}.` : `You're challenging ${opponent}.`}</div>
        {tally}
        <details className="howto">
          <summary>{defending ? 'What defending costs you' : 'What you have on the line'}</summary>
          <div className="howto-body">
            {defending
              ? <><strong>Defending is free.</strong><br /><strong>Win:</strong> keep your gift and take the pot.<br /><strong>Lose:</strong> {dividendNote(cfg, true)}{reprisalNote(cfg, t)}</>
              : <>Your bid is in the pot.<br /><strong>Win:</strong> take the gift and the pot (minus Santa Tax).<br /><strong>Lose:</strong> {dividendNote(cfg, false)}{reprisalNote(cfg, t)}</>}
          </div>
        </details>
      </Card>
    );
  }

  // A spectator: the vote is the whole screen — tap a corner to back them
  const myVote = t.duelVotes?.[me]?.votedFor;
  const canVote = !t.isTiebreak && !myVote && votes.accepting && !picking;
  const record = (id: string) => {
    const p = game.players?.[id];
    const w = p?.duelWins || 0, l = p?.duelLosses || 0;
    return w + l ? `${w}–${l} in duels` : 'First duel';
  };
  const vote = async (id: string) => {
    if (await act((g) => A.voteOnDuel(g, me, id, Date.now() + offset))) toast(`Voted for ${nm(game, id)}! 🗳️`);
  };
  const corner = (id: string, side: 'red' | 'green') => {
    const inner = (
      <>
        <span className="callit-corner-tag">{side === 'red' ? 'Red corner' : 'Green corner'}</span>
        <span className="callit-name f6-disp">{nm(game, id)}</span>
        <span className="callit-record">{record(id)}</span>
        {myVote === id && <span className="fz-sticker yel callit-pick">Your pick</span>}
      </>
    );
    const cls = `callit-corner ${side}${myVote === id ? ' picked' : ''}${myVote && myVote !== id ? ' faded' : ''}`;
    return canVote
      ? <button key={id} className={cls} data-testid={`vote-${id}`} aria-label={`🏆 ${nm(game, id)} wins`} onClick={() => vote(id)}>{inner}</button>
      : <div key={id} className={cls}>{inner}</div>;
  };
  return (
    <Card tone="callit">
      <div className="callit-head">
        <span className="callit-title f6-disp">Duel in progress</span>
        {canVote && votes.endsAt != null && <span className="callit-clock f6-num" aria-label={`${secondsLeft} seconds to vote`}>{secondsLeft}s</span>}
      </div>
      <div className="callit-q">{t.isTiebreak ? 'Tiebreak: the winner goes on to the duel' : `Who takes ${gift ? `the ${gift.name}` : 'it'}?`}</div>
      {duelGameLine}
      <div className="callit-corners">
        {corner(t.challengerId!, 'red')}
        <span className="callit-vs f6-disp fz-riso" aria-hidden>vs</span>
        {corner(t.defenderId!, 'green')}
      </div>
      {!t.isTiebreak && (
        <div className="callit-foot">
          {myVote ? <><Icon name="check" /> You voted for <strong>{nm(game, myVote)}</strong></>
            : picking ? <>Voting opens once the duel is picked.</>
            : canVote ? <>Tap a corner to back them. {voteStakeNote(cfg, player.chips)}</>
            : <>Voting closed. Watch the TV.</>}
        </div>
      )}
      {!t.isTiebreak && tally}
      <details className="howto">
        <summary>How duels work</summary>
        <div className="howto-body">The <strong>defender picks the duel</strong> (rock-paper-scissors, thumb war…). {spectatorLoserNote(cfg, t)}</div>
      </details>
    </Card>
  );
}

function ReprisalWatch({ t }: { t: Turn }) {
  const { game, me } = usePlayer();
  const who = nm(game, t.reprisalPlayerId);
  const target = Object.values(game.gifts || {}).some((g) => g.ownerId === me);
  return (
    <Card tone="active">
      <div className="status-icon"><Icon name="reprisal" /></div>
      <div className="status-title">Reindeer Reprisal!</div>
      <div className="status-desc">{who || 'Player'} is choosing a target...</div>
      {target && (
        <div className="help-box warning">
          <div className="help-title">⚠️ You could be targeted!</div>
          <div className="help-content">{who} can challenge <strong>your gift</strong>. If they do, you'll defend (for free)!</div>
        </div>
      )}
    </Card>
  );
}

function EndgameChoice({ t }: { t: Turn }) {
  const { game, me, act } = usePlayer();
  if (t.activePlayerId !== me) return <Simple icon="🎰" tone="active" title="Misfit Lottery" desc={`${nm(game, t.activePlayerId)} is choosing their path...`} />;
  const canSteal = stealTargets(game, me).length > 0;
  return (
    <Card tone="your-turn">
      <div className="status-icon"><Icon name="bid" /></div>
      <div className="status-title">Your Misfit Lottery Turn!</div>
      <div className="status-desc">Target: "{t.misfitName}"</div>
      <div className="help-box warning">
        <div className="help-title"><Icon name="target" /> Choose Your Path</div>
        <div className="help-content">
          <strong>Path A - Steal:</strong> Challenge any player's gift. Loser takes the misfit.<br />
          <strong>Path B - Auction:</strong> Let everyone bid on the misfit. You duel the highest bidder.
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
        {canSteal
          ? <button className="btn btn-primary" data-testid="path-a" onClick={() => act(D.chooseSteal)}><Icon name="target" /> Path A: Steal a Gift</button>
          : <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '0.5rem' }}>No gifts to steal yet</div>}
        <button className="btn btn-gold" data-testid="path-b" onClick={() => act(D.startAuction)}><Icon name="bid" /> Path B: Auction for Misfit</button>
      </div>
    </Card>
  );
}

function StealSelect({ t }: { t: Turn }) {
  const { game, me, player, cfg, act } = usePlayer();
  if (t.activePlayerId !== me) {
    return (
      <Card tone="active">
        <div className="status-icon"><Icon name="target" /></div>
        <div className="status-title">Steal in Progress!</div>
        <div className="status-desc">{nm(game, t.activePlayerId)} is choosing a gift to steal...</div>
        {player.giftId && <div className="help-box warning"><div className="help-title">⚠️ You could be targeted!</div><div className="help-content">They might challenge YOUR gift!</div></div>}
      </Card>
    );
  }
  return (
    <Card tone="your-turn">
      <div className="status-icon"><Icon name="target" /></div>
      <div className="status-title">Path A: Steal a Gift!</div>
      <div className="status-desc">Choose a gift to challenge. Loser takes "{t.misfitName}"</div>
      <div style={{ maxHeight: 300, overflowY: 'auto', margin: '1rem 0' }}>
        {stealTargets(game, me).map(([id, gift]) => {
          const cost = minCost(cfg, gift.naughtyLevel || 0);
          return <TargetOption key={id} testid={`steal-target-${id}`} title={gift.name} sub={`Owner: ${nm(game, gift.ownerId)}`}
            naughty={gift.naughtyLevel} cost={cost} afford={player.chips >= cost} onPick={() => act((g) => D.executeSteal(g, id))} />;
        })}
      </div>
      <button className="btn btn-secondary" data-testid="cancel-steal" onClick={() => act(D.cancelSteal)} style={{ width: '100%' }}>← Back to Path Choice</button>
    </Card>
  );
}

function NoBids({ t }: { t: Turn }) {
  const { game, me, act } = usePlayer();
  if (t.activePlayerId !== me) return <Simple icon="🎰" tone="active" title="No Bids!" desc={`${nm(game, t.activePlayerId)} is choosing a defender...`} />;
  const others = Object.entries(game.players || {}).filter(([id]) => id !== me);
  return (
    <Card tone="your-turn">
      <div className="status-icon"><Icon name="bid" /></div>
      <div className="status-title">No Bids! Choose Your Challenger</div>
      <div className="status-desc">Duel someone for "{t.misfitName}". You get it either way!</div>
      <div className="help-box"><div className="help-content"><strong>Win:</strong> Get misfit FREE!<br /><strong>Lose:</strong> Get misfit, but pay 1 chip toll to defender.</div></div>
      <div style={{ maxHeight: 250, overflowY: 'auto', margin: '1rem 0' }}>
        {others.map(([id, p]) => (
          <div key={id} className="action-option" data-testid={`nobids-defender-${id}`} onClick={() => act((g) => D.noBidsDuel(g, id))}
            style={{ background: 'var(--bg-card-hover)', padding: '1rem', borderRadius: 8, marginBottom: '0.5rem', cursor: 'pointer' }}>
            <div style={{ fontWeight: 600 }}><Icon name="duel" /> Challenge {p.name}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

function EndgameAuction({ t }: { t: Turn }) {
  const { game, me } = usePlayer();
  if (t.activePlayerId === me && t.phase === 'endgame_bidding') {
    return (
      <Card tone="your-turn">
        <div className="status-icon"><Icon name="bid" /></div>
        <div className="status-title">Auction in Progress!</div>
        <div className="status-desc">Others are bidding on "{t.misfitName}"</div>
        <div className="help-box"><div className="help-content">Wait for bids, then you'll duel the highest bidder (or choose a defender if no bids).</div></div>
      </Card>
    );
  }
  return <Simple icon="🎰" tone="active" title="Misfit Lottery" desc={`${nm(game, t.activePlayerId)}'s turn for "${t.misfitName}"`} />;
}
