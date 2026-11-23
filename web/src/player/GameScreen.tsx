// The phone's game screen: header, the card for the current moment, my gift, the table.
import { useEffect } from 'react';
import type { ComponentType } from 'react';
import { Avatar, avatarColor } from '../ui/avatar';
import { observe as mascotsObserve } from '../ui/mascots';
import { Recap } from '../ui/Recap';
import { BidPanel } from './BidPanel';
import { DrawOffCard } from './DrawOffCard';
import { GremlinCard } from './GremlinCard';
import { RingTossCard } from './RingTossCard';
import { ConveyorCard } from './ConveyorCard';
import { RobotCard } from './RobotCard';
import { MechaCard } from './MechaCard';
import { TowerCard } from './TowerCard';
import { PartyCard } from './PartyCard';
import { usePlayer } from './PlayerContext';
import { ReprisalCard } from './ReprisalCard';
import { StatusCard } from './StatusCard';
import { getDef } from '../core/minigames/registry';
import { useServerNow } from '../net/hooks';
import { Briefing, GoFlash, PracticeRibbon, useGo } from '../ui/Briefing';
import { Ceremony } from '../ui/Ceremony';
import { useWakeLock } from '../ui/wakeLock';
import { buzz } from '../ui/sound';
import { VerdictModal } from './VerdictModal';

/** The phone's card for each runtime mini-game (its stage on the TV: host/ArenaStage.tsx STAGES). */
const ARENA_CARDS: Record<string, ComponentType> = {
  pass_the_gremlin: GremlinCard,
  reindeer_ring_toss: RingTossCard,
  gift_conveyor: ConveyorCard,
  nutcracker_knockout: RobotCard,
  mecha_santa: MechaCard,
  elf_tower_wars: TowerCard,
};

/** How the mini-game ended, over the game's own card (it comes up a moment after the ending). */
function ArenaCeremony() {
  const { game, me } = usePlayer();
  const now = useServerNow(250);
  return game.arenaView ? <Ceremony game={game} av={game.arenaView} now={now} where="phone" me={me} /> : null;
}

/** A practice round is on: the phone says so too. */
function ArenaPractice() {
  const { game } = usePlayer();
  const now = useServerNow(500);
  const av = game.arenaView;
  return av?.practice && av.lifecycle !== 'briefing' ? <PracticeRibbon endsAt={av.lifecycle === 'active' ? av.practice.endsAt : undefined} now={now} /> : null;
}

/** Before a mini-game starts: how to play it and this phone's Ready button. */
function ArenaBriefing() {
  const { game, me, send } = usePlayer();
  const now = useServerNow(200);
  const av = game.arenaView!;
  const def = getDef(av.gameId);
  if (!def) return null;
  return <Briefing game={game} av={av} label={def.label} emoji={def.emoji} rules={def.rules} now={now} me={me} where="phone" onReady={() => { buzz(20); void send({ type: 'arena_ready' }, { quiet: true }); }} />;
}

export function GameScreen({ onLeave }: { onLeave: () => void }) {
  const { game, me, player, cfg } = usePlayer();
  const t = game.currentTurn;

  // Santa / elf / reindeer: only my own moments
  useEffect(() => { mascotsObserve(game, { me }); }, [game, me]);

  // Which card has the stage right now
  const iBid = t && ((t.phase === 'bidding' && t.openerId !== me) || (t.phase === 'endgame_bidding' && t.activePlayerId !== me));
  const myReprisal = t?.phase === 'reprisal' && t.reprisalPlayerId === me;
  const party = t?.phase === 'minigame_party';
  const drawOff = t?.phase === 'minigame_draw_off'; // AI Draw Off, a duel game
  const arenaGame = t?.phase === 'minigame_arena' ? game.arenaView?.gameId : undefined;
  const arena = !!arenaGame;
  const ArenaCard = arenaGame ? ARENA_CARDS[arenaGame] : undefined;
  const briefing = arena && game.arenaView?.lifecycle === 'briefing' && !!ArenaCard;   // the instructions, before the game's own card
  const showStatus = !iBid && !myReprisal && !party && !drawOff && !arena;
  const go = useGo(game.arenaView?.lifecycle, game.arenaView?.instanceId);
  useEffect(() => { if (go) buzz(30); }, [go]);
  useWakeLock(!!(party || drawOff || arena));          // a phone must not doze off in the middle of a mini-game

  const gift = player.giftId ? game.gifts?.[player.giftId] : undefined;
  const others = Object.entries(game.players || {}).filter(([id]) => id !== me);

  return (
    <div id="game-screen" className="game-screen">
      <div className="header">
        <div className="player-info">
          <Avatar name={player.name} avatar={player.avatar} size={40} />
          <div className="player-name" id="display-name">{player.name}</div>
        </div>
        <div className="chip-count" aria-label={`${player.chips} chips`}><span className="jb-chip" /><span id="chip-display">{player.chips}</span></div>
      </div>

      {showStatus && <StatusCard />}
      {iBid && <BidPanel key={`${t!.phase}|${t!.giftId || t!.misfitId}`} />}
      {myReprisal && <ReprisalCard />}
      {party && <PartyCard />}
      {drawOff && <DrawOffCard />}
      {arena && briefing ? <ArenaBriefing />
        : arena && (ArenaCard ? <ArenaCard />
        // a game this page doesn't know (the room is newer than the page)
        : <div className="card" role="status" data-testid="arena-unknown"><p>This game needs a newer app. Reload this page to play.</p></div>)}
      {arena && go && <GoFlash />}
      {arena && game.arenaView?.practice && <ArenaPractice />}
      {arena && game.arenaView?.result && <ArenaCeremony />}

      {(gift || game.phase !== 'lobby') && (
        <div className="table-rows">
          {gift && (
            <div className="table-row" id="your-gift-card">
              <span className="table-row-k">Your gift</span>
              <span className="f6-gift mini" aria-hidden style={{ ['--gift-color' as string]: avatarColor(gift.name) }}><span className="ribbon-v" /><span className="ribbon-h" /><span className="bow" /></span>
              <span className="table-row-v" id="your-gift-name">{gift.name}</span>
              <span className="naughty-dots" title={`Naughty ${gift.naughtyLevel ?? 0}`}>
                <span className="sr-only">Naughty </span><span id="your-gift-naughty" className="sr-only">{gift.naughtyLevel ?? 0}</span>
                <span aria-hidden>{'●'.repeat(Math.min(5, gift.naughtyLevel ?? 0))}{'○'.repeat(5 - Math.min(5, gift.naughtyLevel ?? 0))}</span>
              </span>
            </div>
          )}
          {game.phase !== 'lobby' && (
            <div className="table-row" id="players-card">
              <span className="table-row-k">The table</span>
              <div className="players-list" id="players-list">
                {others.length === 0 ? <span className="table-row-empty">Nobody else yet</span>
                  : others.map(([id, p]) => (
                    <span key={id} className={`table-seat ${p.giftId ? 'has-gift' : ''}`} title={`${p.name}: ${p.giftId ? 'has a gift' : 'in the bag'}`}>
                      <Avatar name={p.name} avatar={p.avatar} size={30} />
                      <span className="table-seat-name">{p.name}</span>
                      {cfg.publicChips && <span className="table-seat-chips f6-num">{p.chips}</span>}
                    </span>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}

      <button className="leave-link" onClick={onLeave}>Leave game</button>
      <VerdictModal />
      {game.phase === 'ended' && <Recap game={game} me={me} where="phone" />}
    </div>
  );
}
