// The phone's game screen: header, the card for the current moment, my gift, the table.
import { useEffect, useState } from 'react';
import type { ComponentType } from 'react';
import { Avatar, avatarColor } from '../ui/avatar';
import { observe as mascotsObserve } from '../ui/mascots';
import { MvpReveal } from '../ui/MvpReveal';
import { BidPanel } from './BidPanel';
import { DrawOffCard } from './DrawOffCard';
import { GremlinCard } from './GremlinCard';
import { RingTossCard } from './RingTossCard';
import { ConveyorCard } from './ConveyorCard';
import { RobotCard } from './RobotCard';
import { PartyCard } from './PartyCard';
import { usePlayer } from './PlayerContext';
import { ReprisalCard } from './ReprisalCard';
import { StatusCard } from './StatusCard';
import { VerdictModal } from './VerdictModal';

/** The phone's card for each runtime mini-game (its stage on the TV: host/ArenaStage.tsx STAGES). */
const ARENA_CARDS: Record<string, ComponentType> = {
  pass_the_gremlin: GremlinCard,
  reindeer_ring_toss: RingTossCard,
  gift_conveyor: ConveyorCard,
  nutcracker_knockout: RobotCard,
};

export function GameScreen({ onLeave }: { onLeave: () => void }) {
  const { game, me, player, cfg } = usePlayer();
  const t = game.currentTurn;
  const [mvpSeen, setMvpSeen] = useState(false);

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
  const showStatus = !iBid && !myReprisal && !party && !drawOff && !arena;

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
      {arena && (ArenaCard ? <ArenaCard />
        // a game this page doesn't know (the room is newer than the page)
        : <div className="card" role="status" data-testid="arena-unknown"><p>This game needs a newer app. Reload this page to play.</p></div>)}

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
      {game.phase === 'ended' && !mvpSeen && <MvpReveal game={game} me={me} variant="phone" onClose={() => setMvpSeen(true)} />}
    </div>
  );
}
