// A runtime mini-game on the TV and the host console (core/minigames): the stage of whichever game
// is on. A new game's stage goes in STAGES (its phone card: player/GameScreen.tsx ARENA_CARDS).
import type { ComponentType } from 'react';
import { getDef } from '../core/minigames/registry';
import { useServerNow } from '../net/hooks';
import { Briefing } from '../ui/Briefing';
import '../ui/arena.css';
import { useHost } from './HostContext';
import { GremlinStage } from './GremlinStage';
import { RingTossStage } from './RingTossStage';
import { ConveyorStage } from './ConveyorStage';
import { RobotStage } from './RobotStage';
import { MechaStage } from './MechaStage';
import { TowerStage } from './TowerStage';

const STAGES: Record<string, ComponentType> = {
  pass_the_gremlin: GremlinStage,
  reindeer_ring_toss: RingTossStage,
  gift_conveyor: ConveyorStage,
  nutcracker_knockout: RobotStage,
  mecha_santa: MechaStage,
  elf_tower_wars: TowerStage,
};

export function ArenaStage() {
  const { game, tv, send } = useHost();
  const now = useServerNow(200);
  const av = game.arenaView;
  const id = av?.gameId;
  if (!av || !id) return null;
  const Stage = STAGES[id];
  // a game this page doesn't know (the room is newer than the page): say so, never another game's stage
  if (!Stage) return <p className="arena-unknown" role="status" data-testid="arena-unknown">This game needs a newer app. Reload this page.</p>;
  // before it starts: how to play and who is ready, over the game standing still behind it
  const def = getDef(id);
  return (
    <>
      {av.lifecycle === 'briefing' && def && (
        <Briefing game={game} av={av} label={def.label} emoji={def.emoji} rules={def.rules} now={now} where={tv ? 'tv' : 'console'}
          onStartNow={tv ? undefined : () => { void send({ type: 'arena_start_now' }); }} />
      )}
      <Stage />
    </>
  );
}
