// A runtime mini-game on the TV and the host console (core/minigames): the stage of whichever game
// is on. A new game's stage goes in STAGES (its phone card: player/GameScreen.tsx ARENA_CARDS).
import type { ComponentType } from 'react';
import '../ui/arena.css';
import { useHost } from './HostContext';
import { GremlinStage } from './GremlinStage';
import { RingTossStage } from './RingTossStage';
import { ConveyorStage } from './ConveyorStage';

const STAGES: Record<string, ComponentType> = {
  pass_the_gremlin: GremlinStage,
  reindeer_ring_toss: RingTossStage,
  gift_conveyor: ConveyorStage,
};

export function ArenaStage() {
  const { game } = useHost();
  const id = game.arenaView?.gameId;
  if (!id) return null;
  const Stage = STAGES[id];
  // a game this page doesn't know (the room is newer than the page): say so, never another game's stage
  return Stage ? <Stage /> : <p className="arena-unknown" role="status" data-testid="arena-unknown">This game needs a newer app. Reload this page.</p>;
}
