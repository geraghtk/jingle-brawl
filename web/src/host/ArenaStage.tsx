// A runtime mini-game on the TV and the host console (core/minigames): the stage of whichever game
// is on. A new game's stage goes in STAGES (its phone card: player/GameScreen.tsx ARENA_CARDS).
import { useEffect } from 'react';
import type { ComponentType } from 'react';
import { getDef } from '../core/minigames/registry';
import { useServerNow } from '../net/hooks';
import { Briefing, GoFlash, PracticeRibbon, useGo } from '../ui/Briefing';
import { Ceremony } from '../ui/Ceremony';
import { play } from '../ui/sound';
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

/** These start with a sound of their own (a bell, a horn); the rest get the shared one. */
const OWN_START = new Set(['nutcracker_knockout', 'mecha_santa', 'elf_tower_wars']);

export function ArenaStage() {
  const { game, tv, send } = useHost();
  const now = useServerNow(200);
  const av = game.arenaView;
  const id = av?.gameId;
  const go = useGo(av?.lifecycle, av?.instanceId);
  useEffect(() => { if (go && tv && id && !OWN_START.has(id)) play('go'); }, [go, tv, id]);
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
      {go && <GoFlash />}
      {av.practice && av.lifecycle !== 'briefing' && <PracticeRibbon endsAt={av.lifecycle === 'active' ? av.practice.endsAt : undefined} now={now} />}
      <Ceremony game={game} av={av} now={now} where={tv ? 'tv' : 'console'} />
      <Stage />
    </>
  );
}
