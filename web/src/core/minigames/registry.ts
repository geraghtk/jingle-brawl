// The registered games. A new game registers a definition; the runtime needs no change.
import type { GameDefinition } from './contracts';
import { gremlin } from './gremlin/gremlin';
import { ringToss } from './ringtoss/ringtoss';
import { conveyor } from './conveyor/conveyor';
import { robots } from './robots/robots';

const defs = new Map<string, GameDefinition>();

export function register<C, S, A>(def: GameDefinition<C, S, A>): void {
  if (defs.has(def.id) && defs.get(def.id) !== (def as unknown as GameDefinition)) throw new Error(`game ${def.id} is already registered`);
  defs.set(def.id, def as unknown as GameDefinition);
}

/** Built in: Pass the Gremlin (game #1), Reindeer Ring Toss (game #2), the Great Gift Conveyor (game #3), Nutcracker Knockout (game #4). */
register(gremlin);
register(ringToss);
register(conveyor);
register(robots);

export const getDef = (id: string | undefined | null): GameDefinition | undefined => (id ? defs.get(id) : undefined);
export const registered = (): GameDefinition[] => [...defs.values()];
