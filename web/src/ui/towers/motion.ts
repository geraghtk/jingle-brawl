// Elf Tower Wars: the timing of what moves on the TV (host/TowerStage.tsx), as sums so it can be tested.
// Nothing here decides anything: a tower's shot is a loop the screen draws while the tower has a target,
// and its kick is read off the same loop, so the tower jerks back as each snowball leaves it.

/** How long between a tower's shots on the screen (a cannon is slower; a better tower is quicker). */
export const shotEvery = (heavy: boolean, power: number) => (heavy ? 760 : 460) / Math.sqrt(Math.max(0.01, power));
/** How far along its flight (0 to 1) the one shot in the air is; `i` puts each tower out of step with the others. */
export const shotPhase = (now: number, i: number, every: number) => (((now + i * 137) % every) + every) % every / every;

/** How much of a shot's flight the kick lasts. */
const KICK = 0.36;
/** A tower's kick, from the flight of its shot: 1 as the shot leaves, easing to 0, still for the rest. */
export function recoil(phase: number): number {
  if (!(phase >= 0) || phase >= KICK) return 0;
  const left = 1 - phase / KICK;
  return left * left;
}

/** A toy's own place in its walk (ms into a loop of `loopMs`), from its id: two toys sent together are
    not in step, and a pack of mice scurries rather than hopping as one. */
export function stagger(id: string, loopMs: number): number {
  let h = 2166136261;
  for (let k = 0; k < id.length; k++) h = Math.imul(h ^ id.charCodeAt(k), 16777619);
  return (h >>> 0) % Math.max(1, Math.round(loopMs));
}
