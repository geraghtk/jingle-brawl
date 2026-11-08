// AI Draw Off, an in-app duel game: the two duelists draw the same prompt on their phones, the
// room watches on the TV, and the AI Art Critic scores both drawings; the higher score wins the
// duel. The critic is called by the room server (with the room's own key, never stored); this
// module is the pure part. If the critic can't decide (no key, an error, a tie), nobody wins by
// default: the duel goes back to the host, who picks the winner (marked as a host decision).
import type { Rng } from './rng';
import { pick } from './rng';
import { refuse, Tx } from './tx';
import type { Game, Plan, PlayerId } from './types';

export const PROMPT_REVEAL_MS = 6 * 1000;
export const DRAW_OFF_MS = 90 * 1000;
export const RESULTS_MS = 15 * 1000;
export const DRAW_GRACE_MS = 1500;       // a drawing sent at the bell still lands

export const DRAW_OFF_PROMPTS = [
  'A young cow eating grass and admiring the view. Bonus points for adding the farmer.',
  'A sleepy cat curled up on a windowsill watching it rain.',
  'A robot making a sandwich, looking confused about the lettuce.',
  'A penguin wearing a tiny top hat, standing next to an iceberg.',
  'A dragon accidentally setting its own birthday cake on fire.',
  'A squirrel having a heated argument with a bird over an acorn.',
  'A chef slipping on a banana peel while carrying a giant pizza.',
  'A wizard casting a spell that goes hilariously wrong.',
  'A unicorn at a disco, showing off its dance moves.',
  'A detective examining footprints that lead in circles.',
  'A cat pretending to read a newspaper to impress other cats.',
  'A bear trying to fit into a too-small kayak.',
  'An astronaut waving at Earth while floating near the moon.',
  'A pirate ship with a very nervous parrot on the shoulder.',
  'A monster under the bed who is actually scared of the child.',
  'A snowman slowly melting in the sun, looking worried.',
  'A fish wearing tiny sunglasses, lounging in a coral chair.',
  'A snail winning a race by crossing the finish line first.',
  'A giraffe trying to use a vending machine.',
  'A group of ghosts having a tea party in a haunted mansion.',
  'A reindeer trying on Christmas sweaters in front of a mirror.',
  'An elf buried under a mountain of wrapping paper, waving for help.',
  'Santa stuck in a chimney while a cat watches, unimpressed.',
  'A snowman building a tiny person out of snow.',
  'A penguin and a polar bear arguing over the last cookie.',
  'A family photo where the dog is the only one smiling.',
  'A Christmas tree tiptoeing out of the front door.',
  'A gift box with legs sneaking away from the party.',
  'The Grinch secretly enjoying a mug of hot cocoa.',
  'Grandpa asleep in an armchair, wearing a paper crown.',
  'A gingerbread man running from a very hungry fox.',
  'A sleigh with a flat tyre and three annoyed reindeer.',
  'A stocking so full it has burst at the seams.',
  'An elf on roller skates delivering a stack of presents.',
  'A snowball fight where one side brought a catapult.',
  'A cat sitting in the empty box, ignoring the present.',
  'A turkey in sunglasses escaping on a sled.',
  'Carolers singing to a house where everyone is hiding.',
  'A reindeer at the gym lifting candy cane dumbbells.',
  'A scented candle so strong the houseplants are fainting.',
  'Mrs Claus fixing the sleigh engine while Santa reads the manual.',
  'An octopus wrapping eight presents at once.',
  'A hedgehog trying to hug a balloon.',
  'A child unwrapping a pair of socks, reacting dramatically.',
  'A dog that ate the tinsel and looks very proud.',
  'A snowman in a sauna, deeply regretting it.',
  'Two elves fighting over the last roll of tape.',
  'A goldfish wearing a tiny Santa hat in its bowl.',
  'A lighthouse covered in far too many fairy lights.',
  'A gift exchange where everyone brought the same mug.'
];

export interface DrawOffScore { rating: number; description?: string; critique?: string; backend?: string; model?: string }

export interface DrawOff {
  type: 'draw_off';
  phase: 'prompt_reveal' | 'drawing' | 'judging' | 'results';
  prompt: string;
  participants: PlayerId[];                // the two duelists
  submissions?: Record<PlayerId, string>;
  scores?: Record<PlayerId, DrawOffScore>;
  rankings?: PlayerId[];
  winners?: PlayerId[];                    // one winner, or none when the critic couldn't decide
  failed?: string | null;                  // why the critic couldn't decide
  wittyVerdict?: string | null;
  promptRevealEndTime: number;
  drawingStartTime?: number;
  drawingEndTime?: number;
  resultsEndTime?: number;
  startedAt: number;
  claims?: Record<string, true>;
}

export const isDrawOff = (mg: unknown): mg is DrawOff => !!mg && (mg as DrawOff).type === 'draw_off';
const drawOffOf = (game: Game) => (isDrawOff(game.minigame) ? game.minigame : undefined);

/** Start it (inside the duel's own transaction): the turn waits in minigame_draw_off. */
export function startDrawOff(tx: Tx, duelists: [PlayerId, PlayerId], ctx: { now: number; rng: Rng }) {
  const prompt = pick(DRAW_OFF_PROMPTS, ctx.rng);
  const mg: DrawOff = { type: 'draw_off', prompt, participants: duelists, phase: 'prompt_reveal', promptRevealEndTime: ctx.now + PROMPT_REVEAL_MS, startedAt: ctx.now };
  tx.set('minigame', mg).set('currentTurn/phase', 'minigame_draw_off')
    .event(`🎨 AI Draw Off: draw "${prompt}"!`, 'duel');
}

export function startDrawing(game: Game, now: number): Plan {
  const mg = drawOffOf(game);
  if (mg?.phase !== 'prompt_reveal') return refuse('Not in the prompt reveal');
  return new Tx(game).claim('minigame/claims/prompt_reveal', true)
    .set('minigame/phase', 'drawing').set('minigame/drawingStartTime', now).set('minigame/drawingEndTime', now + DRAW_OFF_MS)
    .event(`✏️ Drawing time! ${DRAW_OFF_MS / 1000} seconds to draw "${mg.prompt}"`, 'duel').plan();
}

/** Both drawings are in (or time is up). */
export const everyoneDrew = (mg: DrawOff) => mg.participants.every((p) => mg.submissions?.[p]);

export function startJudging(game: Game): Plan {
  const mg = drawOffOf(game);
  if (mg?.phase !== 'drawing') return refuse('Not drawing');
  return new Tx(game).claim('minigame/claims/drawing', true).set('minigame/phase', 'judging')
    .event('🤖 The AI Art Critic is judging...', 'duel').plan();
}

/** The critic's scores: the higher one wins; a tie or a failure leaves it to the host. */
export function judged(game: Game, result: { scores: Record<PlayerId, DrawOffScore>; verdict?: string | null } | { failed: string }, now: number): Plan {
  const mg = drawOffOf(game);
  if (mg?.phase !== 'judging') return refuse('Not judging');
  const tx = new Tx(game).claim('minigame/claims/judging', true).set('minigame/phase', 'results').set('minigame/resultsEndTime', now + RESULTS_MS);
  if ('failed' in result) {
    return tx.set('minigame/winners', []).set('minigame/failed', result.failed)
      .event(`🤖 The critic couldn't judge (${result.failed}). The host picks the winner.`, 'duel').plan();
  }
  const rankings = [...mg.participants].sort((a, b) => (result.scores[b]?.rating || 0) - (result.scores[a]?.rating || 0));
  const [first, second] = rankings.map((p) => result.scores[p]?.rating || 0);
  const winners = first > second ? [rankings[0]] : [];
  tx.set('minigame/scores', result.scores).set('minigame/rankings', rankings).set('minigame/winners', winners)
    .set('minigame/wittyVerdict', result.verdict ?? null).set('minigame/failed', winners.length ? null : `a tie at ${first}`);
  if (result.verdict && winners.length) tx.event(`🎭 Art Critic: ${result.verdict}`, 'duel');
  else if (!winners.length) tx.event(`🤖 A tie at ${first}! The host picks the winner.`, 'duel');
  return tx.plan();
}

/** Back to the duel: with the critic's winner (resolved like any duel game), or for the host to pick. */
export function completeDrawOff(game: Game): Plan {
  const mg = drawOffOf(game);
  const t = game.currentTurn;
  if (mg?.phase !== 'results' || !t) return refuse('No Draw Off results yet');
  const tx = new Tx(game).claim('minigame/claims/results', true).set('minigame', null).set('currentTurn/phase', 'duel');
  const winner = mg.winners?.[0];
  if (winner) tx.set('currentTurn/duelGameWinner', winner);
  else tx.set('currentTurn/duelGameOutcome', { status: 'no_contest', reason: mg.failed || 'the critic could not decide' });
  return tx.plan();
}
