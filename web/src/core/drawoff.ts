// AI Draw Off: tied bidders draw the same prompt for 3 minutes; the AI Art Critic scores
// every drawing (the host page calls the critic — that part isn't pure) and the best
// one(s) win the tie.
import type { Rng } from './rng';
import { pick } from './rng';
import { config } from './settings';
import { potAndTax, yieldPayment } from './rules';
import { refuse, Tx } from './tx';
import type { Game, Plan, PlayerId } from './types';

export const DRAW_OFF_MS = 3 * 60 * 1000;
export const PROMPT_REVEAL_MS = 10 * 1000;

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

export interface DrawOffScore { rating: number; description?: string; critique?: string; rawVisionResponse?: string; rawRateResponse?: string; backend?: string; model?: string }

export interface DrawOff {
  type: 'draw_off';
  phase: 'prompt_reveal' | 'drawing' | 'judging' | 'results';
  prompt: string;
  participants: PlayerId[];
  tiedBidders: PlayerId[];
  minigameGoal: 'pick_one' | 'pick_two';
  submissions?: Record<PlayerId, string>;
  scores?: Record<PlayerId, DrawOffScore>;
  rankings?: PlayerId[];
  winners?: PlayerId[];
  wittyVerdict?: string | null;
  verdictBackend?: string | null;
  verdictModel?: string | null;
  winnerSource?: string;
  promptRevealEndTime: number;
  drawingStartTime?: number;
  drawingEndTime?: number;
  startedAt: number;
}

export function startDrawOff(game: Game, ctx: { now: number; rng: Rng }): Plan {
  const t = game.currentTurn;
  const tied = t?.tiedBidders || [];
  if (tied.length < 2) return refuse('Not enough players for Draw Off');
  if (!game.settings?.aiCriticEnabled || !game.settings?.aiCriticUrl) {
    return refuse('AI Art Critic must be enabled for Draw Off. Enable it in game settings.');
  }
  const prompt = pick(DRAW_OFF_PROMPTS, ctx.rng);
  const mg: DrawOff = {
    type: 'draw_off', prompt, participants: tied, tiedBidders: tied, minigameGoal: t?.minigameGoal || 'pick_one',
    phase: 'prompt_reveal', promptRevealEndTime: ctx.now + PROMPT_REVEAL_MS, startedAt: ctx.now,
  };
  return new Tx(game).set('minigame', mg).set('currentTurn/phase', 'minigame_draw_off')
    .set('currentTurn/minigameStatus', `Draw Off: Draw "${prompt}" - 3 minutes!`)
    .event(`🎨 Starting AI Draw Off: Draw "${prompt}" - 3 minutes!`, 'bid').plan();
}

export function startDrawing(game: Game, now: number): Plan {
  const mg = game.minigame as DrawOff | undefined;
  if (mg?.type !== 'draw_off' || mg.phase !== 'prompt_reveal') return refuse('Not in the prompt reveal');
  return new Tx(game).claim('minigame/claims/prompt_reveal', true)
    .set('minigame/phase', 'drawing').set('minigame/drawingStartTime', now).set('minigame/drawingEndTime', now + DRAW_OFF_MS)
    .event(`✏️ Drawing time! 3 minutes to draw "${mg.prompt}"`, 'bid').plan();
}

export function startJudging(game: Game): Plan {
  const mg = game.minigame as DrawOff | undefined;
  if (mg?.type !== 'draw_off' || mg.phase !== 'drawing') return refuse('Not drawing');
  return new Tx(game).claim('minigame/claims/drawing', true).set('minigame/phase', 'judging')
    .event('🤖 AI Art Critic is judging all drawings...', 'bid').plan();
}

/** The critic's scores are in: rank, pick the winner(s) (the LLM's verdict if it named one). */
export function judged(game: Game, scores: Record<PlayerId, DrawOffScore>, verdict: { text?: string | null; winnerId?: string | null; backend?: string | null; model?: string | null }): Plan {
  const mg = game.minigame as DrawOff | undefined;
  if (mg?.type !== 'draw_off') return refuse('No Draw Off');
  const rankings = [...mg.participants].sort((a, b) => (scores[b]?.rating || 0) - (scores[a]?.rating || 0));
  const llm = verdict.winnerId && mg.participants.includes(verdict.winnerId) ? verdict.winnerId : null;
  const winners = mg.minigameGoal === 'pick_one'
    ? [llm || rankings[0]]
    : llm ? [llm, rankings.filter((p) => p !== llm)[0] || rankings[1]] : rankings.slice(0, 2);
  const tx = new Tx(game).set('minigame/phase', 'results').set('minigame/scores', scores).set('minigame/rankings', rankings)
    .set('minigame/wittyVerdict', verdict.text ?? null).set('minigame/verdictBackend', verdict.backend ?? null)
    .set('minigame/verdictModel', verdict.model ?? null).set('minigame/winnerSource', llm ? 'LLM verdict' : 'highest AI rating')
    .set('minigame/winners', winners);
  if (verdict.text) tx.event(`🎭 Art Critic: ${verdict.text}`, 'bid');
  return tx.plan();
}

/** "Continue to Duel": the winner duels the Opener, or the two winners duel after a yield. */
export function completeDrawOff(game: Game): Plan {
  const mg = game.minigame as DrawOff | undefined;
  const t = game.currentTurn;
  if (mg?.type !== 'draw_off' || mg.phase !== 'results' || !mg.winners?.length || !t) return refuse('Cannot proceed: no results or winners');
  const tx = new Tx(game).claim('minigame/claims/results', true).set('minigame', null);
  const cfg = config(game.settings);
  if ((t.minigameGoal || 'pick_one') === 'pick_one') {
    const c = mg.winners[0];
    const bid = t.bids?.[c]?.amount || 0;
    const { pot, tax } = potAndTax(cfg, bid);
    tx.add(`players/${c}/chips`, -bid).set('currentTurn/phase', 'duel').set('currentTurn/challengerId', c).set('currentTurn/defenderId', t.openerId)
      .set('currentTurn/pot', pot).set('currentTurn/tax', tax).set('currentTurn/isYieldDuel', false).set('currentTurn/isTiebreak', false)
      .event(`🎨 Draw Off complete! ${game.players?.[c]?.name} wins - duels ${game.players?.[t.openerId!]?.name}!`, 'bid');
  } else {
    if (mg.winners.length < 2) return refuse('Need 2 winners for yield duel');
    const [c1, c2] = mg.winners;
    const pay1 = yieldPayment(t.bids?.[c1]?.amount || 0);
    const pay2 = yieldPayment(t.bids?.[c2]?.amount || 0);
    const { pot, tax } = potAndTax(cfg, pay1 + pay2);
    tx.add(`players/${c1}/chips`, -pay1).add(`players/${c2}/chips`, -pay2)
      .set('currentTurn/phase', 'duel').set('currentTurn/challengerId', c1).set('currentTurn/defenderId', c2)
      .set('currentTurn/pot', pot).set('currentTurn/tax', tax).set('currentTurn/isYieldDuel', true).set('currentTurn/yieldingOpener', t.openerId)
      .event(`🎨 Draw Off complete! ${game.players?.[c1]?.name} & ${game.players?.[c2]?.name} selected to fight.`, 'bid');
  }
  return tx.plan();
}
