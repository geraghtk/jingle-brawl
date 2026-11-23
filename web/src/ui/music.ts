// Background music for the big screens (the TV, the host's console; never a phone: twelve phones out of
// step would be a racket). Made on the spot from the same small instruments as the sound effects
// (`VOICES` in sound.ts): no files, no network. Quieter than the effects, on a gain of its own.
//
// Three layers, kept apart:
//   1. the tunes (`TRACKS`): data. Our own, written for the game; no carol, nothing borrowed.
//   2. the score (`trackNotes`, `trackScore`, `trackFor`): pure, checked without a browser (music.test.ts).
//   3. the player (`setTrack`, `setMusicOn`): a look-ahead scheduler on the AudioContext's own clock.
import type { Game } from '../core/types';
import { getRig, muted, VOICES } from './sound';
import type { Note, Rig, Voice } from './sound';

export type TrackName = 'lobby' | 'game' | 'tense' | 'results';
export const TRACK_NAMES: TrackName[] = ['lobby', 'game', 'tense', 'results'];

// ---- 1. the tunes ---------------------------------------------------------------------------------
// A bar is a string of tokens, `pitch:beats`: `E5:1` a note, `E4+G4+C5:1` several at once, `-:2` a rest.
// Every bar is four beats. `chords` names each bar's harmony (two names = half a bar each); the test
// holds every bass and harmony note, and every melody note of a beat or longer, to it (shorter melody
// notes may pass between chord notes).
interface Part {
  voice: Voice;
  level: number;
  /** How long each note sounds, in seconds (a bell rings on past its beat); else `gate` of its own length. */
  ring?: number;
  gate?: number;
  bars: string[];
}
interface Track { bpm: number; chords: string[]; parts: Record<string, Part> }

const BEATS = 4;
const rep = (n: number, bar: string): string[] => Array.from({ length: n }, () => bar);
const OFF_EIGHTHS = '-:.5 G7:.5 -:.5 G7:.5 -:.5 G7:.5 -:.5 G7:.5';   // a sleigh bell between the beats
/** A chord on beats two and four. */
const pah = (a: string, b = a) => `-:1 ${a}:1 -:1 ${b}:1`;
/** A chord between every beat. */
const chug = (a: string, b = a) => `-:.5 ${a}:.5 -:.5 ${a}:.5 -:.5 ${b}:.5 -:.5 ${b}:.5`;
/** A bass bouncing between a note and its octave, in eighths. */
const bounce = (a: string, b = a) => { const up = (p: string) => p.slice(0, -1) + (Number(p.slice(-1)) + 1); return `${a}:.5 ${up(a)}:.5 ${a}:.5 ${up(a)}:.5 ${b}:.5 ${up(b)}:.5 ${b}:.5 ${up(b)}:.5`; };

const TRACKS: Record<TrackName, Track> = {
  // The lobby: a music box in C, unhurried. Sixteen bars, two eight-bar phrases.
  lobby: {
    bpm: 90,
    chords: ['C', 'Am', 'F', 'G', 'Em', 'Am', 'Dm G', 'C G', 'F', 'C', 'Dm', 'G', 'F', 'Em', 'Dm G', 'C G'],
    parts: {
      lead: { voice: 'chime', level: 0.6, ring: 1.4, bars: [
        'E5:1 G5:1 C6:1.5 G5:.5', 'A5:1.5 G5:.5 E5:1 C5:1', 'F5:1 A5:1 C6:1 A5:1', 'B5:1.5 A5:.5 G5:2',
        'G5:1 B5:1 E6:1.5 D6:.5', 'C6:1 A5:1 E5:1 A5:1', 'F5:1 A5:1 G5:1 B5:1', 'C6:2 G5:1 -:1',
        'A5:1.5 C6:.5 F6:1 C6:1', 'E6:1 C6:1 G5:2', 'F5:1 A5:1 D6:1.5 C6:.5', 'B5:1 G5:1 D5:2',
        'C6:1 A5:1 F5:1 A5:1', 'G5:1.5 A5:.5 B5:1 E5:1', 'D5:1 F5:1 G5:1 D6:1', 'C6:1 E5:1 G5:2',
      ] },
      bass: { voice: 'pluck', level: 0.55, gate: 0.95, bars: [
        'C3:2 G3:2', 'A2:2 E3:2', 'F2:2 C3:2', 'G2:2 D3:2', 'E2:2 B2:2', 'A2:2 E3:2', 'D3:2 G2:2', 'C3:2 G2:1 B2:1',
        'F2:2 C3:2', 'E3:2 C3:2', 'D3:2 A2:2', 'G2:2 D3:2', 'F2:2 C3:2', 'E2:2 B2:2', 'D3:2 G2:2', 'C3:2 G2:1 B2:1',
      ] },
      harmony: { voice: 'mallet', level: 0.28, ring: 0.5, bars: [
        pah('E4+G4+C5'), pah('E4+A4+C5'), pah('F4+A4+C5'), pah('D4+G4+B4'), pah('E4+G4+B4'), pah('E4+A4+C5'), pah('F4+A4+D5', 'D4+G4+B4'), pah('E4+G4+C5', 'D4+G4+B4'),
        pah('F4+A4+C5'), pah('E4+G4+C5'), pah('F4+A4+D5'), pah('D4+G4+B4'), pah('F4+A4+C5'), pah('E4+G4+B4'), pah('F4+A4+D5', 'D4+G4+B4'), pah('E4+G4+C5', 'D4+G4+B4'),
      ] },
      bells: { voice: 'tin', level: 0.16, ring: 0.09, bars: rep(16, OFF_EIGHTHS) },
    },
  },
  // Any mini-game: a toy xylophone in G, quick and bright. Sixteen bars.
  game: {
    bpm: 126,
    chords: ['G', 'Em', 'C', 'D', 'G', 'Em', 'C D', 'G', 'C', 'G', 'Am', 'D7', 'C', 'G', 'Am D', 'G D'],
    parts: {
      lead: { voice: 'mallet', level: 0.8, ring: 0.32, bars: [
        'D5:.5 G5:.5 B5:.5 G5:.5 D6:1 B5:1', 'E5:.5 G5:.5 B5:.5 G5:.5 E6:1 B5:1', 'E6:.5 D6:.5 C6:.5 G5:.5 E5:1 G5:1', 'A5:.5 D6:.5 F#6:.5 D6:.5 A5:1 D5:1',
        'D5:.5 G5:.5 B5:.5 G5:.5 D6:1 G6:1', 'G6:.5 F#6:.5 E6:.5 B5:.5 G5:1 B5:1', 'C6:.5 E6:.5 C6:.5 G5:.5 A5:.5 D6:.5 F#5:.5 A5:.5', 'G5:1 B5:.5 D6:.5 G5:1 -:1',
        'E5:.5 G5:.5 C6:.5 G5:.5 E6:1.5 D6:.5', 'B5:.5 D6:.5 B5:.5 G5:.5 D5:1 G5:1', 'A5:.5 C6:.5 E6:.5 C6:.5 A5:1 C6:1', 'D6:.5 C6:.5 A5:.5 F#5:.5 D5:1 A5:1',
        'G5:.5 C6:.5 E6:.5 C6:.5 G6:1 E6:1', 'D6:.5 B5:.5 G5:.5 B5:.5 D6:1 B5:1', 'A5:.5 C6:.5 E6:.5 C6:.5 D6:.5 A5:.5 F#5:.5 A5:.5', 'G5:1 B5:1 A5:1 D5:1',
      ] },
      sparkle: { voice: 'chime', level: 0.35, ring: 1.6, bars: [
        'G6:4', '-:4', 'E6:4', '-:4', 'G6:4', '-:4', 'E6:4', '-:4', 'E6:4', '-:4', 'E6:4', '-:4', 'G6:4', '-:4', 'E6:4', '-:4',
      ] },
      bass: { voice: 'pluck', level: 0.5, gate: 0.9, bars: [
        'G2:1 D3:1 G2:1 D3:1', 'E2:1 B2:1 E2:1 B2:1', 'C3:1 G2:1 C3:1 G2:1', 'D3:1 A2:1 D3:1 A2:1',
        'G2:1 D3:1 G2:1 D3:1', 'E2:1 B2:1 E2:1 B2:1', 'C3:1 G2:1 D3:1 A2:1', 'G2:1 D3:1 G2:1 B2:1',
        'C3:1 G2:1 C3:1 G2:1', 'G2:1 D3:1 G2:1 D3:1', 'A2:1 E3:1 A2:1 E3:1', 'D3:1 A2:1 D3:1 A2:1',
        'C3:1 G2:1 C3:1 G2:1', 'B2:1 D3:1 G2:1 D3:1', 'A2:1 E3:1 D3:1 A2:1', 'G2:1 B2:1 D3:1 F#2:1',
      ] },
      harmony: { voice: 'mallet', level: 0.3, ring: 0.2, bars: [
        chug('B3+D4'), chug('B3+E4'), chug('C4+E4'), chug('A3+D4'), chug('B3+D4'), chug('B3+E4'), chug('C4+E4', 'A3+D4'), chug('B3+D4'),
        chug('C4+E4'), chug('B3+D4'), chug('C4+E4'), chug('A3+D4'), chug('C4+E4'), chug('B3+D4'), chug('C4+E4', 'A3+D4'), chug('B3+D4', 'A3+D4'),
      ] },
      drum: { voice: 'drum', level: 0.45, ring: 0.16, bars: rep(16, 'B1:2 B1:2') },
      bells: { voice: 'tin', level: 0.2, ring: 0.08, bars: rep(16, '-:1 G7:1 -:1 G7:.5 G7:.5') },
    },
  },
  // The last seconds, a final: A minor, driving, a clock ticking in the tin. Eight bars.
  tense: {
    bpm: 138,
    chords: ['Am', 'Am', 'F', 'E7', 'Am', 'Dm', 'F E7', 'Am E7'],
    parts: {
      lead: { voice: 'mallet', level: 0.75, ring: 0.28, bars: [
        'A5:.5 C6:.5 E6:1 A5:.5 C6:.5 E6:1', 'A5:.5 C6:.5 E6:.5 D6:.5 C6:.5 B5:.5 A5:1', 'A5:.5 C6:.5 F6:1 A5:.5 C6:.5 F6:1', 'G#5:.5 B5:.5 E6:.5 D6:.5 B5:.5 G#5:.5 E5:1',
        'E6:.5 C6:.5 A5:1 E6:.5 C6:.5 A5:1', 'D6:.5 F6:.5 A6:1 D6:.5 F6:.5 A6:1', 'F6:.5 E6:.5 C6:.5 A5:.5 G#5:.5 B5:.5 D6:.5 E6:.5', 'A5:1 E5:1 E5:.5 G#5:.5 B5:.5 E6:.5',
      ] },
      bass: { voice: 'pluck', level: 0.5, gate: 0.9, bars: [
        bounce('A2'), bounce('A2'), bounce('F2'), bounce('E2'), bounce('A2'), bounce('D2'), bounce('F2', 'E2'), bounce('A2', 'E2'),
      ] },
      harmony: { voice: 'mallet', level: 0.3, ring: 0.25, bars: [
        pah('A3+C4+E4'), pah('A3+C4+E4'), pah('A3+C4+F4'), pah('G#3+B3+E4'), pah('A3+C4+E4'), pah('A3+D4+F4'), pah('A3+C4+F4', 'G#3+B3+E4'), pah('A3+C4+E4', 'G#3+B3+E4'),
      ] },
      drum: { voice: 'drum', level: 0.4, ring: 0.14, bars: rep(8, 'B1:1 B1:1 B1:1 B1:1') },
      tick: { voice: 'tin', level: 0.12, ring: 0.05, bars: rep(8, 'B7:.5 B7:.5 B7:.5 B7:.5 B7:.5 B7:.5 B7:.5 B7:.5') },
    },
  },
  // A result is up, the party is over: bells in F, pleased with themselves. Eight bars.
  results: {
    bpm: 112,
    chords: ['F', 'Bb', 'C', 'F', 'Dm', 'Bb', 'C', 'F C'],
    parts: {
      lead: { voice: 'chime', level: 0.55, ring: 1.0, bars: [
        'C5:.5 F5:.5 A5:1 C6:1 A5:1', 'D6:1 Bb5:.5 D6:.5 F6:1.5 D6:.5', 'E6:1 C6:.5 E6:.5 G5:1 C6:1', 'A5:.5 C6:.5 F6:1 C6:1 A5:1',
        'D6:1 A5:.5 F5:.5 D5:1 F5:1', 'F5:.5 Bb5:.5 D6:1 C6:.5 Bb5:.5 D6:1', 'E6:.5 D6:.5 C6:.5 G5:.5 E5:1 G5:1', 'F5:1 A5:.5 C6:.5 G5:1 E5:1',
      ] },
      bass: { voice: 'pluck', level: 0.5, gate: 0.9, bars: [
        'F2:1 C3:1 F2:1 C3:1', 'Bb2:1 F3:1 Bb2:1 F3:1', 'C3:1 G2:1 C3:1 G2:1', 'F2:1 C3:1 F2:1 A2:1',
        'D3:1 A2:1 D3:1 A2:1', 'Bb2:1 F2:1 Bb2:1 F2:1', 'C3:1 G2:1 C3:1 G2:1', 'F2:1 C3:1 C3:1 E2:1',
      ] },
      harmony: { voice: 'mallet', level: 0.28, ring: 0.35, bars: [
        pah('A3+C4+F4'), pah('Bb3+D4+F4'), pah('G3+C4+E4'), pah('A3+C4+F4'), pah('A3+D4+F4'), pah('Bb3+D4+F4'), pah('G3+C4+E4'), pah('A3+C4+F4', 'G3+C4+E4'),
      ] },
      drum: { voice: 'drum', level: 0.38, ring: 0.16, bars: rep(8, 'B1:2 B1:2') },
      bells: { voice: 'tin', level: 0.16, ring: 0.09, bars: rep(8, OFF_EIGHTHS) },
    },
  },
};

// ---- 2. the score (pure) --------------------------------------------------------------------------
const SEMITONE: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

/** A note name (`A4`, `F#5`, `Bb3`) as a MIDI number (A4 = 69). */
export function midiOf(name: string): number {
  const m = /^([A-G])([#b]?)(\d)$/.exec(name);
  if (!m) throw new Error(`music: not a note: "${name}"`);
  return SEMITONE[m[1]] + (m[2] === '#' ? 1 : m[2] === 'b' ? -1 : 0) + (Number(m[3]) + 1) * 12;
}
const hz = (midi: number) => 440 * 2 ** ((midi - 69) / 12);

/** One written note: where in the bar it starts and how long it is (in beats), and its pitch. */
export interface Written { beat: number; beats: number; midi: number }
export interface Score { bpm: number; bars: number; beatsPerBar: number; chords: string[]; parts: Record<string, Written[][]> }

function readBar(bar: string, where: string): Written[] {
  const out: Written[] = [];
  let beat = 0;
  for (const token of bar.trim().split(/\s+/)) {
    const [pitches, len] = token.split(':');
    const beats = Number(len);
    if (!(beats > 0)) throw new Error(`music: ${where}: no length in "${token}"`);
    if (pitches !== '-') for (const p of pitches.split('+')) out.push({ beat, beats, midi: midiOf(p) });
    beat += beats;
  }
  if (Math.abs(beat - BEATS) > 1e-9) throw new Error(`music: ${where} is ${beat} beats long, not ${BEATS}`);
  return out;
}

const scores: Partial<Record<TrackName, Score>> = {};
/** A track as written: every part, bar by bar. */
export function trackScore(name: TrackName): Score {
  const known = scores[name];
  if (known) return known;
  const t = TRACKS[name];
  const parts: Score['parts'] = {};
  for (const [part, p] of Object.entries(t.parts)) {
    if (p.bars.length !== t.chords.length) throw new Error(`music: ${name}/${part} has ${p.bars.length} bars, the track ${t.chords.length}`);
    parts[part] = p.bars.map((bar, i) => readBar(bar, `${name}/${part} bar ${i + 1}`));
  }
  return (scores[name] = { bpm: t.bpm, bars: t.chords.length, beatsPerBar: BEATS, chords: t.chords, parts });
}

/** How many bars before a track comes round again, and how long one bar lasts (s). */
export const trackBars = (name: TrackName): number => TRACKS[name].chords.length;
export const barSeconds = (name: TrackName): number => (BEATS * 60) / TRACKS[name].bpm;

const played: Partial<Record<TrackName, Note[][]>> = {};
/** The notes of one bar (it wraps: bar 16 of a 16-bar track is bar 0), in order, each start in seconds from the bar's. */
export function trackNotes(name: TrackName, bar: number): Note[] {
  let bars = played[name];
  if (!bars) {
    const score = trackScore(name), beat = 60 / score.bpm;
    bars = Array.from({ length: score.bars }, (_, i) => {
      const notes: Note[] = [];
      for (const [part, p] of Object.entries(TRACKS[name].parts)) {
        for (const w of score.parts[part][i]) {
          const lean = Number.isInteger(w.beat) ? 1 : 0.88;            // a little weight on the beat
          notes.push([p.voice, hz(w.midi), w.beat * beat, p.ring ?? Math.max(0.12, w.beats * beat * (p.gate ?? 0.9)), Math.min(1, p.level * lean)]);
        }
      }
      return notes.sort((a, b) => a[2] - b[2]);
    });
    played[name] = bars;
  }
  const n = bars.length;
  return bars[((Math.floor(bar) % n) + n) % n];
}

/** A game's clock is this close to the end: the music tightens. */
export const TENSE_MS = 8000;

/** Which music suits what this screen shows (null: none, people are talking). `now`: the room's clock,
 *  to hear the last seconds coming; without it a game never turns tense on time alone. */
export function trackFor(game: Pick<Game, 'phase' | 'currentTurn' | 'arenaView' | 'minigame'>, now?: number): TrackName | null {
  const soon = (end: unknown) => now != null && typeof end === 'number' && end - now <= TENSE_MS;
  const turn = game.currentTurn?.phase;
  if (turn === 'minigame_arena' && game.arenaView) {
    const av = game.arenaView;
    if (av.result || av.lifecycle === 'completed') return 'results';
    if (av.lifecycle === 'active') {
      // every game's view is its own; these are the fields the games share (read defensively)
      const v = (av.view || {}) as { stage?: unknown; round?: unknown; sudden?: unknown; endsAt?: unknown; stageEndsAt?: unknown };
      if (v.stage === 'overtime' || v.stage === 'tiebreak' || v.stage === 'sudden_death_intro' || v.round === 'sudden_death' || v.sudden === true) return 'tense';
      if (soon(v.endsAt) || ((v.stage === 'throwing' || v.stage === 'fighting') && soon(v.stageEndsAt))) return 'tense';
      return 'game';
    }
    if (av.lifecycle !== 'cancelled') return 'lobby';      // the instructions, a pause: keep it gentle
  }
  if (turn === 'minigame_party' || turn === 'minigame_draw_off') {
    const mg = (game.minigame || {}) as { phase?: unknown; phaseEndsAt?: unknown; drawingEndTime?: unknown };
    if (mg.phase === 'intro' || mg.phase === 'prompt_reveal') return 'lobby';
    if (mg.phase === 'results') return 'results';
    if (mg.phase === 'judging') return 'tense';             // the critic is making up its mind
    if (mg.phase === 'playing' && soon(mg.phaseEndsAt)) return 'tense';
    if (mg.phase === 'drawing' && soon(mg.drawingEndTime)) return 'tense';
    return 'game';
  }
  if (game.phase === 'lobby') return 'lobby';
  if (game.phase === 'ended') return 'results';
  return null;
}

// ---- the choice (per device, remembered) ----------------------------------------------------------
const KEY = 'jb_music';
let fallback = true;

/** What a screen plays before anyone has chosen: the TV does, the console does not (both open in one
 *  room would double it). */
export function setMusicDefault(on: boolean): void { fallback = on; }

export function musicOn(): boolean {
  try { const v = localStorage.getItem(KEY); return v === '1' ? true : v === '0' ? false : fallback; } catch { return fallback; }
}
export function setMusicOn(on: boolean): void {
  try { localStorage.setItem(KEY, on ? '1' : '0'); } catch { /* private mode: the choice just isn't remembered */ }
  audible = on && !muted();
  if (wanted) start();
}

// ---- 3. the player --------------------------------------------------------------------------------
const LEVEL = 0.3;            // under the effects (about a third of them)
const TICK_MS = 50;           // how often the scheduler looks
const AHEAD = 0.25;           // how far ahead it writes notes (s)
const AHEAD_HIDDEN = 1.6;     // a background tab's timers fire about once a second
const FADE = 0.6;             // one track into the next (s)
const FADE_OFF = 0.8;         // into silence (s)

/** One track playing: its own faders, where its bar 0 started, how far it has been written. */
interface Deck {
  name: TrackName; rig: Rig; faders: GainNode[]; nodes: GainNode[];
  t0: number; bar: number; idx: number; until: number; endAt: number | null;
}

let wanted: TrackName | null = null;
let audible = true;
let live: Deck | null = null;
let dying: Deck[] = [];
let bus: { dry: GainNode; wet: GainNode } | null = null;
let timer: ReturnType<typeof setInterval> | null = null;
let ticks = 0;
let touched = false;
let listening = false;

const WAKERS = ['pointerdown', 'pointerup', 'keydown', 'touchend', 'click'] as const;

/** Has this page been touched? Browsers keep audio shut until it has; we don't knock before. */
function unlocked(): boolean {
  return touched || !!(navigator as Navigator & { userActivation?: { hasBeenActive?: boolean } }).userActivation?.hasBeenActive;
}
function wake(): void {
  touched = true;
  try {
    if (!wanted || !audible) return;
    const r = getRig();
    if (!r) return;
    if (r.ctx.state === 'suspended') void r.ctx.resume().catch(() => {});
    else if (r.ctx.state === 'running') listen(false);
  } catch { /* no music is fine */ }
}
function listen(on: boolean): void {
  if (on === listening) return;
  listening = on;
  for (const type of WAKERS) (on ? window.addEventListener : window.removeEventListener).call(window, type, wake, true);
}

/** Music is on and wanted, but the page has not been touched yet, so the browser won't let it sound. */
export function musicBlocked(): boolean {
  if (typeof window === 'undefined') return false;
  return !!wanted && audible && !unlocked();
}

function makeDeck(r: Rig, name: TrackName, t0: number): Deck {
  const { ctx } = r;
  if (!bus) {
    bus = { dry: ctx.createGain(), wet: ctx.createGain() };
    bus.dry.gain.value = LEVEL; bus.dry.connect(r.out);
    bus.wet.gain.value = LEVEL; bus.wet.connect(r.wet);
  }
  // the instruments play into `dry` and `send` (which fade in); `dryOut` and `sendOut` fade it out again
  const dry = ctx.createGain(), send = ctx.createGain(), dryOut = ctx.createGain(), sendOut = ctx.createGain();
  for (const g of [dry, send]) { g.gain.setValueAtTime(0.25, t0); g.gain.linearRampToValueAtTime(1, t0 + FADE); }
  dry.connect(dryOut).connect(bus.dry);
  send.connect(sendOut).connect(bus.wet);
  return { name, rig: { ctx, out: dry, wet: send, noise: r.noise }, faders: [dryOut, sendOut], nodes: [dry, send, dryOut, sendOut], t0, bar: 0, idx: 0, until: t0, endAt: null };
}

/** The first bar line this deck has not written past. */
function barLine(d: Deck, now: number): number {
  const len = barSeconds(d.name);
  return d.t0 + Math.max(0, Math.ceil((Math.max(now + 0.05, d.until) - d.t0) / len - 1e-6)) * len;
}

/** Write this deck's notes up to `horizon`. Every time comes from the deck's start and the bar number,
 *  so nothing drifts; a note whose moment has gone (a tab that slept) is dropped, never played late. */
function write(d: Deck, now: number, horizon: number): void {
  const len = barSeconds(d.name), end = d.endAt ?? Infinity;
  const playing = Math.floor((now - d.t0) / len);
  if (playing > d.bar) { d.bar = playing; d.idx = 0; }
  for (;;) {
    const start = d.t0 + d.bar * len;
    if (start >= horizon || start >= end - 1e-3) return;
    const notes = trackNotes(d.name, d.bar);
    while (d.idx < notes.length) {
      const [voice, f, at, sounds, level] = notes[d.idx];
      const t = start + at;
      if (t >= horizon || t >= end - 1e-3) return;
      if (t >= now + 0.005) VOICES[voice](d.rig, f, t, sounds, level ?? 1);
      d.idx++;
    }
    d.bar++; d.idx = 0;
  }
}

function unplug(d: Deck): void {
  for (const n of d.nodes) { try { n.disconnect(); } catch { /* already gone */ } }
}

function tick(): void {
  try {
    if (ticks++ % 20 === 0) audible = musicOn() && !muted();       // another tab may have changed the choice
    const want = audible ? wanted : null;
    if (!want && !live && !dying.length) { if (!wanted) stop(); return; }
    if (!unlocked()) return;
    const r = getRig();
    if (!r) return;
    if (r.ctx.state !== 'running') {
      if (ticks % 20 === 1) void r.ctx.resume().catch(() => {});
      return;
    }
    listen(false);
    const now = r.ctx.currentTime;
    const horizon = now + (document.hidden ? AHEAD_HIDDEN : AHEAD);
    let handover = 0;
    if (live) {
      if (live.name === want) live.endAt = null;                                       // it changed its mind in time
      else if (!want) live.endAt = Math.min(live.endAt ?? Infinity, now + 0.03);        // silence: now
      else if (live.endAt == null) live.endAt = barLine(live, now);                     // another track: at the bar line
      write(live, now, horizon);
      live.until = Math.max(live.until, horizon);
      if (live.endAt != null && live.endAt <= horizon) {
        const at = Math.max(live.endAt, now);
        for (const g of live.faders) { g.gain.setValueAtTime(1, at); g.gain.linearRampToValueAtTime(0, at + (want ? FADE : FADE_OFF)); }
        live.endAt = at;
        dying.push(live);
        handover = want ? at : 0;
        live = null;
      }
    }
    if (!live && want) {
      live = makeDeck(r, want, Math.max(handover, now + 0.06));
      write(live, now, horizon);
      live.until = Math.max(live.until, horizon);
    }
    dying = dying.filter((d) => { if (now < (d.endAt ?? 0) + 2) return true; unplug(d); return false; });
  } catch { /* no music is fine */ }
}

function start(): void {
  if (timer != null || typeof window === 'undefined') return;
  if (!unlocked()) listen(true);
  ticks = 0;
  timer = setInterval(tick, TICK_MS);
  tick();
}
function stop(): void {
  if (timer != null) clearInterval(timer);
  timer = null;
  if (typeof window !== 'undefined') listen(false);
}

/** Play this track (null: fade to silence). The change comes at the next bar line, one track fading
 *  into the other; asking for what is already wanted does nothing. Never throws. */
export function setTrack(name: TrackName | null): void {
  if (name === wanted) return;
  wanted = name;
  if (name) start();
}

/** What was last asked for (not whether it can be heard). */
export const currentTrack = (): TrackName | null => wanted;

// a dev server swapping this module must not leave the old one playing under the new
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    stop();
    for (const d of [...dying, ...(live ? [live] : [])]) unplug(d);
    try { bus?.dry.disconnect(); bus?.wet.disconnect(); } catch { /* fine */ }
  });
}
