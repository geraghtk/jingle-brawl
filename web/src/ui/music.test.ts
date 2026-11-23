import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import type { Game } from '../core/types';
import { barSeconds, midiOf, musicBlocked, musicOn, setMusicDefault, setTrack, TENSE_MS, TRACK_NAMES, trackBars, trackFor, trackNotes, trackScore } from './music';

const VOICES = ['chime', 'mallet', 'pluck', 'brass', 'drum', 'air', 'tin'];

/** The pitch classes of a chord name: C, Am, D7, Bb. */
function chordTones(name: string): number[] {
  const m = /^([A-G][#b]?)(m?)(7?)$/.exec(name);
  if (!m) throw new Error(`not a chord: ${name}`);
  const root = midiOf(`${m[1]}4`) % 12;
  return [0, m[2] ? 3 : 4, 7, ...(m[3] ? [10] : [])].map((s) => (root + s) % 12);
}
/** The chord under a beat: a bar with two names changes half way. */
const chordAt = (bar: string, beat: number, beatsPerBar: number) => {
  const names = bar.split(' ');
  return names[Math.min(names.length - 1, Math.floor((beat / beatsPerBar) * names.length))];
};

describe('the music: the tunes', () => {
  it('every track comes round on a whole number of bars: 8 or 16, every part as long as the track', () => {
    for (const name of TRACK_NAMES) {
      const s = trackScore(name);
      expect([8, 16], name).toContain(s.bars);
      expect(trackBars(name), name).toBe(s.bars);
      expect(s.chords.length, name).toBe(s.bars);
      for (const [part, bars] of Object.entries(s.parts)) expect(bars.length, `${name}/${part}`).toBe(s.bars);
      // it wraps without a seam: the bar after the last is the first
      expect(trackNotes(name, s.bars)).toBe(trackNotes(name, 0));
      expect(barSeconds(name), name).toBeCloseTo((4 * 60) / s.bpm);
    }
  });

  it('every note is on a real instrument, within hearing, starts inside its bar and is not too loud', () => {
    for (const name of TRACK_NAMES) {
      for (let bar = 0; bar < trackBars(name); bar++) {
        const notes = trackNotes(name, bar);
        let last = 0;
        for (const [voice, f, at, len, level] of notes) {
          const where = `${name} bar ${bar + 1}`;
          expect(VOICES, where).toContain(voice);
          expect(f, where).toBeGreaterThan(40); expect(f, where).toBeLessThan(5000);
          expect(at, where).toBeGreaterThanOrEqual(0); expect(at, where).toBeLessThan(barSeconds(name));
          expect(at, where).toBeGreaterThanOrEqual(last); last = at;                 // in order: the scheduler walks them
          expect(len, where).toBeGreaterThan(0.02); expect(len, where).toBeLessThan(2);
          expect(level, where).toBeGreaterThan(0); expect(level, where).toBeLessThanOrEqual(0.85);
        }
      }
    }
  });

  it('no track goes silent for a whole bar, and the melody never rests for one', () => {
    for (const name of TRACK_NAMES) {
      const s = trackScore(name);
      for (let bar = 0; bar < s.bars; bar++) {
        expect(trackNotes(name, bar).length, `${name} bar ${bar + 1}`).toBeGreaterThan(3);
        expect(s.parts.lead[bar].length, `${name} lead bar ${bar + 1}`).toBeGreaterThan(0);
      }
    }
  });

  it('the harmony is real: bass and chords stay in the bar\'s chord, the melody leaves it only in passing', () => {
    for (const name of TRACK_NAMES) {
      const s = trackScore(name);
      for (const [part, bars] of Object.entries(s.parts)) {
        if (part === 'drum' || part === 'bells' || part === 'tick') continue;     // unpitched
        bars.forEach((notes, bar) => {
          for (const n of notes) {
            const chord = chordAt(s.chords[bar], n.beat, s.beatsPerBar);
            const inChord = chordTones(chord).includes(n.midi % 12);
            const passing = part === 'lead' && n.beats < 1;
            expect(inChord || passing, `${name}/${part} bar ${bar + 1} beat ${n.beat + 1}: ${n.midi} over ${chord}`).toBe(true);
          }
        });
      }
    }
  });

  it('a passing note in the melody moves by step from the note before it', () => {
    for (const name of TRACK_NAMES) {
      const s = trackScore(name);
      const line = s.parts.lead.flatMap((notes, bar) => notes.map((n) => ({ ...n, bar })));
      line.forEach((n, i) => {
        if (chordTones(chordAt(s.chords[n.bar], n.beat, s.beatsPerBar)).includes(n.midi % 12) || i === 0) return;
        expect(Math.abs(n.midi - line[i - 1].midi), `${name} bar ${n.bar + 1} beat ${n.beat + 1}`).toBeLessThanOrEqual(2);
      });
    }
  });

  it('each track ends ready to begin again: the last bar leads home to the first bar\'s chord', () => {
    for (const name of TRACK_NAMES) {
      const s = trackScore(name);
      const home = s.chords[0].split(' ')[0];
      const lastChord = s.chords[s.bars - 1].split(' ').pop()!;
      const fifthAbove = (midiOf(`${home.replace(/m|7/g, '')}4`) + 7) % 12;
      // home again, or its dominant
      expect([home, undefined].includes(lastChord) || chordTones(lastChord)[0] === fifthAbove, name).toBe(true);
    }
  });

  it('the tempos are what each moment asks for', () => {
    expect(trackScore('lobby').bpm).toBeLessThan(100);
    expect(trackScore('game').bpm).toBeGreaterThan(trackScore('lobby').bpm + 20);
    expect(trackScore('tense').bpm).toBeGreaterThan(trackScore('game').bpm);
  });

  it('the music is played on the effects\' instruments: music.ts makes no oscillator of its own', () => {
    const src = readFileSync(new URL('./music.ts', import.meta.url), 'utf8');
    expect(src).not.toMatch(/createOscillator|'square'|'sawtooth'/);
    expect(src).not.toMatch(/new Audio|<audio|fetch\(|\.mp3|\.ogg|\.wav/);
  });
});

describe('the music: which track when', () => {
  const NOW = 1_000_000;
  const arena = (lifecycle: string, extra: object = {}, view: object = {}): Game => ({
    phase: 'main', currentTurn: { phase: 'minigame_arena' },
    arenaView: { instanceId: 'a', gameId: 'ringtoss', mode: 'direct_duel', lifecycle, stateVersion: 1, participants: [], view, ...extra },
  } as unknown as Game);
  const party = (phase: string, extra: object = {}, turn = 'minigame_party'): Game => ({
    phase: 'main', currentTurn: { phase: turn }, minigame: { type: 'whack_elf', phase, phaseEndsAt: NOW + 20000, ...extra },
  } as unknown as Game);

  it('the lobby has its tune; the bidding game, where people talk, has none', () => {
    expect(trackFor({ phase: 'lobby' })).toBe('lobby');
    expect(trackFor({ phase: 'lobby', currentTurn: null })).toBe('lobby');
    for (const phase of ['bidding', 'reveal', 'duel_pending', 'duel', 'reprisal', 'tiebreaker_pending', 'endgame_choice'] as const) {
      expect(trackFor({ phase: 'main', currentTurn: { phase } }), phase).toBeNull();
    }
    expect(trackFor({ phase: 'main' })).toBeNull();
    expect(trackFor({ phase: 'endgame', currentTurn: { phase: 'endgame_bidding' } })).toBeNull();
    expect(trackFor({ phase: 'ended' })).toBe('results');
  });

  it('a mini-game: gentle through the instructions, the game tune once it runs, the happy one when a result is up', () => {
    expect(trackFor(arena('briefing'))).toBe('lobby');
    expect(trackFor(arena('paused'))).toBe('lobby');
    expect(trackFor(arena('active'))).toBe('game');
    expect(trackFor(arena('active', { result: { status: 'resolved' } }))).toBe('results');
    expect(trackFor(arena('completed'))).toBe('results');
    expect(trackFor(arena('cancelled'))).toBeNull();
    // the turn says a game is on but its view has not arrived: as the rest of the game
    expect(trackFor({ phase: 'main', currentTurn: { phase: 'minigame_arena' } })).toBeNull();
  });

  it('a Game Lab game in the lobby gets the game tune, not the lobby\'s', () => {
    const lab = { ...arena('active'), phase: 'lobby' as const };
    expect(trackFor(lab)).toBe('game');
    expect(trackFor({ ...party('playing'), phase: 'lobby' })).toBe('game');
  });

  it('the last seconds, overtime and sudden death turn tense', () => {
    expect(trackFor(arena('active', {}, { stage: 'running', endsAt: NOW + 30000 }), NOW)).toBe('game');
    expect(trackFor(arena('active', {}, { stage: 'running', endsAt: NOW + TENSE_MS - 1 }), NOW)).toBe('tense');
    expect(trackFor(arena('active', {}, { stage: 'running', endsAt: NOW + 1000 }))).toBe('game');       // no clock given
    expect(trackFor(arena('active', {}, { stage: 'overtime', endsAt: NOW + 30000 }), NOW)).toBe('tense');
    expect(trackFor(arena('active', {}, { stage: 'holding', round: 'sudden_death' }), NOW)).toBe('tense');
    expect(trackFor(arena('active', {}, { stage: 'fighting', sudden: true, stageEndsAt: NOW + 60000 }), NOW)).toBe('tense');
    // a practice round running out is not the end of anything
    expect(trackFor(arena('active', {}, { stage: 'practice', stageEndsAt: NOW + 1000 }), NOW)).toBe('game');
    expect(trackFor(arena('active', {}, { stage: 'throwing', stageEndsAt: NOW + 1000 }), NOW)).toBe('tense');
    expect(trackFor(arena('briefing', {}, { stage: 'running', endsAt: NOW + 1000 }), NOW)).toBe('lobby');
  });

  it('the party games and AI Draw Off follow their own phases', () => {
    expect(trackFor(party('intro', { awaiting: true }), NOW)).toBe('lobby');
    expect(trackFor(party('playing'), NOW)).toBe('game');
    expect(trackFor(party('playing', { phaseEndsAt: NOW + 5000 }), NOW)).toBe('tense');
    expect(trackFor(party('voting', { phaseEndsAt: NOW + 5000 }), NOW)).toBe('game');
    expect(trackFor(party('results'), NOW)).toBe('results');
    const draw = (phase: string, extra: object = {}) => party(phase, { type: 'draw_off', ...extra }, 'minigame_draw_off');
    expect(trackFor(draw('prompt_reveal'), NOW)).toBe('lobby');
    expect(trackFor(draw('drawing', { drawingEndTime: NOW + 40000 }), NOW)).toBe('game');
    expect(trackFor(draw('drawing', { drawingEndTime: NOW + 4000 }), NOW)).toBe('tense');
    expect(trackFor(draw('judging'), NOW)).toBe('tense');
    expect(trackFor(draw('results'), NOW)).toBe('results');
    // the state not here yet: still a game
    expect(trackFor({ phase: 'main', currentTurn: { phase: 'minigame_party' }, minigame: null }, NOW)).toBe('game');
  });
});

describe('the music: the player, with no audio (a test, a server)', () => {
  it('asking for a track, again and again, is silent, not an error', () => {
    expect(() => { setTrack('lobby'); setTrack('lobby'); setTrack('game'); setTrack(null); setTrack(null); }).not.toThrow();
    expect(musicBlocked()).toBe(false);
  });
  it('before anyone chooses, a screen plays or not by its own default (the TV does, the console does not)', () => {
    setMusicDefault(false); expect(musicOn()).toBe(false);
    setMusicDefault(true); expect(musicOn()).toBe(true);
  });
});
