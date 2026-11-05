import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import * as A from './actions';
import { AVATARS, avatarOf } from './avatars';
import { apply, NOW } from './testkit';
import type { Game } from './types';

describe('avatars', () => {
  it('a player keeps the avatar they picked when they join; an unknown one is not stored', () => {
    let g: Game = { phase: 'lobby' };
    g = apply(g, A.joinPlayer(g, { playerId: 'a', name: 'Al', uid: null, now: NOW, avatar: 'sloth' }));
    g = apply(g, A.joinPlayer(g, { playerId: 'b', name: 'Bo', uid: null, now: NOW, avatar: '<img>' }));
    expect(g.players!.a.avatar).toBe('sloth');
    expect(g.players!.b.avatar).not.toBe('<img>');
    expect(avatarOf('Al', g.players!.a.avatar).id).toBe('sloth');
  });
  it("without a pick, a player gets the avatar their name gives, or the next one nobody at the table has", () => {
    let g: Game = { phase: 'lobby' };
    g = apply(g, A.joinPlayer(g, { playerId: 'a', name: 'Bob', uid: null, now: NOW }));
    expect(g.players!.a.avatar).toBe(avatarOf('Bob').id);
    for (let i = 0; i < AVATARS.length - 1; i++) g = apply(g, A.joinPlayer(g, { playerId: `p${i}`, name: 'Bob', uid: null, now: NOW }));
    const ids = Object.values(g.players!).map((p) => p.avatar);
    expect(new Set(ids).size).toBe(AVATARS.length);                  // 24 Bobs, 24 different avatars
  });
  it('every avatar has its picture, and ids are unique', () => {
    expect(new Set(AVATARS.map((a) => a.id)).size).toBe(AVATARS.length);
    for (const a of AVATARS) expect(existsSync(resolve(__dirname, `../../public/art/avatars/${a.id}.webp`)), a.id).toBe(true);
  });
});
