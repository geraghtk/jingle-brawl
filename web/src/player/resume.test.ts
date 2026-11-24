// A reloaded phone goes straight back to its seat; it is asked only when the link names another room.
import { describe, expect, it } from 'vitest';
import { resume } from './resume';

const seat = { room: 'ABCD', playerId: 'p1', name: 'Alice', token: 'tok' };

describe('resume', () => {
  it('a reload in the same room (or with no room in the address) goes straight in', () => {
    expect(resume(seat, { room: 'ABCD', roomExists: true })).toBe('enter');
    expect(resume(seat, { room: 'abcd ', roomExists: true })).toBe('enter');
    expect(resume(seat, { room: null, roomExists: true })).toBe('enter');
    expect(resume(seat, { room: '', roomExists: true })).toBe('enter');
  });
  it('a link to a different room asks which game', () => {
    expect(resume(seat, { room: 'WXYZ', roomExists: true })).toBe('ask');
  });
  it('?rejoin=1 never asks (the host coming back from the console)', () => {
    expect(resume(seat, { room: 'WXYZ', rejoin: true, roomExists: true })).toBe('enter');
  });
  it('a room that is gone, or a seat with no token, is forgotten: the join form', () => {
    expect(resume(seat, { room: 'ABCD', roomExists: false })).toBe('forget');
    expect(resume(seat, { room: 'WXYZ', rejoin: true, roomExists: false })).toBe('forget');
    expect(resume({ ...seat, token: undefined }, { room: 'ABCD', roomExists: true })).toBe('forget');
  });
});
