// The arena round Nutcracker Knockout's ring is scenery: it must never stand between the camera and the fight.
import { describe, expect, it } from 'vitest';
import { crowdSeats, ROW0 } from './arena3d';

describe('the Knockout arena', () => {
  it('the crowd sits behind the ring, clear of it, never on the camera side', () => {
    const seats = crowdSeats();
    expect(seats.length).toBeGreaterThan(30);
    for (const s of seats) {
      expect(s.z).toBeLessThan(-1.5);                                   // the far side only (the camera is at +z)
      expect(Math.hypot(s.x, s.z)).toBeGreaterThanOrEqual(ROW0 - 1e-6);    // well outside the ring (its rim is at 3.3)
    }
    expect(ROW0).toBeGreaterThan(5);
    // each row is higher than the one in front, so the back rows are seen over it
    const rowY = [0, 1, 2].map((r) => seats.find((s) => s.row === r)!.y);
    expect(rowY[1]).toBeGreaterThan(rowY[0]);
    expect(rowY[2]).toBeGreaterThan(rowY[1]);
  });
});
