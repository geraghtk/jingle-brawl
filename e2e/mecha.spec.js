// @ts-check
// Mecha Santa Mayhem on real pages (from the Game Lab): four at the table, so the two captains lead
// teams of two and everyone has a leg and one more job. A phone is its jobs' pads; a step shows on the
// TV; stepping the same leg twice is a stumble; the wrong player's pad does nothing. One team walks
// the course (through the debug API for the long parts), crosses the line and wins.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

const NAMES = ['Alice', 'Bob', 'Charlie', 'Dana'];
const shot = async (page, name) => { if (process.env.MS_SHOTS) await page.screenshot({ path: `${process.env.MS_SHOTS}/${name}.png` }); };
const arena = (room) => H.dbGet(room, 'arena');
/** Stand team A's robot somewhere on the course, on its feet (the long walk is the unit tests' job). */
async function place(room, pos, more = {}) {
  const a = await arena(room);
  const { move: _m, busy: _b, crouch: _c, ...w } = a.state.teams[0].walker;
  const teams = [{ ...a.state.teams[0], walker: { ...w, pos, busyUntil: 0, lean: 0, last: -1, ...more } }, a.state.teams[1]];
  await H.dbPut(room, 'arena/state', { ...a.state, teams });
  await H.dbPut(room, 'arena/stateVersion', a.stateVersion + 50);
}
const tap = (page, id) => page.getByTestId(id).dispatchEvent('pointerdown', { pointerId: 1, pointerType: 'touch' });

test('Mecha Santa: your job is your pad; legs in turn move the robot on the TV; the same leg twice stumbles; first over the line wins', async ({ browser }) => {
  test.setTimeout(180000);
  const r = await H.createGame(browser, {});
  const { pages, contexts, errors: phoneErrors } = await H.joinPlayers(browser, r.roomCode, NAMES);
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  try {
    const tv = await tvCtx.newPage();
    const tvErrors = [];
    tv.on('pageerror', (e) => tvErrors.push(String(e)));
    await tv.goto(`${H.BASE}/tv.html?room=${r.roomCode}`);
    const p = await H.playersByName(r.roomCode);
    const nameOf = (id) => NAMES.find((n) => p[n].id === id);

    await r.host.getByTestId('lab-open').click();
    await r.host.getByTestId('lab-game').selectOption('mecha_santa');
    await expect(r.host.getByTestId('lab-mode')).toHaveValue('captain_teams');
    await r.host.getByTestId('lab-start').click();
    await expect(tv.getByTestId('mecha-stage')).toBeVisible();

    // two teams of two: a leg and one more job each
    const s0 = (await arena(r.roomCode)).state;
    expect(s0.teams.map((t) => t.members.length)).toEqual([2, 2]);
    const [A, B] = s0.teams;
    const left = pages[nameOf(A.crew[0][0])], right = pages[nameOf(A.crew[1][0])];
    await expect(left.getByTestId('mecha-my-team')).toContainText('Team A');
    await expect(left.getByTestId('mecha-my-team')).toContainText('Left leg');
    await expect(left.getByTestId('mecha-leg-0')).toBeVisible();
    await expect(left.getByTestId('mecha-leg-1')).toHaveCount(0);        // not their leg: no pad for it
    await expect(left.getByTestId('mecha-jump-0')).toBeVisible();        // JUMP is always there, gap or no gap
    await expect(left.getByTestId('mecha-word')).toContainText('call it out');   // the phone never says when to press
    await expect(right.getByTestId('mecha-leg-1')).toBeVisible();
    // the TV: both teams, who has which job, both robots at the start
    await expect(tv.getByTestId('mecha-team-A')).toContainText('Team A');
    await expect(tv.getByTestId('mecha-team-B')).toContainText('Team B');
    await expect(tv.getByTestId('mecha-job-A-0')).toContainText('L leg');
    await expect(tv.getByTestId('mecha-dist-A')).toHaveText('0%');
    await shot(tv, 'tv-start'); await shot(left, 'phone-start');

    // left, right: the robot walks, on the TV too; the other team's stays put
    await tap(left, 'mecha-leg-0');
    await H.waitForGame(r.roomCode, (g) => (g.arena.state.teams[0].walker.last === 0 && !g.arena.state.teams[0].walker.move ? g : false), { label: 'left foot down' });
    await tap(right, 'mecha-leg-1');
    const walked = await H.waitForGame(r.roomCode, (g) => (g.arena.state.teams[0].walker.last === 1 && !g.arena.state.teams[0].walker.move ? g : false), { label: 'right foot down' });
    expect(walked.arena.state.teams[0].walker.pos).toBeGreaterThan(3.9);
    expect(walked.arena.state.teams[1].walker.pos || 0).toBe(0);
    await expect(tv.getByTestId('mecha-dist-A')).not.toHaveText('0%');
    await expect(tv.getByTestId('mecha-dist-B')).toHaveText('0%');

    // the right leg again: a stumble, said on the TV and the phone
    await tap(right, 'mecha-leg-1');
    await H.waitForGame(r.roomCode, (g) => (g.arena.state.teams[0].walker.busy === 'stumble' ? g : false), { label: 'stumble' });
    await expect(right.getByTestId('mecha-word')).toContainText('Stumbled');
    await expect(tv.getByTestId('mecha-word-A')).toContainText('Stumble');
    await shot(tv, 'tv-stumble');

    // a wall ahead: put the robot at it (the long walk is the unit tests' job); nothing on the phone changes: no pad lights up, no word says what to do
    const a1 = await arena(r.roomCode);
    const wall = a1.state.course.obstacles.find((o) => o.kind === 'wall');
    await place(r.roomCode, wall.at);
    const armsPage = pages[nameOf(A.crew[3][0])];
    await armsPage.waitForTimeout(600);
    await expect(armsPage.getByTestId('mecha-smash')).not.toHaveClass(/\bgo\b/);
    await expect(armsPage.getByTestId('mecha-word')).not.toContainText(/wall/i);
    await tap(armsPage, 'mecha-smash');
    await H.waitForGame(r.roomCode, (g) => ((g.arena.state.teams[0].walker.hits || {})[wall.id] === 1 ? g : false), { label: 'one hit on the wall' });
    await shot(tv, 'tv-wall'); await shot(armsPage, 'phone-wall');

    // one stride from the line: a step takes it over; team A wins, the result stays up (practice)
    await place(r.roomCode, a1.state.course.length - 1, { hits: { w1: 9, w2: 9, w3: 9 } });
    await tap(left, 'mecha-leg-0');
    const done = await H.waitForGame(r.roomCode, (g) => (g.arena?.lifecycle === 'completed' ? g : false), { label: 'over the line', timeout: 15000 });
    expect(done.arena.result).toMatchObject({ status: 'resolved', reason: 'first over the line' });
    expect(done.arena.result.duel.winningDuelistId).toBe(A.captainId);
    await expect(tv.getByTestId('mecha-end')).toContainText('Team A wins!');
    await expect(left.getByTestId('mecha-word')).toContainText('Your team won!');
    await expect(pages[nameOf(B.members[0])].getByTestId('mecha-word')).toContainText('The other team won');
    await shot(tv, 'tv-end');
    expect([...r.errors, ...phoneErrors, ...tvErrors]).toEqual([]);
  } finally {
    await tvCtx.close();
    await r.hostCtx.close();
    for (const c of contexts) await c.close();
  }
});
