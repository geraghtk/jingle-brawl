// Join mode of the agent harness (`npm run agents:join -- ROOM`): a person hosts and plays a
// phone, persona agents fill the other seats. The agents must play ONLY their own seats —
// never press host buttons, never move for the person.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');
const { Arena } = require('../agents/arena');

test('join mode: agents sit at a person\'s table, bid for themselves, and leave the host and the person alone', async ({ browser }, info) => {
  test.setTimeout(120000);
  const { host, hostCtx, roomCode } = await H.createGame(browser);
  const { pages, contexts } = await H.joinPlayers(browser, roomCode, ['Kevin']);
  const arena = new Arena({
    browser, join: roomCode, seed: 1, settings: {}, outDir: info.outputPath('agents'),
    seats: [{ name: 'Alice', persona: 'brawler' }, { name: 'Bob', persona: 'wildcard' }],
  });
  const run = arena.setup().then(() => arena.play());
  try {
    await H.waitForGame(roomCode, (g) => Object.keys(g.players || {}).length === 3, { label: '2 agents joined', timeout: 30000 });
    expect((await H.dbGet(roomCode)).phase).toBe('lobby'); // the agents don't press Start — the person does

    await H.startGame(host, roomCode);
    await H.drawGift(host, roomCode, 'Lava Lamp');
    const g = await H.dbGet(roomCode);
    const byName = await H.playersByName(roomCode);
    const kevin = byName.Kevin.id;
    const agentBidders = ['Alice', 'Bob'].map((n) => byName[n].id).filter((id) => id !== g.currentTurn.openerId);

    // Every agent who isn't opening bids by themselves …
    await H.waitForGame(roomCode, (x) => agentBidders.every((id) => x.currentTurn?.bids?.[id]), { label: 'agent bids', timeout: 20000 });
    if (g.currentTurn.openerId !== kevin) {
      // … but nobody bids for the person, and nobody reveals while they're still deciding
      await host.waitForTimeout(2500);
      const t = (await H.dbGet(roomCode)).currentTurn;
      expect(t.bids?.[kevin]).toBeUndefined();
      expect(t.phase).toBe('bidding');
      await H.submitBid(pages.Kevin, 1);
      await H.waitForGame(roomCode, (x) => x.currentTurn?.bids?.[kevin], { label: 'Kevin bid' });
    }
    // All bids are in: revealing is the host's job, and the host is the person
    await host.waitForTimeout(2500);
    expect((await H.dbGet(roomCode)).currentTurn.phase).toBe('bidding');

    arena.stop();
    const result = await run;
    expect(result.outcome).toBe('stopped');
    expect(result.issues.filter((i) => i.kind !== 'retry')).toEqual([]);
  } finally {
    arena.stop();
    await run.catch(() => {});
    await arena.close();
    await hostCtx.close();
    for (const c of contexts) await c.close();
  }
});
