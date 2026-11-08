// @ts-check
// The served pages work under the server's Content-Security-Policy: the host console, a phone, the
// TV, the rule pages (fonts from Google, the room's WebSocket, QR codes and avatars as images), and
// nothing on them trips the policy. The Print button on the quick-reference card still prints.
const { test, expect } = require('@playwright/test');
const H = require('./helpers');

/** Record every CSP violation a page reports (set up before any of its scripts run). */
async function watchCsp(ctx) {
  await ctx.addInitScript(() => {
    window.__csp = [];
    document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(`${e.violatedDirective} ${e.blockedURI}`));
  });
}
const violations = (page) => page.evaluate(() => window.__csp);

test('every page runs under the CSP without a violation', async ({ browser }) => {
  test.setTimeout(90000);
  const res = await fetch(`${H.BASE}/player.html`);
  expect(res.headers.get('content-security-policy')).toContain("script-src 'self'");
  expect(res.headers.get('referrer-policy')).toBe('no-referrer');

  const { host, hostCtx, roomCode, errors } = await H.createGame(browser);
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const tvCtx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  try {
    for (const c of [ctx, tvCtx]) await watchCsp(c);
    const phone = await ctx.newPage();
    await phone.goto(`${H.BASE}/player.html?room=${roomCode}&testId=1&name=Alice`);
    await phone.getByRole('button', { name: /Join Game/i }).click();
    await phone.waitForSelector('#game-screen', { state: 'visible' });
    const tv = await tvCtx.newPage();
    await tv.goto(`${H.BASE}/tv.html?room=${roomCode}`);
    await expect(tv.getByText('Alice').first()).toBeVisible();      // the TV's socket is live
    await expect(host.getByText('Alice').first()).toBeVisible();
    expect(await violations(phone)).toEqual([]);
    expect(await violations(tv)).toEqual([]);
    await watchCsp(hostCtx);
    await host.reload();                                               // the console, watched this time
    await host.getByRole('button', { name: /Rejoin Game/i }).click();
    await expect(host.getByText('Alice').first()).toBeVisible();
    expect(await violations(host)).toEqual([]);

    const rules = await ctx.newPage();
    await rules.goto(`${H.BASE}/rules.html`);
    expect(await violations(rules)).toEqual([]);

    const card = await ctx.newPage();
    await card.goto(`${H.BASE}/quick-reference.html`);
    await card.evaluate(() => { window.__printed = 0; window.print = () => { window.__printed++; }; });
    await card.locator('#print').click();
    expect(await card.evaluate(() => window.__printed)).toBe(1);
    expect(await violations(card)).toEqual([]);
    expect(errors).toEqual([]);

    // a phone's uncaught error reaches the server (counted in /metrics, which answers this machine)
    const count = async () => Number((await (await fetch(`${H.BASE}/metrics`)).text())
      .match(/jb_client_errors_total\{page="player"\} (\d+)/)?.[1] || 0);
    const before = await count();
    await phone.evaluate(() => setTimeout(() => { throw new Error('test: a bug on a phone'); }));
    await expect.poll(count).toBe(before + 1);
  } finally {
    await Promise.all([ctx.close(), tvCtx.close(), hostCtx.close()]);
  }
});
