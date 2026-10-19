# Jingle Brawl — economics & game theory

How the chip economy and the incentives actually behave, measured on the rules engine, plus a
claim-by-claim check of an external game-theory review (2026-10-18). Everything here is
reproducible:

```bash
npm run analyze:economics                 # the full report (GAMES=600 for tighter intervals)
npx vitest run web/src/sim                # the pinned properties (web/src/sim/economics.test.ts)
```

## How it's measured

`web/src/sim/simulate.ts` plays complete games **through the real engine** (`web/src/core`), so
each draw, bid, duel, Gambit, Reprisal, yield, steal, auction and endgame step runs the plan
functions the apps use. The strategies (`strategies.ts`) are small explicit policies:

| strategy | does |
|---|---|
| bid *k* | bids *k* on any gift better than what it holds |
| value-scaled | bids 1 chip per 2 points of gain |
| passive | never bids; Gambits when nobody wants its gift |
| random | coin flips (baseline) |

**Payoff = the value (0–12, to you) of the gift you end with.** Duels are coin flips. Each gift
has a shared quality plus each player's own taste. A **deviation gain** is: one player switches
strategy at a table of others, and we measure how their final gift compares with the table's, in
the same seeded games. Positive means switching pays, and a table where no switch pays is an
equilibrium (among these strategies).

Limits: duels are 50/50 (real duels reward skill), the strategies don't learn, and votes, bets and
party mini-games aren't simulated.

## The big finding: what are leftover chips worth?

The rules say the gift is the prize, and **chips are worth nothing when the game ends**. But the
app's final standings and the phone's "1st place" rank players by **chips**. Which of the two the
table plays for flips the whole equilibrium:

| leftover chip worth (w) | bid 2 joining a passive table | passive joining a bid-2 table | bid 4 joining a bid-2 table |
|---|---|---|---|
| 0 (the rules) | **+1.42** ▲ | +0.01 | **+0.37** ▲ |
| 0.25 | **+0.57** ▲ | +0.29 | **−0.91** ▼ |
| 0.5 | −0.29 | **+0.56** ▲ | **−2.18** ▼ |

- **w = 0:** passive play is invaded, and bidding escalates (outbidding to 3 or 4 pays). Chips are
  so cheap that 68–94 % go unspent.
- **w ≈ 0.5:** passive play becomes a best response and high bids are badly dominated. This is
  the world the external review describes, and probably what its self-play run optimised for.

**Recommendation:** decide what chips are for at the end, because it matters more than any tax
tweak. Either rank the final standings by the gift (or by Party Score) so chips really are only
for spending, or give leftover chips an explicit use (for example, chips break ties for Party
MVP, or buy a last swap). Pinned by *"the regime switch…"* and *"KNOWN ISSUE — chips go mostly
unspent"*.

## The external review, claim by claim

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Santa Tax makes 2 a dominant ceiling; bids of 3–4 are strictly dominated | **Not under the rules as written.** The chip arithmetic is right (pinned in the cost-table test), but it leaves out two effects. A higher bid **wins the right to duel** over other bidders, and the winning bid becomes the gift's **Naughty level**, so taking it back costs base + bid. With chips worth nothing at the end, a lone bid-3 at a bid-2 table ends with a better gift (**+0.54 ± 0.27**). The claim does hold once leftover chips count (w ≥ 0.25). | tests *"bidding above 2 is NOT dominated"*, *"…a high winning bid also buys protection"* |
| — | "Progressive Santa Tax is the single highest-leverage change" | **Low leverage.** Changing the cap or removing the tax shifts the deviation gains by at most ~0.15. At the common bid of 2 the tax collects **0** chips (a pot of 2 pays out in full). | test *"the Santa Tax barely moves incentives"*; report table 3 |
| 2 | First-price sealed bids reward deception; use Vickrey | **Mostly moot here.** The thing being auctioned is a *coin-flip duel*, not the gift. Bids are capped at 4 out of 10–12 chips, and chips have no end value. So there's little "true value" to shade, and the best response is mainly *how high to go* (see #1). A second-price toggle is cheap to add and to simulate, but there's no evidence yet that it fixes a real problem. | reasoning; not simulated |
| 3 | Two Nash equilibria; passive is stable and Pareto-superior | **Depends on chip worth.** At w = 0 a passive table is invaded by every bidding strategy (+0.6 to +1.7). At w = 0.5 passive is a best response. Average gift value is about the same for every table (5.65–5.77), because the economy **reallocates** gifts rather than creating value, so "Pareto-superior" doesn't hold for gift value. Value-scaled bidding allocates slightly better (5.77). | report tables 1, 2, 4; test *"a passive table … is not stable"* |
| 4 | Chip-depleted players face gift "fortresses" | **The prices are real; running out of chips isn't.** Max Naughty reaches 5.4 at bid-4 tables (a 6-chip challenge), but players end with 68–94 % of their chips, so affordability rarely binds. A fortress is a deterrent price, not a lock-out. Naughty decay would lower that deterrence. | report table 2 (max Naughty, chips left) |
| 5 | Mild deflation at high bids | **Yes, but the cause is different.** At bid 2 there's no tax at all. Supply shrinks because **Gambit, Reprisal and Steal costs are destroyed** (≈ 10 chips a game) and only about 3 dividends are minted. It doesn't matter much while chips are this abundant. The ledger test proves no chips leak beyond these named flows. | tests *"the chip ledger"*, *"…the chip drain is Gambit/Reprisal/Steal costs"* |
| 6 | Yield gives the Opener nothing, so it's dominated | **The premise is wrong.** A yielding Opener goes **back in the draw bag** and opens another gift later. Whether yielding pays depends on how good the contested gift is to them. | test *"yielding is not 'receiving nothing'"* |
| — | Contested Gift Bonus, Misfit Dividend, Yield Bonus (+1 chip each) | **Low value while chips are worthless at the end:** these mint more of a currency players already leave unspent. They'd matter if leftover chips count (see the big finding). | inference from table 2 |
| — | Spectator bets on by default | Zero-sum, so it doesn't change gift incentives. It's an engagement choice (already a setting). Simulating it needs vote strategies. | not simulated |

## The pinned properties (`web/src/sim/economics.test.ts`)

**Invariants (must always hold):**
- **The chip ledger:** across 300 simulated games with mixed strategies and settings, end chips =
  start − Santa Tax − Gambit/Reprisal/Steal costs + Loser's Dividends. Tolls are zero-sum. No
  negative chips, and every game ends.
- **The sealed-duel cost table** (defaults):

  | bid | challenger wins | challenger loses | defender wins | defender loses |
  |---|---|---|---|---|
  | 1 | 0 | −1 | +1 | +1 (dividend) |
  | 2 | 0 | −2 | +2 | +1 |
  | 3 | −1 | −3 | +2 | +1 |
  | 4 | −2 | −4 | +2 | +1 |

- **Protection:** a winning bid sets Naughty = the bid, so challenging that gift costs 1 + bid.
- **Yield:** the yielding Opener returns to the draw bag.

**Findings (seeded and deterministic; a rule change that flips one should come with an update to
this file):**
- Outbidding 2 pays (w = 0).
- The tax barely moves incentives.
- A passive table isn't stable (w = 0).
- The regime switch at w = 0.5.
- Chips go mostly unspent (**known issue**).
- At bid 2 the chip drain is costs, not tax.

## Full report (600 games per cell)

Run `GAMES=600 npm run analyze:economics` to regenerate it. Its tables are summarised above. The
deviation matrix (row = table, column = lone deviant, w = 0):

| table \ deviant | bid 1 | bid 2 | bid 3 | bid 4 | value-scaled | passive |
|---|---|---|---|---|---|---|
| **bid 1** | — | +0.85 ▲ | +0.68 ▲ | +0.57 ▲ | +0.75 ▲ | −0.01 |
| **bid 2** | −0.31 ▼ | — | +0.54 ▲ | +0.37 ▲ | +0.20 | +0.01 |
| **bid 3** | −0.25 | −0.29 ▼ | — | +0.16 | +0.23 | +0.08 |
| **bid 4** | +0.03 | +0.03 | +0.01 | — | +0.37 ▲ | +0.53 ▲ |
| **value-scaled** | −0.30 ▼ | −0.44 ▼ | −0.04 | −0.07 | — | +0.60 ▲ |
| **passive** | +1.68 ▲ | +1.42 ▲ | +0.88 ▲ | +0.57 ▲ | +1.08 ▲ | — |

With these strategies, **bid 3** is the only table nobody gains by leaving (the 95 % intervals are
about ±0.27).
