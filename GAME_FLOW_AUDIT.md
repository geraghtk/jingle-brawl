# Jingle Brawl Game Flow Audit

## Phase 1: Main Game Scenarios

### Scenario 1: No Bids (Opener Keeps Gift)
- **Trigger**: No one bids on newly opened gift
- **Options**: 
  - Keep gift → Turn ends
  - Grinch's Gambit → Challenge someone else
- **State Changes**: None if kept
- **Status**: ✅ Implemented

### Scenario 2: Single Bidder
- **Trigger**: Exactly one player bids
- **Flow**: Bidder vs Opener duel
- **State Changes**: 
  - Bidder pays bid to pot
  - Winner gets gift + pot
  - Loser gets dividend
  - Reprisal opportunity
- **Status**: ✅ Implemented

### Scenario 3: Two Bidders (Different Amounts)
- **Trigger**: Two bidders, different amounts
- **Flow**: 
  1. Opener can Fight (vs highest) or Yield
  2. If Fight: Highest bidder vs Opener
  3. If Yield: C1 vs C2 (both pay half bids)
- **Status**: ✅ Implemented

### Scenario 4: Two Bidders (Tie)
- **Trigger**: Two bidders, same amount
- **Flow**:
  1. Opener chooses Fight or Yield
  2. If Fight: Tiebreaker → Winner vs Opener
  3. If Yield: C1 vs C2 directly
- **Status**: ✅ Implemented

### Scenario 5: 3+ Way Tie
- **Trigger**: 3+ bidders with same highest amount
- **Flow**:
  1. Random select 2
  2. Opener chooses Fight or Yield
  3. If Fight: Tiebreaker → Winner vs Opener
  4. If Yield: Selected C1 vs C2
- **Status**: ✅ Implemented

---

## Duel Types

### Regular Duel (Challenger vs Defender)
- **Pot**: Challenger's bid (minus tax if ≥3)
- **Winner**: Gets gift + pot
- **Loser**: Gets +1 dividend, reprisal opportunity
- **Status**: ✅ Implemented

### Tiebreaker Duel
- **Pot**: 0
- **Winner**: Becomes challenger for main duel
- **Loser**: No dividend, no reprisal
- **Status**: ✅ Implemented

### Yield Duel (C1 vs C2)
- **Pot**: Half bids from both (rounded up), minus tax
- **Winner**: Gets gift
- **Loser**: No dividend, no reprisal
- **Opener**: Becomes giftless
- **Status**: ✅ Implemented (fixed opener giftless issue)

### Reprisal Duel
- **Cost**: Min cost (1 + naughty level) to bank
- **Pot**: 0
- **Winner**: Gets target gift
- **Loser**: Gets +1 dividend
- **Chain**: Can trigger another reprisal (up to max depth)
- **Status**: ⚠️ Issue found (see below)

### Grinch's Gambit
- **Cost**: Min cost to bank
- **Pot**: 0
- **Winner**: Gets target gift, their old gift to loser
- **Loser**: Gets +1 dividend, reprisal opportunity
- **Status**: ✅ Implemented

---

## Swap Algorithm Scenarios

### Case A: Winner had NO gift, Loser had contested gift
- Winner gets gift
- Loser becomes giftless
- **Status**: ✅ Fixed

### Case B: Winner had gift X, Loser had contested gift only
- Winner gets contested gift
- Loser gets gift X
- **Status**: ✅ Implemented

### Case C: Winner had gift X, Loser had contested gift AND another gift Y
- Winner gets contested gift
- Loser gets gift X
- Gift Y goes to misfit pile
- **Status**: ✅ Implemented

### Case D: Yield Duel - Winner is C1, Loser is C2, Opener steps aside
- Winner (C1) gets contested gift
- C1's old gift (if any) goes to C2
- C2's old gift (if any) goes to misfit
- Opener becomes giftless
- **Status**: ✅ Implemented

---

## Phase 2: Endgame Scenarios

### Path A: Steal
- Active (giftless) challenges player's gift
- **If Active Wins**: Gets defender's gift, defender takes misfit
- **If Active Loses**: Takes misfit, defender keeps gift
- **Status**: ✅ Implemented

### Path B: Auction (With Bids)
- Active vs highest bidder
- **If Active Wins**: Gets misfit
- **If Bidder Wins**: 
  - Gets misfit
  - If bidder had gift: Active gets bidder's old gift
  - If bidder was giftless: Active must restart turn
- **Status**: ✅ Implemented

### Path B: No Bids Duel
- Active chooses defender
- Active ALWAYS gets misfit
- **If Active Loses**: Pays 1 chip toll to defender
- **Status**: ✅ Implemented

---

## Issues Found & Fixed

### Issue 1: Reprisal Misfit Claim - Old Gift Not Handled ✅ FIXED
**Scenario**: Reprisal player has a gift and claims a misfit (auto-win)
**Bug**: Their old gift is not moved to misfit pile
**Fix**: Added logic in both `host.html` and `player.html` to move old gift to misfit pile before claiming

### Issue 2: Yield Duel - Opener Not Cleared ✅ FIXED (Earlier)
**Scenario**: Opener yields, C1 vs C2 fight for gift
**Bug**: Opener's giftId was not being cleared
**Fix**: Added yield duel handling in `recordDuelWinner`

### Issue 3: Tiebreaker Flag Not Cleared ✅ FIXED (Earlier)
**Scenario**: After tiebreaker, winner vs opener should be regular duel
**Bug**: `isTiebreak: true` flag persisted, causing wrong buttons
**Fix**: `startDuel` now explicitly sets `isTiebreak: false`

### Issue 4: Winner Had No Gift - Loser Stays Giftless ✅ FIXED (Earlier)
**Scenario**: Giftless challenger beats opener
**Bug**: Opener's giftId was not cleared
**Fix**: Added logic to clear loser's giftId when winner had no gift to trade

### Issue 5: Gambit - Reprisal Allowed ✅ VERIFIED OK
**Scenario**: Opener uses Gambit, loses, should get reprisal
**Status**: `isGambit` is NOT in the reprisal exclusion list, so reprisal works correctly
**No fix needed**

---

## Verification Matrix

| Scenario | Pot | Dividend | Reprisal | Naughty | Swap |
|----------|-----|----------|----------|---------|------|
| Regular Duel | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tiebreaker | ✅ (0) | ✅ (none) | ✅ (none) | N/A | N/A |
| Yield Duel | ✅ | ✅ (none) | ✅ (none) | ✅ | ✅ |
| Reprisal | ✅ (0) | ✅ | ✅ | ✅ | ✅ FIXED |
| Gambit | ✅ (0) | ✅ | ✅ | ✅ | ✅ |
| Endgame Steal | ✅ (0) | ✅ | N/A | ✅ | ✅ |
| Endgame Auction | ✅ | ✅ | N/A | ✅ | ✅ |
| No-Bids Duel | ✅ (0) | ✅ (none) | N/A | ✅ | ✅ |

---

## Test Scenarios Checklist

### Main Game Scenarios
- [ ] Open gift, no bids → Opener keeps gift
- [ ] Open gift, no bids → Opener uses Gambit, wins
- [ ] Open gift, no bids → Opener uses Gambit, loses → gets reprisal
- [ ] Open gift, one bidder → Challenger wins → gets gift, reprisal for loser
- [ ] Open gift, one bidder → Defender wins → keeps gift, reprisal for challenger
- [ ] Open gift, two bidders (different) → No yield → regular duel
- [ ] Open gift, two bidders (different) → Yield → C1 vs C2
- [ ] Open gift, two bidders (tie) → Fight → tiebreaker → winner vs opener
- [ ] Open gift, two bidders (tie) → Yield → C1 vs C2
- [ ] Open gift, 3+ way tie → Random select 2 → Fight → tiebreaker → winner vs opener
- [ ] Open gift, 3+ way tie → Random select 2 → Yield → C1 vs C2

### Swap Algorithm Scenarios
- [ ] Winner had no gift, loser had gift → loser becomes giftless
- [ ] Winner had gift, loser had gift → loser gets winner's gift
- [ ] Winner had gift, loser had different gift → loser's old gift to misfit
- [ ] Yield: both challengers have gifts → misfit creation

### Reprisal Scenarios
- [ ] Reprisal: target player's gift → duel starts
- [ ] Reprisal: target misfit → auto-claim
- [ ] Reprisal: target misfit when already have gift → old gift to misfit, claim new
- [ ] Reprisal: can't afford → auto-skip
- [ ] Reprisal: no targets → auto-skip
- [ ] Reprisal: chain (up to max depth)

### Endgame Scenarios
- [ ] Path A: Active wins steal → gets gift, defender takes misfit
- [ ] Path A: Active loses steal → takes misfit
- [ ] Path B: Auction with bids, active wins → gets misfit
- [ ] Path B: Auction with bids, bidder wins, had gift → active gets bidder's gift
- [ ] Path B: Auction with bids, bidder wins, giftless → active restarts turn
- [ ] Path B: No bids, active wins → gets misfit free
- [ ] Path B: No bids, active loses → gets misfit, pays 1 toll

