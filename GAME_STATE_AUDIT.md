# Game State Audit - All Scenarios

## Main Game Scenarios

### 1. Normal Duel (Challenger vs Opener/Defender)

| Scenario | Winner Has Gift? | Loser Has Gift? | Expected Result | Status |
|----------|------------------|-----------------|-----------------|--------|
| 1A | No | Yes (contested) | Winner gets gift, Loser giftless, +1 dividend, reprisal | ✅ FIXED |
| 1B | Yes | Yes (contested) | Winner gets contested, Loser gets winner's old | ✅ OK |
| 1C | Yes | Yes (both have) | Winner gets contested, Loser gets winner's old, Loser's original → misfit | ✅ OK |
| 1D | Yes (contested) | No | Defender keeps gift, Challenger unchanged | ✅ OK |
| 1E | Yes (contested) | Yes | Defender keeps gift, Challenger keeps their gift | ✅ OK |

### 2. Yield Duel (C1 vs C2, Opener Steps Aside)

| Scenario | C1 Has Gift? | C2 Has Gift? | Expected Result | Status |
|----------|--------------|--------------|-----------------|--------|
| 2A | No | No | Winner gets gift, Opener giftless, NO dividend, NO reprisal | ✅ FIXED |
| 2B | No | Yes | Winner gets gift, C2 keeps theirs, Opener giftless | ✅ OK |
| 2C | Yes | No | Winner gets gift, Loser gets winner's old, Opener giftless | ✅ OK |
| 2D | Yes | Yes | Winner gets gift, Loser gets winner's old, Loser's original → misfit, Opener giftless | ✅ OK |

### 3. Tiebreaker Duel

| Scenario | Expected Result | Status |
|----------|-----------------|--------|
| Winner goes to real duel | No swap, no dividend, proceeds to Winner vs Opener | ✅ OK |

### 4. Grinch's Gambit (Voluntary Challenge)

| Scenario | Expected Result | Status |
|----------|-----------------|--------|
| Opener wins | Opener gets target gift, Target gets opener's newly opened gift | ✅ OK |
| Opener loses | Both keep their gifts, Loser (opener) gets dividend + reprisal | ✅ OK |

### 5. Reprisal

| Scenario | Expected Result | Status |
|----------|-----------------|--------|
| Challenge player's gift | Standard duel, can chain up to max depth | ✅ OK |
| Claim misfit (no owner) | Auto-win, no naughty increase, no dividend | ✅ OK |

## Endgame Scenarios - NEED FIXES!

### 6. Endgame Path A: Steal a Gift

| Scenario | Expected Result | Current Behavior | Status |
|----------|-----------------|------------------|--------|
| Active WINS | Active gets defender's gift, **Defender takes misfit** | Defender becomes giftless! | ❌ BUG |
| Active LOSES | **Active takes misfit**, Defender keeps gift | Active stays giftless! | ❌ BUG |

### 7. Endgame Path B: Auction (With Bids)

| Scenario | Expected Result | Current Behavior | Status |
|----------|-----------------|------------------|--------|
| Active WINS | Active gets misfit, Bidder keeps their gift (if any) | ✅ OK (standard swap) | ✅ OK |
| Bidder WINS, had gift | Bidder gets misfit, Active gets bidder's old gift | ✅ OK (standard swap) | ✅ OK |
| Bidder WINS, was giftless | Bidder gets misfit, **Active restarts with next misfit** | Active stays giftless, no restart | ❌ BUG |

### 8. Endgame Path B: Auction (No Bids)

| Scenario | Expected Result | Current Behavior | Status |
|----------|-----------------|------------------|--------|
| Active WINS | Active gets misfit FREE | ✅ OK | ✅ OK |
| Active LOSES | **Active gets misfit, pays 1 chip toll to defender** | Active stays giftless! | ❌ BUG |

## Missing Flags

| Flag | Where Set | Where Checked | Issue |
|------|-----------|---------------|-------|
| `isMisfitDuel` | Nowhere! | `recordDuelWinner` (dividend/reprisal check) | Never true, misfit duels give dividend when they shouldn't |
| `isEndgameSteal` | `executeEndgameSteal` | Nowhere! | No special handling |
| `isEndgameAuction` | `startEndgameAuctionDuel` | Nowhere! | No special handling |
| `isNoBidsDuel` | `executeNoBidsDuel` | Nowhere! | No special handling |

## Required Fixes

1. **Set `isMisfitDuel: true`** for all endgame duels (no dividend, no reprisal for misfit duels)
2. **Endgame Steal - Active Wins**: Force defender to take the misfit
3. **Endgame Steal - Active Loses**: Force active player to take the misfit
4. **Endgame Auction - Bidder Wins + Giftless**: Trigger restart with next misfit
5. **Endgame No-Bids - Active Loses**: Active takes misfit, pays 1 chip toll to defender

