# Jingle Brawl

A game of high stakes, low blows, and questionable gifts.

---

## Core Concepts

### Jingle Chips

Your currency for bidding and challenging.

### North Pole Fund (The Bank)

Holds chips paid as costs and collects Santa Tax. The Bank also pays the Loser’s Dividend.

### Gifts & Naughty Level

Each gift has a Naughty Level (0, 1, 2, …).

* A gift’s Naughty Level increases by \+1 every time that gift is involved in a duel, win or lose.  
* Naughty Level never goes down.

### Paying for Duels (Two Types)

1. **Sealed Bids (create a Pot)**  
   * Sealed bids can create a Duel Pot.  
   * The winner takes the pot, after Santa Tax.

2. **Direct Challenges (no Pot)**  
   Grinch’s Gambit, Reindeer Reprisals, and Endgame Steals are direct challenges:  
   * The challenger pays the required cost to the Bank  
   * No pot is created

### Santa Tax

If a Duel Pot is **3 chips or more**, then:

* **1 chip** goes to the North Pole fund (Santa Tax).   
  The **rest** goes to the duel winner

### Loser’s Dividend

Whenever a player **loses a duel** (as Challenger or Defender), they immediately receive **\+1 chip from the North Pole fund.**

**No dividend is paid for:**

* Misfit duels (any duel in the Misfit Lottery / Endgame involving a Misfit)  
* Yield duels  
* Tie-break duels

---

## Setup

1. Give each player **10 Jingle Chips** for players up to 10, **12 chips** for larger groups.  
2. Put all player names in a **Draw Bag** (**the Santa Sack**).  
3. Choose the **Head Elf** (the organizer by default).  
4. Define your duel methods (**see Duels below**).

---

## Duels

* A duel determines a winner between two players.  
* The Defender chooses the duel type/method.  
* The duel winner wins the contested outcome (gift, pot, etc.) based on the situation.

---

# Phase 1: The Main Game

*(Continue while wrapped gifts remain.)*

## 1\) Unwrap

* The Head Elf draws a name from the Draw Bag.  
* That player becomes the Opener and opens one wrapped gift.  
* The opened gift starts at Naughty Level \= 0\.

## 2\) Sealed Bidding

* Everyone except the Opener may place a sealed bid (chips hidden in their hand).  
* Reveal bids simultaneously.

**If nobody bids:**

* The Opener is safe and may keep the gift or use a Grinch’s Gambit Challenge (see Special Actions).

**If one or more players bid:**

* The highest bidder becomes Challenger 1\.  
* If there is a second-highest bidder, they are Challenger 2\.  
* If the highest bid is tied, resolve it with a Tie-break Duel among tied players: If there are more than 2 tied bids, randomly select two challengers.  
  * The tie-break winner becomes Challenger 1\.  
  * No Loser’s Dividend is paid for tie-break duels.

## 3\) The Duel (and Yielding)

By default, Challenger 1 duels the Opener.

### Yielding

If there was a Challenger 2, the Opener may choose to Yield (step aside).

* If the Opener yields, the duel becomes Challenger 1 vs Challenger 2\.

### Pot & Payout (Sealed Bid Duels Only)

Sealed bid duels create a pot. How much goes into the pot depends on whether it’s a normal duel or a yield duel:  
A) Normal duel (Challenger 1 vs Opener):

* Challenger 1 pays their full bid into the Duel Pot.

B) Yield duel (Challenger 1 vs Challenger 2):

* Both challengers pay half their bids into the pot, rounded down.  
  * Example: bid of 5 → pay 2  
  * bid of 4 → pay 2

The winner collects the pot, applying Santa Tax if pot ≥ 3\.

### Naughty Level

After the duel, the contested gift’s Naughty Level increases by \+1.

### Loser’s Dividend Reminder

* The duel loser gets \+1 chip immediately except in:  
  * misfit duels  
  * yield duels  
  * tie-break duels

## 4\) Swap Algorithm (Resolving the Gift)

After the duel:

1. The winner takes the contested gift.  
2. If the winner already had a gift, the winner’s old gift goes to the loser.  
3. If the loser already had a gift (and just received the winner’s old gift), then, they choose which gift to put in the Misfit Gift pile.  
4. The player whose gift was sent to the Misfit Gift pile becomes the new Head Elf.

## 5\) Draw Bag Maintenance

* When someone wins a gift by challenging, their name is removed from the Draw Bag.  
* Any player who ends up with zero gifts is added back to the Draw Bag.

---

# Special Actions

## The Grinch's Gambit

*Why settle when you can steal..*

If nobody bids on a newly opened gift:

* The Opener may keep the gift OR  
* Pay the Minimum Cost to the Bank to challenge any player for their gift.  
* This is a direct challenge: no pot is created.

(Minimum Cost is based on the target gift’s Naughty Level.)

## Reindeer Reprisal

*Because reindeer kick back..*

When a **Grinch’s Gambit** or a **standard Duel** concludes, the **Loser** gets one immediate Reprisal Duel.

**Rules for the Reprisal:**

* The Reprisal Challenger may challenge **any gift**, including Misfits (if any exist) except for the gift **involved in the duel you just lost.**  
* The player must pay the **Minimum Cost** (based on the target gift’s Naughty Level) to the Bank.  
* A Reprisal can trigger at most one other Reprisal (2 reprisals max)

---

# Phase 2: The Misfit Lottery (Endgame)

*Starts when the last wrapped gift is opened, but names still remain in the Draw Bag. (Note: If all players have gifts, this phase is skipped)*

Players still in the Draw Bag enter the endgame one by one. They are currently giftless.

### Endgame Setup (Each Turn)

* The Head Elf draws the top misfit from the Misfit Gift pile.  
  * The top misfit is the "Oldest Misfit" (First-In, First-Out)  
* The active (giftless) player chooses Path A or Path B.

## Path A: The Steal (Attack a Player)

* The active player challenges any current gift holder.  
* Cost: Pay the gift’s Challenge Cost to the Bank (based on Naughty Level).  
* Resolution:  
  * If active player wins: they take the gift; the loser must take the Target Misfit.  
  * If active player loses: they must take the Target Misfit.

## Path B: The Auction (Bid on the Misfit)

The active player tries to claim the target Misfit. Bidding opens to everyone (including others in the bag).

### Scenario 1: No Bids

* Active Player vs Chosen Defender  
  * **If Active Player wins:** They get the Misfit for **free**.  
  * **If Active Player loses:** They get the Misfit, but they must pay **1 Chip (Misfit Toll)** to the **Defender they chose**.

### Scenario 2: Bidders Exist

* Active Player vs Highest Bidder  
  * Similar to the main phase, the bidder pays into the pot.  
  * If Active Player wins: takes the Misfit gift and pot minus tax.  
  * If Bidder wins: bidder takes the Misfit gift and pot minus tax.  
    * Active Player gets the bidder’s old gift (if the bidder had one)  
    * If the bidder was in the bag (giftless), the Active Player gets nothing and their name is placed back in the draw bag.   
      If they are the last player in the bag, they do not need to challenge someone for the gift. However, other players can still bid on the gift.

### Endgame Dividend Note

All Misfit Lottery duels are misfit duels → no Loser’s Dividend.

## Minimum Cost (Challenge Cost)

Some challenges don’t use sealed bidding. Instead, the challenger must pay a Minimum Cost to the North Pole Fund (Bank) based on the Naughty Level of the target gift.  
This applies to:

* Grinch’s Gambit  
* Reindeer Reprisals  
* Endgame Steals (Path A)

Default Formula  
Minimum Cost \= 1 \+ (Naughty Level of the target gift)  
Examples

* Target gift Naughty Level 0 → Cost 1 chip  
* Target gift Naughty Level 1 → Cost 2 chips  
* Target gift Naughty Level 2 → Cost 3 chips  
* Target gift Naughty Level 5 → Cost 6 chips

Notes

* The cost is paid only by the challenger. Defending is always free.  
* Paying the Minimum Cost creates no pot (it goes straight to the Bank).  
* The target gift’s Naughty Level still increases by \+1 after the duel (because it was involved in a duel).

---

## FAQ

**Q: Do I ever pay chips to defend my gift?**  
**A**: No. Defending is always free. Only the aggressor pays.

**Q: What if I run out of chips?**  
**A**: You can’t initiate challenges, but you can still defend. If you lose a duel (and it’s not an exempt duel), you get \+1 chip from the Bank.

**Q: Can I refuse a gift?**  
**A**: No. If you win it (or get stuck with it), you take it.

**Q: Does Naughty Level ever go down?**  
**A**: No. It only goes up.

**Q: Who decides the duel method?**  
**A**: The Defender.

**Q: In a "Yield Duel," what happens to the Opener?**   
**A:** The Opener gives up the gift and steps out of the fight. The two Challengers duel for the unwrapped gift. Since the Opener is left with nothing, their name goes back into the **Draw Bag**, and they will get another turn to open a gift later in the game

**Q: Can I use a "Reindeer Reprisal" if I have 0 chips?**  
**A:** No. You must be able to pay the Minimum Cost to the Bank to initiate a Reprisal. However, you still collect your Loser’s Dividend (+1 chip) first, which might give you just enough to afford a Level 0 gift attack\!

**Q: What happens if I win a duel but my "old gift" forces the loser to trash theirs, but they really wanted to keep it?**   
**A:** They have a choice\! The Swap Algorithm leaves the Loser with two gifts (their original \+ your old one). They must put one into the Misfit Pile, but they get to decide which one to trash and which one to keep.

**Q: In the Endgame, if I choose Path B (Auction) and win the Misfit, do I pay anything?**  
**A:** No\! If you choose Path B and successfully scare everyone off (no bids), or you defeat the chosen defender in a duel, you get the Misfit for free. You only pay if you lose to the chosen defender.

**Q: In the Endgame, if I choose a defender (Scenario 1\) and I lose, do I pay the toll to them personally?**  
**A:** Yes\! In this specific scenario, the defender keeps your chip. Choose your opponent wisely\!

**Q: In the Endgame, if a "Giftless" player bids against me and wins, what do I get?**  
**A:** You get... nothing. You lost the Misfit gift to someone more desperate than you. Your name goes back in the bag where you’ll get another attempt.   
If there are no other names in the bag, you immediately pick the last remaining gift. Since you are the only one left, you claim the gift automatically. However, this counts as 'Opening' a gift, so other players may still initiate a **Sealed Bid** to challenge you for it.

