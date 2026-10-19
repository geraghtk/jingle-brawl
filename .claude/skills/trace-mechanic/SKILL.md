---
name: trace-mechanic
description: "Trace a game mechanic across host.html and player.html to understand how it works end-to-end. Use when investigating how a feature works, debugging cross-file issues, or before modifying a mechanic."
argument-hint: "[mechanic name, e.g. 'bidding', 'grinch gambit', 'minigame', 'duel']"
allowed-tools: Read Grep Glob
model: sonnet
context: fork
agent: Explore
---

Trace the game mechanic "$ARGUMENTS" across the Jingle Brawl codebase.

## What to find

1. **Host-side logic** in `host.html`: Functions that drive this mechanic, Firebase writes, state transitions
2. **Player-side logic** in `player.html`: Firebase listeners, UI updates, player inputs/writes
3. **Firebase paths**: What paths are read/written, and by which side
4. **State machine transitions**: What phase/state changes occur
5. **Config dependencies**: Any `GAME_CONFIG` values from `firebase-config.js` that affect behavior
6. **Edge cases**: Tie handling, timeouts, error states

## Output format

Produce a structured trace:

```
## [Mechanic Name] — End-to-End Trace

### Flow
1. [Step] — host.html:L### — [what happens]
2. [Step] — Firebase path: [path] — [what's written]
3. [Step] — player.html:L### — [what listens and reacts]
...

### Firebase Paths
- `games/{room}/path` — written by [host|player], read by [host|player]

### Config Values
- `GAME_CONFIG.X` — [what it controls]

### Edge Cases
- [scenario] — [how it's handled]
```
