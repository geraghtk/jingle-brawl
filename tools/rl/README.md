# Learning agents against the game (OpenSpiel)

Fixed-habit bots (`web/src/sim/`, `tools/economy-review/`) only find the exploits someone thought to
write down. Learning agents find the ones nobody did. This folder is where they will live.

## What is installed (2026-11-13)

OpenSpiel 2.0.2 (DeepMind's game-theory and reinforcement-learning library) in a virtual environment
of its own at the repo root, `.venv-openspiel/` (not in git: see `.git/info/exclude`). It has a
Windows wheel for Python 3.13, so no WSL or build is needed. To recreate it:

```bash
python -m venv .venv-openspiel
.venv-openspiel/Scripts/python.exe -m pip install open_spiel==2.0.2
.venv-openspiel/Scripts/python.exe -c "import pyspiel; print(len(pyspiel.registered_names()), 'games')"
```

Checked working: CFR on its built-in Kuhn poker reaches an exploitability of 0.003 in 200 iterations.

## Not built yet: the game itself as an OpenSpiel game

OpenSpiel plays games written to its own interface (a state with legal actions, chance, returns,
and what each player can see). The plan, so the rules stay in one place:

1. **Talk to the real engine, not a copy of the rules.** `web/src/sim/table-bridge.ts` already serves
   a whole table over JSON lines (`offers` = the legal actions for each seat, `perform` = take one,
   seeded and on a virtual clock). A Python `pyspiel.Game` / `State` pair wraps one bridge process:
   `legal_actions()` from the seat's offers, `apply_action()` sends the press, the bridge's seed is the
   chance outcome. `jingle_brawl_env.py` at the repo root is an older stand-alone model of the rules
   and is out of date: do not extend it.
2. **Start with a cut-down game** that a solver can handle exactly (one gift, sealed bids 0 to 4,
   then a duel decided by skill; then add the Reprisal choice), solved with CFR, as a check on
   `tools/economy-review/bne.ts`.
3. **Then the whole game with self-play learners** (independent Q-learning or policy gradient from
   `open_spiel.python.algorithms`), with duel skill and gift tastes as chance. What to look for: any
   learned habit that beats the table (the simulator's deviation matrix, but discovered, not listed),
   and whether the Party Score standings can be farmed.
4. Returns: the gift's value to the player (the simulator's 1 to 10 scale) and, separately, their
   place in the Party Score standings; the economy review showed these pull in different directions.

The betting market (ROADMAP Step 2c) was simulated with a small learner written in TypeScript in
`tools/economy-review/`, not with OpenSpiel: it is a few lines and its learned policy can be read.
