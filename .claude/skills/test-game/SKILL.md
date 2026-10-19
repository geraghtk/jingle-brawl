---
name: test-game
description: "Run Playwright E2E tests and report results. Use after making game logic changes to verify nothing is broken."
allowed-tools: Bash Read
disable-model-invocation: true
---

Run the Playwright E2E tests and analyze the results.

```!
cd "$CLAUDE_SKILL_DIR/../../.." && npm run test:e2e 2>&1 | tail -50
```

## After running

1. Report which tests passed and which failed
2. For any failures, read the relevant test file and the error output to identify the root cause
3. Suggest specific fixes if the failures are related to recent changes
