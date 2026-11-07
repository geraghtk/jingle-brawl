// Wait until BOTH local emulators answer before any test runs. The webServer
// readiness URL only covers the database; if a test starts before the Auth
// emulator is listening, anonymous sign-in fails and the rules reject writes.
const ENDPOINTS = [
  'http://127.0.0.1:9000/.json?ns=jingle-brawl-e8308-default-rtdb',
  'http://127.0.0.1:9099/emulator/v1/projects/jingle-brawl-e8308/config',
];

module.exports = async () => {
  if (process.env.CI || process.env.TRANSPORT === 'server') return; // the room server needs no emulators
  const deadline = Date.now() + 60000;
  for (const url of ENDPOINTS) {
    for (;;) {
      try {
        if ((await fetch(url)).status < 500) break; // 401/403 = up (the rules lock the root)
      } catch { /* not listening yet */ }
      if (Date.now() > deadline) throw new Error(`Emulator not ready: ${url}`);
      await new Promise((r) => setTimeout(r, 500));
    }
  }
};
