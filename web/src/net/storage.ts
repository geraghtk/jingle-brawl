// Per-tab storage: ?testId=N makes a browser tab its own player (its own saved session).
const params = new URLSearchParams(location.search);
export const testId = params.get('testId');

/** localStorage key, separate per ?testId tab. */
export const storageKey = (k: string) => k + (testId ? `_test${testId}` : '');
