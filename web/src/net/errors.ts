// A page's uncaught errors go to the room server (POST /api/client-error), so a bug on someone's
// phone shows up in the server's log. Only which page, the message and the stack: no URL (room
// codes and names ride in it), nothing typed. A few per page load at most.

const MAX_REPORTS = 5;

export interface ErrorReport { page: string; message: string; stack?: string }

/** Turn what the browser caught into a report (null for noise not worth sending). */
export function toReport(page: string, err: unknown, fallback?: string): ErrorReport | null {
  const e = err as { message?: unknown; stack?: unknown } | null;
  const message = String((e && typeof e.message === 'string' && e.message) || fallback || err || '').slice(0, 500);
  // a script from somewhere else (an extension) failing tells us nothing
  if (!message || message === 'Script error.' || message.includes('ResizeObserver loop')) return null;
  return { page, message, ...(e && typeof e.stack === 'string' ? { stack: e.stack.slice(0, 2000) } : {}) };
}

export function reportErrors(page: 'host' | 'player' | 'tv', send: (body: string) => void = beacon): void {
  let sent = 0;
  const report = (r: ErrorReport | null) => {
    if (!r || sent >= MAX_REPORTS) return;
    sent++;
    try { send(JSON.stringify(r)); } catch { /* reporting must never break the page */ }
  };
  window.addEventListener('error', (ev) => report(toReport(page, ev.error, ev.message)));
  window.addEventListener('unhandledrejection', (ev) => report(toReport(page, ev.reason)));
}

function beacon(body: string): void {
  const blob = new Blob([body], { type: 'application/json' });
  if (navigator.sendBeacon?.('/api/client-error', blob)) return;
  void fetch('/api/client-error', { method: 'POST', body, keepalive: true, headers: { 'Content-Type': 'application/json' } }).catch(() => {});
}
