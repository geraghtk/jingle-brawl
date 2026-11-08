// The host is playing too: their phone has two tabs, their own game and the host's console. The
// console runs in a frame (its own page and styles, the same site and host seat), loaded once and kept
// running while hidden, so switching is instant and it can say when the game is waiting on the host
// (a dot on the tab; host/App posts it).
import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { Icon } from '../ui/icons';

export type SeatTab = 'game' | 'host';
const KEY = 'jb_seat_tab';

function firstTab(): SeatTab {
  const asked = new URLSearchParams(location.search).get('tab');
  if (asked === 'host' || asked === 'game') return asked;
  try { return sessionStorage.getItem(KEY) === 'host' ? 'host' : 'game'; } catch { return 'game'; }
}

export function SeatTabs({ children }: { children: ReactNode }) {
  const [tab, setTab] = useState<SeatTab>(firstTab);
  const [waiting, setWaiting] = useState(false);
  useEffect(() => {
    const on = (e: MessageEvent) => {
      if (e.origin === location.origin && e.data?.jb === 'host-move') setWaiting(!!e.data.waiting);
    };
    window.addEventListener('message', on);
    return () => window.removeEventListener('message', on);
  }, []);
  const pick = (t: SeatTab) => {
    setTab(t);
    try { sessionStorage.setItem(KEY, t); } catch { /* fine */ }
    window.scrollTo(0, 0);
  };
  return (
    <>
      <div className="seat-tabs" role="tablist" aria-label="Your screens">
        <button type="button" role="tab" id="seat-tab-game" aria-selected={tab === 'game'} aria-controls="seat-panel-game" data-testid="tab-game" onClick={() => pick('game')}>
          🎮 My game
        </button>
        <button type="button" role="tab" id="seat-tab-host" aria-selected={tab === 'host'} aria-controls="seat-panel-host" data-testid="tab-host" onClick={() => pick('host')}>
          <Icon name="crown" /> Host
          {waiting && tab !== 'host' && <span className="seat-dot" data-testid="host-waiting" role="status" aria-label="The game is waiting for you" />}
        </button>
      </div>
      <div role="tabpanel" id="seat-panel-game" aria-labelledby="seat-tab-game" hidden={tab !== 'game'}>{children}</div>
      <div role="tabpanel" id="seat-panel-host" aria-labelledby="seat-tab-host" hidden={tab !== 'host'} className="seat-host-panel">
        <iframe className="host-frame" title="Host controls" src="host.html?autoLoad=1&embedded=1" data-testid="host-frame" />
      </div>
    </>
  );
}
