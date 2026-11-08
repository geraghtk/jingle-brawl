// A strip across the top while this screen has lost the room (Wi-Fi hop, a phone waking up). Presses
// made meanwhile are kept and go through once it is back. Shown only after a moment, so a quick
// reconnect doesn't flicker.
import { useEffect, useState } from 'react';
import type { Status } from '../net/client';
import './connection.css';

export const BANNER_AFTER_MS = 1500;

export function ConnectionBanner({ status, loaded }: { status: Status; loaded: boolean }) {
  const [show, setShow] = useState(false);
  useEffect(() => {
    if (status === 'open') { setShow(false); return; }
    const t = setTimeout(() => setShow(true), BANNER_AFTER_MS);
    return () => clearTimeout(t);
  }, [status]);
  if (!show) return null;
  return (
    <div className="conn-banner" role="status" data-testid="reconnecting">
      <span className="conn-dot" aria-hidden="true" />
      {loaded ? 'Reconnecting... anything you tap now goes through once you are back.' : 'Connecting to the game...'}
    </div>
  );
}
