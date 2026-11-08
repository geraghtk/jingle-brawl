// Host (TV) app: settings → the room server makes the room and hands back the host's token →
// the game screen. A refreshed host is offered a rejoin (?autoLoad=1 skips the prompt).
import { useEffect, useRef, useState } from 'react';
import type { Game, Settings } from '../core/types';
import { createRoom, lobby } from '../net/api';
import { OffsetProvider, useRoom } from '../net/hooks';
import { forgetHostSeat, hostSeat, saveHostSeat } from '../net/seats';
import { GameScreen } from './GameScreen';
import { HostProvider } from './HostContext';
import { Setup } from './Setup';
import { Icon } from '../ui/icons';
import { ConnectionBanner } from '../ui/ConnectionBanner';
import { hostsMove } from '../core/nudge';

const forget = forgetHostSeat;
const autoLoad = new URLSearchParams(location.search).get('autoLoad') === '1';
/** The AI key typed at setup: handed to the room over the host's socket, then dropped. Never stored. */
let pendingCriticKey = '';

export function App() {
  const [room, setRoom] = useState<string | null>(null);
  const [offer, setOffer] = useState<string | null>(null);

  useEffect(() => {
    const saved = hostSeat()?.room;
    if (!saved) return;
    lobby(saved).then((l) => {
      if (!l.exists) return forget();
      if (autoLoad) setRoom(saved); else setOffer(saved);
    }).catch(() => {});
  }, []);

  async function create(settings: Settings, criticKey: string) {
    try {
      const { code, token } = await createRoom(settings);
      forget();
      saveHostSeat(code, token);
      pendingCriticKey = criticKey;
      window.scrollTo(0, 0);
      setRoom(code);
    } catch {
      alert('Could not create a game right now. Please try again.');
    }
  }

  return (
    <>
      {room ? <Playing room={room} token={hostSeat()?.token || ''} onExit={() => { forget(); location.reload(); }} />
        : <Setup onCreate={create} />}
      {offer && !room && (
        <div className="modal-overlay active" id="modal-overlay">
          <div className="modal" id="modal-content">
            <h2><Icon name="tree" /> Welcome Back!</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              You have an active game: <strong style={{ color: 'var(--accent-gold)' }}>{offer}</strong>
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <button className="btn btn-primary" style={{ width: '100%' }} onClick={() => { setRoom(offer); setOffer(null); }}><Icon name="swap" /> Rejoin Game</button>
              <button className="btn btn-secondary" style={{ width: '100%' }} onClick={() => { forget(); setOffer(null); }}>🆕 Create New Game</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/** The host console's own view of the room, live. */
function Playing({ room, token, onExit }: { room: string; token: string; onExit: () => void }) {
  const { client, game, offset, fatal, status } = useRoom(room, token);
  const sentKey = useRef(false);
  useEffect(() => {
    if (sentKey.current || !pendingCriticKey) return;
    sentKey.current = true;
    client.setCriticKey(pendingCriticKey);   // goes once the socket is open
    pendingCriticKey = '';
  }, [client]);
  useHostMoveSignal(fatal ? null : game, offset);
  if (fatal) return <div className="card" style={{ margin: '2rem auto', maxWidth: 420, textAlign: 'center' }}><p>{fatal}</p><button className="btn btn-primary" onClick={onExit}>Start over</button></div>;
  const banner = <ConnectionBanner status={status} loaded={!!game} />;
  if (!game) return banner;
  return (
    <OffsetProvider offset={offset}>
      {banner}
      <HostProvider room={room} game={game} client={client}><GameScreen onExit={onExit} /></HostProvider>
    </OffsetProvider>
  );
}

/** In the Host tab of the host's phone: tell the page around it whether the game is waiting on the
 *  host (it lights a dot on the tab). Checked on every change, and every second for time-outs. */
function useHostMoveSignal(game: Game | null, offset: number) {
  const framed = window.parent !== window;
  const [, tick] = useState(0);
  useEffect(() => {
    if (!framed) return;
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [framed]);
  const waiting = !!game && framed && hostsMove(game, Date.now() + offset);
  useEffect(() => {
    if (framed) window.parent.postMessage({ jb: 'host-move', waiting }, location.origin);
  }, [framed, waiting]);
}
