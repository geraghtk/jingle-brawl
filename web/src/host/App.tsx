// Host (TV) app: settings → create a game → the game screen. On the room server the server makes
// the room and hands back the host's token; on Firebase the page claims a room code itself.
// A refreshed host is offered a rejoin (?autoLoad=1 skips the prompt).
import { runTransaction } from 'firebase/database';
import { useEffect, useState } from 'react';
import { newGame } from '../core/actions';
import type { Settings } from '../core/types';
import { signedIn } from '../firebase/app';
import { gameRef, readGame, useGame } from '../firebase/game';
import { createRoom, lobby } from '../net/api';
import { OffsetProvider, useRoom } from '../net/hooks';
import { onServer } from '../net/transport';
import { GameScreen } from './GameScreen';
import { HostProvider } from './HostContext';
import { Setup } from './Setup';
import { Icon } from '../ui/icons';

const K = { room: 'jingleBrawl_host_room', id: 'jingleBrawl_host_id', token: 'jingleBrawl_host_token' };
const forget = () => { for (const k of Object.values(K)) localStorage.removeItem(k); };
const autoLoad = new URLSearchParams(location.search).get('autoLoad') === '1';

function roomCode() {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  return Array.from({ length: 4 }, () => chars[Math.floor(Math.random() * chars.length)]).join('');
}

export function App() {
  const [room, setRoom] = useState<string | null>(null);
  const [offer, setOffer] = useState<string | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem(K.room);
    const hostId = localStorage.getItem(K.id);
    if (onServer) {
      if (!saved || !localStorage.getItem(K.token)) return;
      lobby(saved).then((l) => {
        if (!l.exists) return forget();
        if (autoLoad) setRoom(saved); else setOffer(saved);
      }).catch(() => {});
      return;
    }
    if (!saved || !hostId) return;
    readGame(saved).then(async (g) => {
      if (g?.hostId !== hostId) { localStorage.removeItem(K.room); localStorage.removeItem(K.id); return; }
      if (autoLoad) { await signedIn(); setRoom(saved); } else setOffer(saved);
    });
  }, []);

  async function create(settings: Settings, openaiKey: string) {
    if (onServer) {
      try {
        const { code, token } = await createRoom(settings);
        forget();
        localStorage.setItem(K.room, code);
        localStorage.setItem(K.token, token);
        if (openaiKey) sessionStorage.setItem(`jingleBrawl_openai_key_${code}`, openaiKey);
        window.scrollTo(0, 0);
        setRoom(code);
      } catch {
        alert('Could not create a game right now. Please try again.');
      }
      return;
    }
    const uid = await signedIn();
    if (!uid) { alert('Could not sign in. Check your connection and try again.'); return; }
    const hostId = `host_${Date.now()}`;
    const data = newGame({ hostId, hostUid: uid, settings, now: Date.now() });
    for (let attempt = 0; attempt < 10; attempt++) {
      const code = roomCode();
      const { committed } = await runTransaction(gameRef(code), (cur) => (cur === null ? data : undefined));
      if (!committed) continue; // taken — try another code
      localStorage.setItem(K.room, code);
      localStorage.setItem(K.id, hostId);
      if (openaiKey) sessionStorage.setItem(`jingleBrawl_openai_key_${code}`, openaiKey);
      window.scrollTo(0, 0);
      setRoom(code);
      return;
    }
    alert('Could not create a game right now. Please try again.');
  }

  return (
    <>
      {room ? (onServer
        ? <ServerPlaying room={room} token={localStorage.getItem(K.token) || ''} onExit={() => { forget(); location.reload(); }} />
        : <Playing room={room} onExit={() => { forget(); location.reload(); }} />)
        : <Setup onCreate={create} />}
      {offer && !room && (
        <div className="modal-overlay active" id="modal-overlay">
          <div className="modal" id="modal-content">
            <h2><Icon name="tree" /> Welcome Back!</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              You have an active game: <strong style={{ color: 'var(--accent-gold)' }}>{offer}</strong>
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <button className="btn btn-primary" style={{ width: '100%' }} onClick={async () => { if (!onServer) await signedIn(); setRoom(offer); setOffer(null); }}><Icon name="swap" /> Rejoin Game</button>
              <button className="btn btn-secondary" style={{ width: '100%' }} onClick={() => { forget(); setOffer(null); }}>🆕 Create New Game</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function Playing({ room, onExit }: { room: string; onExit: () => void }) {
  const { game } = useGame(room);
  if (!game) return null;
  return <HostProvider room={room} game={game}><GameScreen onExit={onExit} /></HostProvider>;
}

/** On the room server: the host console's own view of the room, live. */
function ServerPlaying({ room, token, onExit }: { room: string; token: string; onExit: () => void }) {
  const { client, game, offset, fatal } = useRoom(room, token);
  if (fatal) return <div className="card" style={{ margin: '2rem auto', maxWidth: 420, textAlign: 'center' }}><p>{fatal}</p><button className="btn btn-primary" onClick={onExit}>Start over</button></div>;
  if (!game) return null;
  return (
    <OffsetProvider offset={offset}>
      <HostProvider room={room} game={game} client={client}><GameScreen onExit={onExit} /></HostProvider>
    </OffsetProvider>
  );
}
