// Host (TV) app: settings → create a game (atomic room-code claim) → the game screen.
// A refreshed host is offered a rejoin (?autoLoad=1 skips the prompt).
import { runTransaction } from 'firebase/database';
import { useEffect, useState } from 'react';
import { newGame } from '../core/actions';
import type { Settings } from '../core/types';
import { signedIn } from '../firebase/app';
import { gameRef, readGame, useGame } from '../firebase/game';
import { GameScreen } from './GameScreen';
import { HostProvider } from './HostContext';
import { Setup } from './Setup';

const K = { room: 'jingleBrawl_host_room', id: 'jingleBrawl_host_id' };
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
    if (!saved || !hostId) return;
    readGame(saved).then(async (g) => {
      if (g?.hostId !== hostId) { localStorage.removeItem(K.room); localStorage.removeItem(K.id); return; }
      if (autoLoad) { await signedIn(); setRoom(saved); } else setOffer(saved);
    });
  }, []);

  async function create(settings: Settings, openaiKey: string) {
    const uid = await signedIn();
    if (!uid) { alert('Could not sign in — check your connection and try again'); return; }
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
    alert('Could not create a game right now — please try again.');
  }

  return (
    <>
      {room ? <Playing room={room} onExit={() => { localStorage.removeItem(K.room); localStorage.removeItem(K.id); location.reload(); }} />
        : <Setup onCreate={create} />}
      {offer && !room && (
        <div className="modal-overlay active" id="modal-overlay">
          <div className="modal" id="modal-content">
            <h2>🎄 Welcome Back!</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              You have an active game: <strong style={{ color: 'var(--accent-gold)' }}>{offer}</strong>
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <button className="btn btn-primary" style={{ width: '100%' }} onClick={async () => { await signedIn(); setRoom(offer); setOffer(null); }}>🔄 Rejoin Game</button>
              <button className="btn btn-secondary" style={{ width: '100%' }} onClick={() => { localStorage.removeItem(K.room); localStorage.removeItem(K.id); setOffer(null); }}>🆕 Create New Game</button>
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
