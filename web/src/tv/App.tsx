// The shared screen (tv.html?room=CODE): open it on whatever is cast to or plugged into the TV.
// It joins with the room code only, so the room sends it the public view: no controls, no
// secrets. The host runs the game from their own phone.
import { useState } from 'react';
import { HostProvider } from '../host/HostContext';
import { GameScreen } from '../host/GameScreen';
import { OffsetProvider, useRoom } from '../net/hooks';

const fromUrl = (new URLSearchParams(location.search).get('room') || '').trim().toUpperCase();

export function App() {
  const [room, setRoom] = useState(fromUrl);
  const [typed, setTyped] = useState('');
  if (room) return <Screen room={room} />;
  return (
    <div className="card" style={{ margin: '4rem auto', maxWidth: 420, textAlign: 'center' }} data-testid="tv-room-form">
      <h2>Show a game on this screen</h2>
      <p style={{ color: 'var(--text-secondary)' }}>Enter the room code from the host's phone.</p>
      <input className="input room-code-input" maxLength={4} value={typed} onChange={(e) => setTyped(e.target.value.toUpperCase())} placeholder="ROOM" />
      <button className="btn btn-primary" style={{ marginTop: '1rem', width: '100%' }} disabled={typed.trim().length !== 4}
        onClick={() => { history.replaceState(null, '', `?room=${typed.trim()}`); setRoom(typed.trim()); }}>Show it</button>
    </div>
  );
}

function Screen({ room }: { room: string }) {
  const { client, game, offset, fatal } = useRoom(room);
  if (fatal) return <div className="card" style={{ margin: '4rem auto', maxWidth: 420, textAlign: 'center' }}><p>{fatal}</p><a className="btn btn-primary" href="tv.html">Another room</a></div>;
  if (!game) return null;
  return (
    <OffsetProvider offset={offset}>
      <HostProvider room={room} game={game} client={client} tv><GameScreen onExit={() => {}} /></HostProvider>
    </OffsetProvider>
  );
}
