// Phone app: join / rejoin, then the game screen for this player. Joining returns a session
// token from the room server; a refresh rejoins with it.
import { useEffect, useState } from 'react';
import { cleanName } from '../core/actions';
import { AVATARS } from '../core/avatars';
import { joinRoom, lobby } from '../net/api';
import { OffsetProvider, useRoom } from '../net/hooks';
import { GameScreen } from './GameScreen';
import { PlayerProvider, useToast } from './PlayerContext';
import { clearSession, loadSession, saveSession } from './session';
import type { Session } from './session';
import { Glyphs, Icon } from '../ui/icons';
import { ConnectionBanner } from '../ui/ConnectionBanner';
import { SeatTabs } from './SeatTabs';
import { hosts } from '../net/seats';

const params = new URLSearchParams(location.search);

function Hero() {
  return (
    <div className="join-hero">
      <div className="logo-wrap"><img src="ydlogo2.png" alt="Yuel Duel" /></div>
      <h1 className="join-title fz-riso">Yuel <span style={{ color: 'var(--accent)' }}>Duel</span></h1>
      <div className="join-tagline">A game of <span>high stakes</span>, <span>low blows</span>, and <span>questionable gifts</span>.</div>
    </div>
  );
}

export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [saved, setSaved] = useState<Session | null | undefined>(undefined); // undefined = checking
  const [msg, show, toast] = useToast();

  // A saved session whose player still exists → offer to rejoin
  useEffect(() => {
    const seat = params.get('seat'), pid = params.get('pid'), room = params.get('room');
    if (seat && pid && room) {
      const s = { room: room.toUpperCase(), playerId: pid, name: params.get('name') || 'Player', token: seat };
      saveSession(s);
      history.replaceState(null, '', location.pathname);   // the link is a key: off the address bar
      setSaved(null);
      setSession(s);
      return;
    }
    const s = loadSession();
    if (!s) { setSaved(null); return; }
    // the room checks the token when the phone connects; here: is the room still there?
    // ?rejoin=1 (the host switching back from their console): straight back in, no question
    lobby(s.room).then((l) => {
      if (!l.exists || !s.token) { clearSession(); setSaved(null); return; }
      if (params.get('rejoin') === '1') { setSaved(null); setSession(s); } else setSaved(s);
    })
      .catch(() => setSaved(null));
  }, []);

  async function join(room: string, rawName: string, age: number | null, avatar: string | null = null) {
    room = room.trim().toUpperCase();
    const name = cleanName(rawName);
    if (room.length !== 4) return toast('Please enter a valid 4-letter room code');
    if (!name) return toast('Please enter your name');
    try {
      const { pid, token } = await joinRoom(room, { name, age, avatar });
      const s = { room, playerId: pid, name, token };
      saveSession(s);
      setSession(s);
      toast('Joined the game! 🎄');
    } catch (e) {
      toast((e as Error).message || 'Could not join. Please try again.');
    }
  }

  function rejoin(s: Session) {
    setSession(s);
    toast('Reconnected! 🎄');
  }

  return (
    <>
      <div className="container">
        {session ? <Playing session={session} toast={toast} />
          : saved === undefined ? null
          : saved ? <Rejoin s={saved} onRejoin={() => rejoin(saved)} onOther={() => { clearSession(); setSaved(null); }} />
          : <Join onJoin={join} />}
      </div>
      <div className={`toast${show ? ' show' : ''}`} id="toast"><Glyphs>{msg}</Glyphs></div>
    </>
  );
}

function Join({ onJoin }: { onJoin: (room: string, name: string, age: number | null, avatar: string | null) => void }) {
  const [room, setRoom] = useState((params.get('room') || '').toUpperCase());
  const [name, setName] = useState(params.get('name') || '');
  // Only under-16s are asked their age: the youngest duellist picks the duel (V2 Under-16 Exception)
  const [young, setYoung] = useState(false);
  const [age, setAge] = useState('');
  const ageValue = young && age ? Number(age) : null;
  const [avatar, setAvatar] = useState<string | null>(null);
  // avatars already at the table (once the room code is in): one each, so they're greyed out
  const [taken, setTaken] = useState<Set<string>>(new Set());
  useEffect(() => {
    if (room.trim().length !== 4) { setTaken(new Set()); return; }
    let live = true;
    lobby(room.trim()).then((l) => l.avatars || []).then((list) => {
      if (!live) return;
      const ids = new Set(list);
      setTaken(ids);
      setAvatar((a) => (a && ids.has(a) ? null : a));
    }).catch(() => {});
    return () => { live = false; };
  }, [room]);
  useEffect(() => {
    if (params.get('autoJoin') === '1' && room && name) onJoin(room, name, null, params.get('avatar'));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div id="join-screen" className="join-screen">
      <Hero />
      <div className="join-form">
        <input type="text" id="room-code-input" className="room-code-input" placeholder="ROOM" maxLength={4} autoComplete="off"
          value={room} onChange={(e) => setRoom(e.target.value.toUpperCase())} />
        <input type="text" id="player-name-input" placeholder="Your name" autoComplete="off" value={name} onChange={(e) => setName(e.target.value)} />
        <fieldset className="avatar-pick" data-testid="avatar-pick">
          <legend>Pick your avatar <small>{avatar ? AVATARS.find((a) => a.id === avatar)?.label : "or we'll pick one for you"}</small></legend>
          <div className="avatar-grid">
            {AVATARS.map((a) => (
              <button key={a.id} type="button" className={`avatar-opt${avatar === a.id ? ' on' : ''}`} style={{ background: a.color }}
                aria-label={taken.has(a.id) ? `${a.label} (taken)` : a.label} aria-pressed={avatar === a.id} data-avatar={a.id}
                disabled={taken.has(a.id)} onClick={() => setAvatar(avatar === a.id ? null : a.id)}>
                <img src={`art/avatars/${a.id}.webp`} alt="" draggable={false} />
              </button>
            ))}
          </div>
        </fieldset>
        <label className="young-toggle">
          <input type="checkbox" data-testid="under-16" checked={young} onChange={(e) => setYoung(e.target.checked)} />
          <span>I'm under 16 <small>you'll get to pick your duels</small></span>
        </label>
        {young && (
          <label className="young-age">
            <span>How old are you?</span>
            <select data-testid="age" value={age} onChange={(e) => setAge(e.target.value)}>
              <option value="">Choose</option>
              {Array.from({ length: 11 }, (_, i) => 5 + i).map((a) => <option key={a} value={a}>{a}</option>)}
            </select>
          </label>
        )}
        <button className="btn btn-primary" data-testid="join" disabled={young && !age} onClick={() => onJoin(room, name, ageValue, avatar)}>🎮 Join Game</button>
      </div>
      <p style={{ marginTop: '2rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>Ask the host for the room code, or <a href="host.html" data-testid="host-a-game">host a new game</a></p>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>No accounts, nothing kept after the party. <a href="privacy.html" target="_blank" rel="noreferrer" data-testid="privacy-link">Privacy</a></p>
    </div>
  );
}

function Rejoin({ s, onRejoin, onOther }: { s: Session; onRejoin: () => void; onOther: () => void }) {
  return (
    <div id="join-screen" className="join-screen">
      <Hero />
      <div className="card" style={{ marginTop: 0, maxWidth: 320, textAlign: 'center' }}>
        <h3 style={{ marginBottom: '0.5rem' }}>Welcome back, {s.name}!</h3>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
          Room: <strong style={{ color: 'var(--accent-gold)' }}>{s.room}</strong>
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <button className="btn btn-primary" id="rejoin-btn" data-testid="rejoin" onClick={onRejoin}><Icon name="swap" /> Rejoin Game</button>
          <button className="btn btn-secondary" onClick={onOther}>🆕 Join Different Game</button>
        </div>
      </div>
    </div>
  );
}

/** This phone's own view of the room, live. A refused token means the phone isn't in that game
 *  any more: back to the join screen. */
function Playing({ session, toast }: { session: Session; toast: (m: string) => void }) {
  const { client, game, offset, fatal, status } = useRoom(session.room, session.token);
  useEffect(() => { if (fatal) clearSession(); }, [fatal]);
  if (fatal) {
    return <div className="card" style={{ margin: '2rem auto', maxWidth: 320, textAlign: 'center' }}><p>{fatal}</p><button className="btn btn-primary" onClick={() => location.reload()}>Join a game</button></div>;
  }
  const banner = <ConnectionBanner status={status} loaded={!!game} />;
  if (!game) return banner;
  const leave = () => {
    if (!confirm('Leave this game?')) return;
    client.stop();
    clearSession();
    location.reload();
  };
  return (
    <OffsetProvider offset={offset}>
      {banner}
      <PlayerProvider room={session.room} me={session.playerId} game={game} toast={toast} client={client}>
        {hosts(session.room) ? <SeatTabs><GameScreen onLeave={leave} /></SeatTabs> : <GameScreen onLeave={leave} />}
      </PlayerProvider>
    </OffsetProvider>
  );
}
