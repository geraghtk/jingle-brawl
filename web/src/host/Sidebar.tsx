// The host sidebar: Game (controls, stats, quick actions) · Event log · Admin.
import { useMemo, useState } from 'react';
import qrcode from 'qrcode-generator';
import { useJoinOrigin } from '../net/lan';
import { pick, rng } from '../core/rng';
import { giftless, wrappedGifts } from '../core/rules';
import { MISFIT } from '../core/types';
import { useHost } from './HostContext';
import { useModal } from './Modal';
import { Glyphs, Icon } from '../ui/icons';
import '../ui/connection.css';

type Tab = 'game' | 'events' | 'admin';

export function Sidebar({ onExit }: { onExit: () => void }) {
  const [tab, setTab] = useState<Tab>('game');
  return (
    <div className="sidebar">
      <div className="sidebar-tab-bar">
        {(['game', 'events', 'admin'] as Tab[]).map((t) => (
          <button key={t} type="button" className={`sidebar-tab-btn${tab === t ? ' active' : ''}`} data-tab={t} onClick={() => setTab(t)}>
            {t === 'game' ? 'Game' : t === 'events' ? 'Event log' : 'Admin'}
          </button>
        ))}
      </div>
      <div className="sidebar-tab-panels">
        <div id="tab-game" className={`sidebar-tab-panel${tab === 'game' ? ' active' : ''}`}><GameTab onExit={onExit} /><TvCard /><SeatCard /><CriticCard /></div>
        <div id="tab-events" className={`sidebar-tab-panel${tab === 'events' ? ' active' : ''}`}><EventLog /></div>
        <div id="tab-admin" className={`sidebar-tab-panel${tab === 'admin' ? ' active' : ''}`}><AdminPanel /></div>
      </div>
    </div>
  );
}

/** Where the shared screen comes from: open this on whatever is cast to the TV. */
function TvCard() {
  const { room } = useHost();
  const origin = useJoinOrigin();
  const url = `${origin()}/tv.html?room=${room}`;
  const svg = useMemo(() => { const qr = qrcode(0, 'M'); qr.addData(url); qr.make(); return qr.createSvgTag({ cellSize: 3, margin: 2, scalable: true }); }, [url]);
  return (
    <div className="card" id="tv-card" data-testid="tv-card">
      <div className="card-title"><Icon name="eye" /> The TV screen</div>
      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Open this on the device showing the TV (it shows no controls and no secrets):</div>
      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
        <div style={{ width: 84, flexShrink: 0 }} dangerouslySetInnerHTML={{ __html: svg }} />
        <a href={url} target="_blank" rel="noreferrer" data-testid="tv-link" style={{ wordBreak: 'break-all', fontWeight: 700 }}>{url}</a>
      </div>
    </div>
  );
}

/** A player's phone died (or they want to switch): a link that puts them back in their seat on
 *  another phone, with their chips and gift. Anyone holding it plays as them, so it is shown, not sent. */
function SeatCard() {
  const { game, room, client } = useHost();
  const origin = useJoinOrigin();
  const [pid, setPid] = useState('');
  const [link, setLink] = useState<{ pid: string; url: string } | null>(null);
  const [error, setError] = useState('');
  const players = Object.entries(game.players || {});
  const svg = useMemo(() => {
    if (!link) return '';
    const qr = qrcode(0, 'M'); qr.addData(link.url); qr.make();
    return qr.createSvgTag({ cellSize: 3, margin: 2, scalable: true });
  }, [link]);
  if (!players.length) return null;
  const make = async () => {
    setError(''); setLink(null);
    const token = await client.seatToken(pid);
    if (!token) return setError('Could not make a link. Try again in a moment.');
    const name = game.players?.[pid]?.name || '';
    setLink({ pid, url: `${origin()}/player.html?room=${room}&pid=${encodeURIComponent(pid)}&name=${encodeURIComponent(name)}&seat=${encodeURIComponent(token)}` });
  };
  return (
    <div className="card" id="seat-card" data-testid="seat-card">
      <div className="card-title"><Icon name="swap" /> A phone died?</div>
      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Put a player back in their seat on another phone (their chips and gift come along):</div>
      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <select className="form-input" value={pid} onChange={(e) => { setPid(e.target.value); setLink(null); }} data-testid="seat-player" style={{ flex: 1 }}>
          <option value="">Who?</option>
          {players.map(([id, p]) => <option key={id} value={id}>{p.name}</option>)}
        </select>
        <button className="btn btn-secondary" disabled={!pid} onClick={make} data-testid="seat-make">Make a link</button>
      </div>
      {error && <div style={{ color: 'var(--accent-red)', fontSize: '0.85rem', marginTop: '0.5rem' }}>{error}</div>}
      {link && link.pid === pid && (
        <>
          <div className="seat-link">
            <div style={{ width: 96, flexShrink: 0 }} dangerouslySetInnerHTML={{ __html: svg }} />
            <a href={link.url} target="_blank" rel="noreferrer" data-testid="seat-link">{link.url}</a>
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>Scan it with {game.players?.[pid]?.name}'s new phone. Anyone with this link plays as them, so only show it to them.</div>
        </>
      )}
    </div>
  );
}

/** AI Draw Off's critic: is it ready, and the host's own key (sent to the room's memory, never stored). */
function CriticCard() {
  const { game, cfg, client } = useHost();
  const [key, setKey] = useState('');
  const backend = game.settings?.aiCriticBackend;
  if (!cfg.duelAppGames.includes('draw_off')) return null;
  const c = client.critic;
  return (
    <div className="card" id="critic-card" data-testid="critic-card">
      <div className="card-title"><Icon name="critic" /> AI critic</div>
      <div data-testid="critic-status" style={{ marginBottom: '0.5rem', fontSize: '0.9rem' }}>
        {c?.ready ? <>Ready ({c.backend})</> : c?.error ? <>Not working: {c.error}</> : c ? 'Checking the key…' : 'No key: AI Draw Off falls back to you picking the winner'}
      </div>
      {backend !== 'local' && backend !== 'ollama' && (
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <input type="password" className="input" autoComplete="off" placeholder="Your API key" value={key} onChange={(e) => setKey(e.target.value)} style={{ flex: 1 }} />
          <button className="btn btn-primary" disabled={!key.trim()} onClick={() => { client.setCriticKey(key.trim()); setKey(''); }}>Use</button>
          {c?.ready && <button className="btn btn-secondary" onClick={() => client.setCriticKey(null)}>Forget</button>}
        </div>
      )}
    </div>
  );
}

function GameTab({ onExit }: { onExit: () => void }) {
  const { game, send } = useHost();
  const setModal = useModal();
  const players = Object.keys(game.players || {}).length;
  const wrapped = wrappedGifts(game);
  const bag = giftless(game).length;
  const busy = !!game.currentTurn;
  const gifts = Object.values(game.gifts || {});
  const canDraw = !busy && wrapped > 0 && bag > 0;
  const canEndgame = wrapped === 0 && bag > 0 && game.phase !== 'endgame' && !busy;
  const drawLabel = busy ? '⏳ Turn in Progress...' : wrapped === 0 ? '🎁 All Gifts Opened' : bag === 0 ? '👥 Bag Empty' : "🎅 Draw from Santa's Sack";

  const draw = () => {
    // The prompt names who's drawn: preview with the same seed the plan then draws with
    const seed = Date.now() ^ Math.floor(Math.random() * 1e9);
    const who = game.players?.[pick(giftless(game), rng(seed))]?.name || 'Someone';
    const giftName = prompt(`🎅 ${who} was drawn!\n\nEnter the gift they're opening:`);
    if (!giftName) return;
    send({ type: 'draw', giftName: giftName.slice(0, 80), seed });
  };

  return (
    <>
      <div className="card" id="controls-card">
        <div className="card-title">Controls</div>
        {game.phase === 'lobby' ? (
          <div id="lobby-controls" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', margin: '0.25rem 0' }}>Waiting in the lobby. Start the game from the room-code card.</p>
          </div>
        ) : (
          <div id="game-controls" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <button className="btn btn-gold" id="btn-draw-player" data-testid="draw" disabled={!canDraw} onClick={draw} style={{ fontSize: '1.1rem', padding: '1.25rem' }}>{drawLabel}</button>
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
              <button className="btn btn-secondary" style={{ flex: 1 }} onClick={() => setModal(<AddGiftModal onClose={() => setModal(null)} />)}>Add Gift</button>
              <button className="btn btn-secondary" id="btn-endgame" data-testid="endgame" disabled={!canEndgame} style={{ flex: 1 }} onClick={() => send({ type: 'endgame' })}>
                <Glyphs>{game.phase === 'endgame' ? '🎰 In Endgame' : 'Endgame'}</Glyphs>
              </button>
            </div>
          </div>
        )}
      </div>
      <div className="card" id="stats-card">
        <div className="card-title">Game stats</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
          {([['stat-wrapped', wrapped, 'Wrapped', 'gold'], ['stat-opened', gifts.filter((g) => g.ownerId !== MISFIT).length, 'Opened', 'green'],
            ['stat-in-bag', players ? bag : 0, 'In Bag', 'blue'], ['stat-misfit', gifts.filter((g) => g.ownerId === MISFIT).length, 'Misfit', 'red']] as const).map(([id, v, label, c]) => (
            <div key={id} style={{ background: 'var(--bg-card-hover)', padding: '0.75rem', borderRadius: 8, textAlign: 'center' }}>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: `var(--accent-${c})` }} id={id}>{v}</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{label}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="card">
        <div className="card-title">Quick actions</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <button className="btn btn-secondary" onClick={() => window.open('quick-reference.html', '_blank')}>Quick Reference</button>
          <button className="btn btn-secondary" onClick={() => { if (confirm('Exit this game? You can rejoin later if you refresh.')) onExit(); }}>Exit Game</button>
          <button className="btn btn-danger" data-testid="end-game" onClick={() => { if (confirm('Are you sure you want to end the game?')) send({ type: 'admin_end_game' }); }}>End Game for All</button>
        </div>
      </div>
    </>
  );
}

function AddGiftModal({ onClose }: { onClose: () => void }) {
  const { game, send } = useHost();
  const [name, setName] = useState('');
  const [owner, setOwner] = useState<string>(MISFIT);
  return (
    <>
      <h2>➕ Add Gift Manually</h2>
      <div style={{ textAlign: 'left', marginTop: '1rem' }}>
        <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Gift Name</label>
        <input type="text" id="manual-gift-name" value={name} onChange={(e) => setName(e.target.value)} style={{ width: '100%', padding: '0.75rem', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-card-hover)' }} />
      </div>
      <div style={{ textAlign: 'left', marginTop: '1rem' }}>
        <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Owner</label>
        <select id="manual-gift-owner" value={owner} onChange={(e) => setOwner(e.target.value)} style={{ width: '100%', padding: '0.75rem', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-card-hover)' }}>
          <option value={MISFIT}>🧸 Misfit Pile</option>
          {Object.entries(game.players || {}).map(([id, p]) => <option key={id} value={id}>{p.name}</option>)}
        </select>
      </div>
      <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1.5rem' }}>
        <button className="btn btn-primary" onClick={async () => { if (await send({ type: 'admin_add_gift', name, ownerId: owner })) onClose(); }}>Add Gift</button>
        <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
      </div>
    </>
  );
}

function EventLog() {
  const { log, debug, setDebug, addEvent } = useHost();
  return (
    <div className="card">
      <div className="card-title">Event log</div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
        <span />
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-muted)', cursor: 'pointer' }}>
          <input type="checkbox" id="debug-log-toggle" checked={debug} onChange={(e) => { setDebug(e.target.checked); if (e.target.checked) addEvent('🔍 Detailed debug logging enabled', 'gift'); }} />
          Detailed Debug Log
        </label>
      </div>
      <div className="event-log" id="event-log">
        {log.map((e, i) => (
          <div key={`${e.at}-${i}`} className={`event ${e.kind}`}>
            <div><Glyphs>{e.text}</Glyphs></div>
            <div className="time">{new Date(e.at).toLocaleTimeString()}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

const field = { padding: '0.25rem', borderRadius: 4, background: 'var(--bg-card)', border: '1px solid var(--border)' };

function AdminPanel() {
  const { game, send } = useHost();
  const [open, setOpen] = useState(false);
  const [pid, setPid] = useState('');
  const [chips, setChips] = useState(0);
  const [gid, setGid] = useState('');
  const [edit, setEdit] = useState({ name: '', naughty: 0, owner: '' });
  const [newGift, setNewGift] = useState('');
  const players = Object.entries(game.players || {});
  const gifts = Object.entries(game.gifts || {});

  const loadGift = (id: string) => {
    setGid(id);
    const g = game.gifts?.[id];
    if (g) setEdit({ name: g.name, naughty: g.naughtyLevel || 0, owner: g.ownerId !== MISFIT ? g.ownerId : '' });
  };

  return (
    <>
      <div className="card" id="debug-panel" style={{ border: '2px dashed var(--accent-gold)' }}>
        <div className="card-title"><Icon name="critic" /> Bots &amp; auto-pilot</div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Replaced by the agent play-testing harness: <code>npm run agents:watch</code> plays a whole game with persona agents on real phones.
        </div>
      </div>
      <div className="card" id="admin-panel" style={{ border: '2px solid var(--accent-red)' }}>
        <div className="card-title" style={{ cursor: 'pointer' }} onClick={() => setOpen(!open)}>⚙️ Host Admin <span id="admin-toggle-icon">{open ? '▲' : '▼'}</span></div>
        {open && (
          <div id="admin-content" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <button className="btn btn-secondary" title="Force skip current phase" onClick={() => { if (confirm('Skip current phase? This may cause unexpected state.')) send({ type: 'admin_reset_turn' }); }}>⏭️ Skip Phase</button>
              <button className="btn btn-secondary" title="Reset to waiting state" onClick={() => { if (confirm('Reset current turn? All pending actions will be cleared.')) send({ type: 'admin_reset_turn' }); }}><Icon name="swap" /> Reset Turn</button>
              <button className="btn btn-secondary" title="Undo the last duel" onClick={() => {
                if (!game.undo) return alert('No state to undo. Undo captures state before each duel is resolved.');
                if (confirm('Restore players and gifts to before the last duel?')) send({ type: 'admin_undo_duel' });
              }}>↩️ Undo</button>
            </div>

            <div style={{ background: 'var(--bg-card-hover)', padding: '0.75rem', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>👤 Edit Player</div>
              <select id="admin-player-select" value={pid} style={{ width: '100%', padding: '0.5rem', borderRadius: 6, marginBottom: '0.5rem' }}
                onChange={(e) => { setPid(e.target.value); setChips(game.players?.[e.target.value]?.chips ?? 0); }}>
                <option value="">Select player...</option>
                {players.map(([id, p]) => <option key={id} value={id}>{p.name} (🪙{p.chips})</option>)}
              </select>
              {pid && (
                <div id="admin-player-controls">
                  <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', flex: 1 }}><Icon name="chip" /> <input type="number" id="admin-player-chips" min={0} max={99} value={chips} onChange={(e) => setChips(parseInt(e.target.value) || 0)} style={{ ...field, width: 60 }} /></label>
                    <button className="btn btn-primary" style={{ padding: '0.25rem 0.75rem' }} onClick={() => send({ type: 'admin_set_chips', pid, chips })}>Save</button>
                  </div>
                  <button className="btn btn-secondary" style={{ width: '100%', fontSize: '0.75rem', padding: '0.25rem' }} onClick={() => send({ type: 'admin_remove_player_gift', pid })}><Icon name="gift" /> Remove Gift</button>
                </div>
              )}
            </div>

            <div style={{ background: 'var(--bg-card-hover)', padding: '0.75rem', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}><Icon name="gift" /> Edit Gift</div>
              <select id="admin-gift-select" value={gid} style={{ width: '100%', padding: '0.5rem', borderRadius: 6, marginBottom: '0.5rem' }} onChange={(e) => loadGift(e.target.value)}>
                <option value="">Select gift...</option>
                {gifts.map(([id, g]) => <option key={id} value={id}>{g.name} ({g.ownerId === MISFIT ? 'Misfit' : game.players?.[g.ownerId]?.name || 'Unknown'}, 😈{g.naughtyLevel || 0})</option>)}
              </select>
              {gid && game.gifts?.[gid] && (
                <div id="admin-gift-controls">
                  <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <input type="text" id="admin-gift-name" value={edit.name} onChange={(e) => setEdit({ ...edit, name: e.target.value })} style={{ ...field, flex: 1 }} />
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}><Icon name="naughty" /> <input type="number" id="admin-gift-naughty" min={0} max={20} value={edit.naughty} onChange={(e) => setEdit({ ...edit, naughty: parseInt(e.target.value) || 0 })} style={{ ...field, width: 50 }} /></label>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <select id="admin-gift-owner" value={edit.owner} onChange={(e) => setEdit({ ...edit, owner: e.target.value })} style={{ ...field, flex: 1 }}>
                      <option value="">No owner (Misfit)</option>
                      {players.map(([id, p]) => <option key={id} value={id}>{p.name}</option>)}
                    </select>
                    <button className="btn btn-primary" style={{ padding: '0.25rem 0.75rem' }} onClick={() => send({ type: 'admin_save_gift', giftId: gid, name: edit.name.trim() || undefined, naughtyLevel: edit.naughty, ownerId: edit.owner || null })}>Save</button>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button className="btn btn-secondary" data-testid="admin-to-misfit" style={{ flex: 1, fontSize: '0.75rem', padding: '0.25rem' }} onClick={() => send({ type: 'admin_move_misfit', giftId: gid })}>📦 → Misfit Pile</button>
                    <button className="btn btn-secondary" style={{ flex: 1, fontSize: '0.75rem', padding: '0.25rem', background: 'var(--accent-red)' }}
                      onClick={() => { if (confirm(`Delete "${game.gifts?.[gid]?.name}"? This cannot be undone.`)) { send({ type: 'admin_delete_gift', giftId: gid }); setGid(''); } }}>🗑️ Delete</button>
                  </div>
                </div>
              )}
            </div>

            <div style={{ background: 'var(--bg-card-hover)', padding: '0.75rem', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>➕ Add Gift</div>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <input type="text" id="admin-new-gift-name" placeholder="Gift name" value={newGift} onChange={(e) => setNewGift(e.target.value)} style={{ ...field, flex: 1, padding: '0.5rem' }} />
                <button className="btn btn-primary" onClick={async () => { if (await send({ type: 'admin_add_gift', name: newGift, ownerId: MISFIT })) setNewGift(''); }}>Add</button>
              </div>
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'center' }}>⚠️ Use with caution - changes are immediate</div>
          </div>
        )}
      </div>
    </>
  );
}

