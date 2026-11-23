// The music under the lobby and the mini-games, on the big screens only: the TV plays it unless it is
// switched off there, the host's console only if it is switched on (both open in one room would play
// it twice). One cell of the top strip: it picks the track from the game (ui/music.ts `trackFor`) and
// is the on/off button. The console inside the host's phone (the Host tab) has none: phones stay quiet.
import { useEffect, useState } from 'react';
import { useServerNow } from '../net/hooks';
import { musicBlocked, musicOn, setMusicDefault, setMusicOn, setTrack, trackFor } from '../ui/music';
import { useHost } from './HostContext';

const EMBEDDED = new URLSearchParams(location.search).get('embedded') === '1';

export function MusicToggle() {
  const { game, tv } = useHost();
  const now = useServerNow(500);                                  // to hear a game's last seconds coming
  const [on, setOn] = useState(() => { setMusicDefault(tv); return musicOn(); });
  const track = EMBEDDED ? null : trackFor(game, now);
  useEffect(() => { setTrack(track); }, [track]);
  useEffect(() => () => setTrack(null), []);
  if (EMBEDDED) return null;
  // a page nobody has touched yet may not make a sound: say so (that first click only wakes it)
  const blocked = on && musicBlocked();
  return (
    <button type="button" className="cell minor music-toggle" data-testid="music-toggle" aria-pressed={on}
      aria-label={on ? 'Music on' : 'Music off'} title={blocked ? 'Click anywhere to start the music' : on ? 'Turn the music off' : 'Turn the music on'}
      onClick={() => { if (blocked) return; setMusicOn(!on); setOn(!on); }}>
      <span className="k">Music</span>
      <span className="v f6-num">{blocked ? 'Click to start' : on ? 'On' : 'Off'}</span>
    </button>
  );
}
