// The setup screen: house settings, then "Create New Game".
import { useState } from 'react';
import type { ReactNode } from 'react';
import { DEFAULT_DUEL_GAMES } from '../core/settings';
import type { DuelApp } from '../core/duelgame';
import type { Settings } from '../core/types';
import { Glyphs, Icon } from '../ui/icons';

const PARTY = ['sketch_off', 'whack_elf'];
const NL = '\n'; // the duel list is one per line

function Select({ id, label, hint, value, onChange, options }: {
  id: string; label: string; hint?: ReactNode; value: string; onChange: (v: string) => void; options: [string, string][];
}) {
  return (
    <div className="form-group">
      <label className="label"><Glyphs>{label}</Glyphs></label>
      <select id={`setting-${id}`} className="select" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
      </select>
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}

export function Setup({ onCreate }: { onCreate: (settings: Settings, criticKey: string) => void }) {
  const [s, setS] = useState({
    'reprisal-depth': '1', 'max-bid': '4', 'max-payout': '2', 'starting-chips': '', 'loser-dividend': 'defenders',
    'spectator-bets': '0', 'vote-window': '30', 'duel-pick': 'defender', 'tax-threshold': '2', 'base-cost': '1', 'tiebreaker-method': 'party_mix', 'party-prize': 'points_chips',
  });
  const [publicChips, setPublicChips] = useState(false);
  const [duelGames, setDuelGames] = useState(DEFAULT_DUEL_GAMES.join(NL));
  const [duelApps, setDuelApps] = useState<DuelApp[]>([]);
  const toggleApp = (a: DuelApp) => setDuelApps((x) => (x.includes(a) ? x.filter((y) => y !== a) : [...x, a]));
  // AI Draw Off's critic: the host's own key goes to the room server once (memory only), never into the game
  const [critic, setCritic] = useState<{ backend: 'openai' | 'gemini' | 'local'; key: string }>({ backend: 'openai', key: '' });
  const set = (k: keyof typeof s) => (v: string) => setS((x) => ({ ...x, [k]: v }));
  const method = s['tiebreaker-method'];

  const create = () => {
    const settings: Settings = {
      publicChips,
      maxReprisalDepth: parseInt(s['reprisal-depth']),
      maxBid: parseInt(s['max-bid']),
      maxWinnerPayout: parseInt(s['max-payout']),
      loserDividendRule: s['loser-dividend'] as Settings['loserDividendRule'],
      spectatorBets: parseInt(s['spectator-bets']),
      voteWindow: parseInt(s['vote-window']),
      duelPick: s['duel-pick'] as Settings['duelPick'],
      duelGames: duelGames.split(NL).map((x) => x.trim()).filter(Boolean).join(NL), // text: an empty list must stay empty
      duelAppGames: duelApps,
      taxThreshold: parseInt(s['tax-threshold']),
      baseCost: parseInt(s['base-cost']),
      tiebreakerMethod: method as Settings['tiebreakerMethod'],
      partyPrize: s['party-prize'] as Settings['partyPrize'],
      ...(duelApps.includes('draw_off') ? { aiCriticBackend: critic.backend } : {}),
      ...(s['starting-chips'] ? { startingChips: parseInt(s['starting-chips']) } : {}),
    };
    onCreate(settings, duelApps.includes('draw_off') && critic.backend !== 'local' ? critic.key.trim() : '');
  };

  return (
    <div id="setup-screen" className="setup-screen">
      <div className="setup-hero">
        <div className="logo-wrap">
          <img src="ydlogo2.png?v=2" alt="Yuel Duel" style={{ display: 'block', width: 'auto', height: 'auto', maxWidth: 420, maxHeight: 260, minHeight: 140, objectFit: 'contain', margin: '0 auto', filter: 'drop-shadow(0 2px 8px rgba(0,0,0,0.1))' }} />
        </div>
        <h1 className="setup-title fz-riso">Yuel <span style={{ color: 'var(--accent)' }}>Duel</span></h1>
        <div className="setup-tagline" id="setup-tagline" style={{ fontSize: '1.25rem', color: 'var(--muted)', fontStyle: 'italic', margin: '1rem 0', padding: '0.75rem 1rem', background: 'var(--bg2)', border: '1px solid var(--line)', borderRadius: 16, fontFamily: 'var(--f6-body)' }}>
          A game of <span style={{ color: 'var(--accent)', fontWeight: 700 }}>high stakes</span>, <span style={{ color: 'var(--accent)', fontWeight: 700 }}>low blows</span>, and <span style={{ color: 'var(--accent)', fontWeight: 700 }}>questionable gifts</span>.
        </div>
      </div>

      <div className="card setup-card" style={{ maxWidth: 420 }}>
        <h3 className="card-title" style={{ marginBottom: 'var(--space-lg)' }}>Game Settings</h3>
        <Select id="reprisal-depth" label="🦌 Reprisal Chain Depth" hint="How many times can a reprisal trigger another reprisal?" value={s['reprisal-depth']} onChange={set('reprisal-depth')}
          options={[['0', 'None (no reprisals)'], ['1', 'Single (no chaining)'], ['2', 'Chain once (2 max)'], ['3', 'Chain twice (3 max)']]} />
        <Select id="max-bid" label="🎯 Maximum Bid" value={s['max-bid']} onChange={set('max-bid')} options={[['3', '3 chips'], ['4', '4 chips'], ['5', '5 chips'], ['99', 'Unlimited']]} />
        <Select id="max-payout" label="🎅 Santa Tax (Max Winner Payout)" hint="Extra chips go to the North Pole Fund (bank)" value={s['max-payout']} onChange={set('max-payout')}
          options={[['1', '1 chip (heavy tax)'], ['2', '2 chips (moderate)'], ['3', '3 chips (light tax)'], ['99', 'No tax (winner gets all)']]} />
        <Select id="starting-chips" label="🪙 Starting Chips" value={s['starting-chips']} onChange={set('starting-chips')}
          options={[['', 'Variable (based on player count)'], ['8', '8 chips'], ['10', '10 chips'], ['12', '12 chips'], ['15', '15 chips'], ['20', '20 chips']]} />
        <Select id="loser-dividend" label="🎁 Loser Dividend" hint={'"Defenders only" makes attacking riskier'} value={s['loser-dividend']} onChange={set('loser-dividend')}
          options={[['everyone', 'Everyone who loses (+1)'], ['defenders', 'Defenders only (+1)'], ['nobody', 'Nobody (harsh!)']]} />
        <Select id="spectator-bets" label="🎲 Spectator Bets (experiment)" hint="Right callers split the wrong callers' chips. Correct calls always add to the Party Score." value={s['spectator-bets']} onChange={set('spectator-bets')}
          options={[['0', 'Off: voting is free'], ['1', 'On: each duel vote stakes 1 chip']]} />
        <Select id="duel-pick" label="🎡 Picking the Duel" value={s['duel-pick']} onChange={set('duel-pick')}
          hint="Yield duels and tiebreaks always spin the wheel: nobody is really defending."
          options={[['defender', 'The defender picks (or spins)'], ['wheel', 'Always spin the wheel'], ['off', "Off: the duellists agree on one"]]} />
        {s['duel-pick'] !== 'off' && (
          <div className="form-group" id="duel-games-setting">
            <label className="label" htmlFor="setting-duel-games">🤜 Duels you'll play in the room</label>
            <textarea id="setting-duel-games" className="input duel-games-input" rows={6} value={duelGames} onChange={(e) => setDuelGames(e.target.value)} />
            <div className="hint">One per line. These go on the wheel and in the defender's list.</div>
            <div className="duel-app-games">
              {(['whack_elf', 'sketch_off'] as DuelApp[]).map((a) => (
                <label key={a} className="duel-app-game">
                  <input type="checkbox" id={`setting-duel-app-${a}`} checked={duelApps.includes(a)} onChange={() => toggleApp(a)} />
                  <span><Glyphs>{a === 'whack_elf' ? '🔨 Whack-an-Elf' : '✏️ Sketch-Off'}</Glyphs> <small>on phones; everyone plays, the better-placed duellist wins</small></span>
                </label>
              ))}
              <label className="duel-app-game">
                <input type="checkbox" id="setting-duel-app-draw_off" checked={duelApps.includes('draw_off')} onChange={() => toggleApp('draw_off')} />
                <span><Glyphs>🎨 AI Draw Off</Glyphs> <small>the two duellists draw; an AI art critic picks the winner</small></span>
              </label>
              {duelApps.includes('draw_off') && (
                <div id="ai-critic-setting" className="ai-critic-setting">
                  <select id="setting-ai-critic-backend" className="select" value={critic.backend} onChange={(e) => setCritic({ ...critic, backend: e.target.value as typeof critic.backend })}>
                    <option value="openai">OpenAI (your API key)</option>
                    <option value="gemini">Google Gemini (your API key)</option>
                    <option value="local">The local critic on this PC (Ollama, free)</option>
                  </select>
                  {critic.backend !== 'local' ? <>
                    <input type="password" id="setting-ai-critic-key" className="input" autoComplete="off" placeholder={critic.backend === 'openai' ? 'sk-...' : 'Gemini API key'}
                      value={critic.key} onChange={(e) => setCritic({ ...critic, key: e.target.value })} />
                    <div className="hint">Your key is kept in the game server's memory for this game only: never saved, never shown on any screen, forgotten when the game ends or sits idle. Calls are billed to your account.</div>
                  </> : <div className="hint">Run <code>ollama pull llava</code> and <code>python ai_art_critic.py</code> on the PC running the game.</div>}
                </div>
              )}
            </div>
          </div>
        )}
        <Select id="vote-window" label="🗳️ Duel Voting" hint="The winner buttons wait until every spectator has voted, or this runs out" value={s['vote-window']} onChange={set('vote-window')}
          options={[['30', 'Wait for votes (up to 30s)'], ['60', 'Wait for votes (up to 60s)'], ['0', 'Don\'t wait (host can call it any time)']]} />
        <Select id="tax-threshold" label="💰 Santa Tax Threshold" hint="Lower = more chips go to the bank" value={s['tax-threshold']} onChange={set('tax-threshold')}
          options={[['2', '2+ chips (tight)'], ['3', '3+ chips (original V2 rule)'], ['4', '4+ chips (loose)'], ['99', 'Never (no tax)']]} />
        <Select id="base-cost" label="⚔️ Base Challenge Cost" hint="Cost for Gambits, Reprisals & Endgame steals" value={s['base-cost']} onChange={set('base-cost')}
          options={[['0', '0 + Naughty (free start)'], ['1', '1 + Naughty (standard)'], ['2', '2 + Naughty (expensive)'], ['3', '3 + Naughty (very expensive)']]} />
        <div className="form-group">
          <label className="label" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', cursor: 'pointer' }}>
            <input type="checkbox" id="setting-public-chips" className="input" style={{ width: 'auto', minHeight: 'auto' }} checked={publicChips} onChange={(e) => setPublicChips(e.target.checked)} />
            <span><Icon name="eye" /> Show everyone's chip counts</span>
          </label>
        </div>
        <Select id="tiebreaker-method" label="⚖️ Tiebreaker Party Game" value={method} onChange={set('tiebreaker-method')}
          hint="A tied top bid is always settled by a party game the whole table plays: the best-placed tied bidder wins the tie, everyone else plays for party points."
          options={[['party_mix', '🎉 Mix it up (a random party game each tie)'], ['whack_elf', '🔨 Whack-an-Elf (everyone plays, 20 s)'], ['sketch_off', '✏️ Sketch-Off (everyone draws, everyone votes)']]} />
        {(PARTY.includes(method) || method === 'party_mix') && (
          <div id="party-prize-setting">
            <Select id="party-prize" label="🎉 Party Game Prizes" value={s['party-prize']} onChange={set('party-prize')}
              hint="Top 3 score 3/2/1 party points (shown at the final tally). The Elf Bonus: the best finisher who wasn't in the tie gets +1 chip from the North Pole Fund."
              options={[['points_chips', 'Party points + Elf Bonus chip'], ['points', 'Party points only'], ['none', 'No prizes']]} />
          </div>
        )}
        <button type="button" className="btn btn-primary" data-testid="create-game" onClick={create} style={{ width: '100%' }}>🎮 Create New Game</button>
      </div>
    </div>
  );
}
