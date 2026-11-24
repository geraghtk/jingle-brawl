// The setup screen: house settings, then "Create New Game".
import { useState } from 'react';
import type { ReactNode } from 'react';
import { DEFAULT_DUEL_GAMES } from '../core/settings';
import { tableGames, TEAM_TABLE } from '../core/duelgame';
import type { DuelApp } from '../core/duelgame';
import { registered } from '../core/minigames/registry';
import type { DuelTableGames, Settings } from '../core/types';

/** At setup nobody has joined yet: tick the games for a table of any size (a team game still waits for four). */
const ANY_TABLE = 99;
import { Glyphs, Icon } from '../ui/icons';
import './setup.css';

const PARTY = ['sketch_off', 'whack_elf'];
const NL = '\n'; // the duel list is one per line

function Select({ id, label, hint, value, onChange, options }: {
  id: string; label: string; hint?: ReactNode; value: string; onChange: (v: string) => void; options: [string, string][];
}) {
  return (
    <div className="form-group">
      <label className="label" htmlFor={`setting-${id}`}><Glyphs>{label}</Glyphs></label>
      <select id={`setting-${id}`} className="select" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
      </select>
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}

/** What a game needs from the room, said before it is picked (not found out mid-duel). */
function needs(d: { supportedModes: string[]; compatibility: { displayRole: string } }): string[] {
  const out: string[] = [];
  if (d.compatibility.displayRole === 'required_shared_world') out.push('TV');
  if (d.supportedModes.length === 1 && d.supportedModes[0] === 'captain_teams') out.push(`${TEAM_TABLE}+ players`);
  return out;
}

/** One line per game on the setup screen (the whole of its rules is the row's tooltip, and the briefing before it starts). */
const GAME_LINE: Record<string, string> = {
  whack_elf: 'Everyone whacks elves, the best score wins',
  sketch_off: 'Everyone draws, everyone votes',
  draw_off: 'The two duellists draw, an AI critic judges',
  pass_the_gremlin: 'Hot potato with a puzzle to solve',
  reindeer_ring_toss: 'Flick rings onto the reindeer',
  gift_conveyor: 'Two teams build toys on a factory belt',
  nutcracker_knockout: 'The duellists box wind-up robots',
  mecha_santa: 'Each team drives one giant Santa',
  elf_tower_wars: 'Two workshops, toys against towers',
};
const firstSentence = (text: string) => (text.match(/^[^.!?:]+/)?.[0] || text).trim();

type GroupId = 'chips' | 'duels' | 'winnings' | 'minigames' | 'ties';
const GROUPS: GroupId[] = ['chips', 'duels', 'winnings', 'minigames', 'ties'];

/** A group of settings: closed, it is one row that says what is chosen. */
function Group({ id, title, summary, open, onToggle, children }: {
  id: GroupId; title: string; summary: string; open: boolean; onToggle: () => void; children: ReactNode;
}) {
  return (
    <section className={`sg${open ? ' open' : ''}`} data-group={id}>
      <button type="button" className="sg-head" aria-expanded={open} aria-controls={`setup-group-${id}`} data-testid={`setup-group-${id}`} onClick={onToggle}>
        <span className="sg-title"><Glyphs>{title}</Glyphs></span>
        <span className="sg-sum" data-testid={`setup-summary-${id}`}>{summary}</span>
        <span className="sg-chev" aria-hidden="true" />
      </button>
      {open && <div className="sg-body" id={`setup-group-${id}`}>{children}</div>}
    </section>
  );
}

/** A game the table can play on phones as a duel: a tick, its name, one line, what it needs. */
function GameRow({ id, name, line, rules, tags, checked, onChange }: {
  id: string; name: string; line: string; rules?: string; tags?: string[]; checked: boolean; onChange: () => void;
}) {
  return (
    <label className="sg-game" title={rules}>
      <input type="checkbox" id={`setting-duel-app-${id}`} checked={checked} onChange={onChange} />
      <span className="sg-game-name"><Glyphs>{name}</Glyphs></span>
      <span className="sg-game-line">{line}</span>
      {!!tags?.length && <span className="sg-game-tags" data-testid={`duel-app-needs-${id}`} aria-label={`Needs ${tags.join(' and ')}`}>{tags.map((t) => <span key={t} className="sg-tag">{t}</span>)}</span>}
    </label>
  );
}

export function Setup({ onCreate, error }: { onCreate: (settings: Settings, criticKey: string) => void; error?: string }) {
  const [s, setS] = useState({
    'reprisal-depth': '1', 'max-bid': '4', 'max-payout': '2', 'starting-chips': '', 'loser-dividend': 'defenders',
    'spectator-bets': '0', 'vote-window': '30', 'duel-pick': 'defender', 'tax-threshold': '2', 'base-cost': '1', 'tiebreaker-method': 'party_mix', 'party-prize': 'points_chips',
    'gremlin-challenges': 'mixed', 'duel-table-games': 'teams', 'ready-check': 'on', 'practice-round': 'on', 'pace': 'normal',
  });
  const [publicChips, setPublicChips] = useState(false);
  const [duelGames, setDuelGames] = useState(DEFAULT_DUEL_GAMES.join(NL));
  // the whole-table games that come up as a duel: the team games unless the host says otherwise (the ticks are the fine control)
  const [duelApps, setDuelApps] = useState<DuelApp[]>(() => tableGames('teams', ANY_TABLE));
  const setTableGames = (mode: string) => {
    setS((x) => ({ ...x, 'duel-table-games': mode }));
    setDuelApps((x) => [...tableGames(mode as DuelTableGames, ANY_TABLE), ...x.filter((a) => a === 'draw_off')]);
  };
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
      readyCheck: s['ready-check'] !== 'off',
      practiceRound: s['practice-round'] !== 'off',
      pace: s['pace'] === 'relaxed' ? 'relaxed' : 'normal',
      duelPick: s['duel-pick'] as Settings['duelPick'],
      duelGames: duelGames.split(NL).map((x) => x.trim()).filter(Boolean).join(NL), // text: an empty list must stay empty
      duelTableGames: s['duel-table-games'] as DuelTableGames,
      duelAppGames: duelApps,
      ...(duelApps.includes('pass_the_gremlin') ? { gremlinChallenges: s['gremlin-challenges'] as Settings['gremlinChallenges'] } : {}),
      taxThreshold: parseInt(s['tax-threshold']),
      baseCost: parseInt(s['base-cost']),
      tiebreakerMethod: method as Settings['tiebreakerMethod'],
      partyPrize: s['party-prize'] as Settings['partyPrize'],
      ...(duelApps.includes('draw_off') ? { aiCriticBackend: critic.backend } : {}),
      ...(s['starting-chips'] ? { startingChips: parseInt(s['starting-chips']) } : {}),
    };
    onCreate(settings, duelApps.includes('draw_off') && critic.backend !== 'local' ? critic.key.trim() : '');
  };

  // which groups are open: none to begin with (each row says what is chosen); opening one closes the others
  const [open, setOpen] = useState<GroupId[]>([]);
  const allOpen = open.length === GROUPS.length;
  const toggle = (g: GroupId) => setOpen((x) => (x.includes(g) ? x.filter((y) => y !== g) : x.length === GROUPS.length ? x : [g]));
  const duelsOn = s['duel-pick'] !== 'off';
  const games = duelsOn ? duelApps.length : 0;
  const sum: Record<GroupId, string> = {
    chips: [
      s['starting-chips'] ? `${s['starting-chips']} chips each` : 'Chips by table size',
      s['max-bid'] === '99' ? 'any bid' : `bids up to ${s['max-bid']}`,
      `a challenge costs ${s['base-cost']} + Naughty`,
      ...(publicChips ? ['chip counts shown'] : []),
    ].join(' · '),
    duels: [
      s['duel-pick'] === 'defender' ? 'The defender picks' : s['duel-pick'] === 'wheel' ? 'The wheel picks' : 'The duellists agree',
      s['vote-window'] === '0' ? 'no wait for votes' : `votes wait ${s['vote-window']} s`,
      s['reprisal-depth'] === '0' ? 'no reprisals' : s['reprisal-depth'] === '1' ? 'one reprisal' : `reprisals chain to ${s['reprisal-depth']}`,
      ...(s['spectator-bets'] === '1' ? ['bets on'] : []),
    ].join(' · '),
    winnings: [
      s['max-payout'] === '99' ? 'Winner gets all' : `Winner paid up to ${s['max-payout']}`,
      s['tax-threshold'] === '99' ? 'never taxed' : `tax at ${s['tax-threshold']}+ chips`,
      `dividend: ${s['loser-dividend'] === 'everyone' ? 'every loser' : s['loser-dividend'] === 'defenders' ? 'defenders' : 'nobody'}`,
    ].join(' · '),
    minigames: [
      games ? `${games} game${games === 1 ? '' : 's'} on phones as duels` : 'No games on phones as duels',
      s['ready-check'] === 'off' ? 'no ready check' : 'ready check',
      s['practice-round'] === 'off' ? 'no practice' : 'practice round',
      ...(s['pace'] === 'relaxed' ? ['relaxed pace'] : []),
    ].join(' · '),
    ties: [
      method === 'whack_elf' ? 'Whack-an-Elf' : method === 'sketch_off' ? 'Sketch-Off' : 'A random party game',
      s['party-prize'] === 'points_chips' ? 'party points + Elf Bonus' : s['party-prize'] === 'points' ? 'party points' : 'no prizes',
    ].join(' · '),
  };

  return (
    <div id="setup-screen" className="setup-screen setup-v2">
      <div className="setup-side">
      <div className="setup-hero">
        <div className="logo-wrap">
          <img src="ydlogo3.png" alt="YuleDuel" />
        </div>
        <h1 className="setup-title fz-riso">Yule<span style={{ color: 'var(--accent)' }}>Duel</span></h1>
        <div className="setup-tagline" id="setup-tagline">
          A game of <span style={{ color: 'var(--accent)', fontWeight: 700 }}>high stakes</span>, <span style={{ color: 'var(--accent)', fontWeight: 700 }}>low blows</span>, and <span style={{ color: 'var(--accent)', fontWeight: 700 }}>questionable gifts</span>.
        </div>
      </div>

      <div className="setup-go">
        {error && <p className="setup-error" role="alert" data-testid="create-error">{error}</p>}
        <button type="button" className="btn btn-primary" data-testid="create-game" onClick={create}>🎮 Create New Game</button>
      </div>
      <p className="setup-foot">
        The house rules are ready to play as they are. Games are deleted two hours after they end. <a href="privacy.html" target="_blank" rel="noreferrer" data-testid="privacy-link">Privacy</a>
      </p>
      </div>

      <div className="setup-groups" data-testid="setup-groups">
        <div className="setup-groups-head">
          <h2 className="setup-groups-title">House rules</h2>
          <button type="button" className="setup-all" data-testid="setup-open-all" aria-pressed={allOpen} onClick={() => setOpen(allOpen ? [] : GROUPS)}>{allOpen ? 'Close all' : 'Show all'}</button>
        </div>

        <Group id="chips" title="🪙 Chips and bids" summary={sum.chips} open={open.includes('chips')} onToggle={() => toggle('chips')}>
          <div className="sg-grid">
            <Select id="starting-chips" label="Starting chips" value={s['starting-chips']} onChange={set('starting-chips')}
              options={[['', 'By the number of players'], ['8', '8 chips'], ['10', '10 chips'], ['12', '12 chips'], ['15', '15 chips'], ['20', '20 chips']]} />
            <Select id="max-bid" label="Highest bid" value={s['max-bid']} onChange={set('max-bid')} options={[['3', '3 chips'], ['4', '4 chips'], ['5', '5 chips'], ['99', 'Unlimited']]} />
            <Select id="base-cost" label="Cost of a challenge" hint="Gambits, Reprisals and Endgame steals" value={s['base-cost']} onChange={set('base-cost')}
              options={[['0', '0 + Naughty (free start)'], ['1', '1 + Naughty (standard)'], ['2', '2 + Naughty (expensive)'], ['3', '3 + Naughty (very expensive)']]} />
            <div className="form-group sg-check">
              <label className="label">
                <input type="checkbox" id="setting-public-chips" checked={publicChips} onChange={(e) => setPublicChips(e.target.checked)} />
                <span><Icon name="eye" /> Show everyone's chip counts</span>
              </label>
            </div>
          </div>
        </Group>

        <Group id="duels" title="⚔️ Duels" summary={sum.duels} open={open.includes('duels')} onToggle={() => toggle('duels')}>
          <div className="sg-grid">
            <Select id="duel-pick" label="Who picks the duel" value={s['duel-pick']} onChange={set('duel-pick')}
              hint="Yield duels and tiebreaks always spin the wheel: nobody is really defending."
              options={[['defender', 'The defender (or a spin)'], ['wheel', 'Always the wheel'], ['off', 'Off: the duellists agree on one']]} />
            <Select id="vote-window" label="Calling the winner" hint="The winner buttons wait until every spectator has voted, or this runs out" value={s['vote-window']} onChange={set('vote-window')}
              options={[['30', 'Wait for votes (up to 30s)'], ['60', 'Wait for votes (up to 60s)'], ['0', "Don't wait"]]} />
            <Select id="reprisal-depth" label="Reprisals" hint="How many times a reprisal can set off another" value={s['reprisal-depth']} onChange={set('reprisal-depth')}
              options={[['0', 'None'], ['1', 'One (no chaining)'], ['2', 'Chain once (2 max)'], ['3', 'Chain twice (3 max)']]} />
            <Select id="spectator-bets" label="Spectator bets (experiment)" hint="Right callers split the wrong callers' chips. Correct calls always add to the Party Score." value={s['spectator-bets']} onChange={set('spectator-bets')}
              options={[['0', 'Off: voting is free'], ['1', 'On: a vote stakes 1 chip']]} />
          </div>
          {duelsOn && (
            <div className="form-group" id="duel-games-setting">
              <label className="label" htmlFor="setting-duel-games">Duels you'll play in the room</label>
              <textarea id="setting-duel-games" className="input duel-games-input" rows={5} value={duelGames} onChange={(e) => setDuelGames(e.target.value)} />
              <div className="hint">One per line. These go on the wheel and in the defender's list.</div>
            </div>
          )}
        </Group>

        <Group id="winnings" title="🎅 Winnings and Santa Tax" summary={sum.winnings} open={open.includes('winnings')} onToggle={() => toggle('winnings')}>
          <div className="sg-grid">
            <Select id="max-payout" label="Most a duel winner is paid" hint="Extra chips go to the North Pole Fund (the bank)" value={s['max-payout']} onChange={set('max-payout')}
              options={[['1', '1 chip (heavy tax)'], ['2', '2 chips (moderate)'], ['3', '3 chips (light tax)'], ['99', 'No tax (winner gets all)']]} />
            <Select id="tax-threshold" label="Santa Tax threshold" hint="Lower means more chips go to the bank" value={s['tax-threshold']} onChange={set('tax-threshold')}
              options={[['2', '2+ chips (tight)'], ['3', '3+ chips (original rule)'], ['4', '4+ chips (loose)'], ['99', 'Never (no tax)']]} />
            <Select id="loser-dividend" label="Loser Dividend (+1 chip)" hint={'"Defenders only" makes attacking riskier'} value={s['loser-dividend']} onChange={set('loser-dividend')}
              options={[['everyone', 'Everyone who loses'], ['defenders', 'Defenders only'], ['nobody', 'Nobody (harsh!)']]} />
          </div>
        </Group>

        <Group id="minigames" title="🎮 Mini-games" summary={sum.minigames} open={open.includes('minigames')} onToggle={() => toggle('minigames')}>
          {duelsOn ? (
            <div className="sg-games-block">
              <Select id="duel-table-games" label="Games on phones as duels" value={s['duel-table-games']} onChange={setTableGames}
                hint="So a duel is not just two people: with four or more, the duellists captain teams. Tick or untick any game."
                options={[['teams', 'Team games (everyone is on a team)'], ['all', 'Every game the whole table plays'], ['off', 'Only the ones I tick']]} />
              <div className="sg-games">
                {(['whack_elf', 'sketch_off'] as DuelApp[]).map((a) => (
                  <GameRow key={a} id={a} name={a === 'whack_elf' ? '🔨 Whack-an-Elf' : '✏️ Sketch-Off'} line={GAME_LINE[a]}
                    rules="On phones; everyone plays, the better-placed duellist wins." checked={duelApps.includes(a)} onChange={() => toggleApp(a)} />
                ))}
                {registered().map((d) => (
                  <GameRow key={d.id} id={d.id} name={`${d.emoji} ${d.label}`} line={GAME_LINE[d.id] || firstSentence(d.rules)} rules={d.rules} tags={needs(d)}
                    checked={duelApps.includes(d.id)} onChange={() => toggleApp(d.id)} />
                ))}
                <GameRow id="draw_off" name="🎨 AI Draw Off" line={GAME_LINE.draw_off} checked={duelApps.includes('draw_off')} onChange={() => toggleApp('draw_off')} />
              </div>
              {(duelApps.includes('pass_the_gremlin') || duelApps.includes('draw_off')) && (
                <div className="sg-grid sg-extras">
                  {duelApps.includes('pass_the_gremlin') && (
                    <Select id="gremlin-challenges" label="The Gremlin's challenges" hint="What the holder must do before passing it on" value={s['gremlin-challenges']} onChange={set('gremlin-challenges')}
                      options={[['mixed', 'A mix: math, memory and timing'], ['quick_math', 'Quick math only'], ['memory_flash', 'Memory only (repeat the symbols)'], ['perfect_timing', 'Timing only (stop it in the zone)']]} />
                  )}
                  {duelApps.includes('draw_off') && (
                    <div id="ai-critic-setting" className="form-group ai-critic-setting">
                      <label className="label" htmlFor="setting-ai-critic-backend">AI Draw Off's critic</label>
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
              )}
            </div>
          ) : (
            <p className="sg-note" data-testid="setup-no-duel-games">Picking the duel is off (see Duels), so no game on phones comes up as a duel: the duellists agree on one themselves.</p>
          )}
          <div className="sg-grid">
            <Select id="ready-check" label="Before a mini-game" hint="The instructions come up first, and nobody starts until every player has pressed Ready" value={s['ready-check']} onChange={set('ready-check')}
              options={[['on', 'Instructions, then everyone is ready'], ['off', 'Start straight away']]} />
            <Select id="practice-round" label="The first time a game comes up" hint="A short go that does not count, so nobody loses a gift to a game they have never seen" value={s['practice-round']} onChange={set('practice-round')}
              options={[['on', 'A practice round first (about 25 s)'], ['off', 'Straight in: it counts from the start']]} />
            <Select id="pace" label="Pace" hint="Relaxed gives more time: slower machines, longer clocks, longer to draw and vote. Good with small children or anyone new to phone games" value={s['pace']} onChange={set('pace')}
              options={[['normal', 'Normal'], ['relaxed', 'Relaxed (more time)']]} />
          </div>
        </Group>

        <Group id="ties" title="⚖️ Ties" summary={sum.ties} open={open.includes('ties')} onToggle={() => toggle('ties')}>
          <div className="sg-grid">
            <Select id="tiebreaker-method" label="A tied top bid is settled by" value={method} onChange={set('tiebreaker-method')}
              hint="A party game the whole table plays: the best-placed tied bidder wins the tie, everyone else plays for party points."
              options={[['party_mix', '🎉 Mix it up (a random party game each tie)'], ['whack_elf', '🔨 Whack-an-Elf (everyone plays, 20 s)'], ['sketch_off', '✏️ Sketch-Off (everyone draws, everyone votes)']]} />
            {(PARTY.includes(method) || method === 'party_mix') && (
              <div id="party-prize-setting">
                <Select id="party-prize" label="Party game prizes" value={s['party-prize']} onChange={set('party-prize')}
                  hint="Top 3 score 3/2/1 party points (shown at the final tally). The Elf Bonus: the best finisher who wasn't in the tie gets +1 chip from the North Pole Fund."
                  options={[['points_chips', 'Party points + Elf Bonus chip'], ['points', 'Party points only'], ['none', 'No prizes']]} />
              </div>
            )}
          </div>
        </Group>
      </div>
    </div>
  );
}
