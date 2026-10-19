// Santa, an elf and a reindeer who pop up at game moments with a one-liner.
// Shared by host.html (the TV: table-wide moments) and player.html (only YOUR moments).
//
//   Mascots.observe(gameState, { me })   call on every state change; it diffs against the
//                                        previous call and decides whether anyone pops up.
//   Mascots.say('elf', 'text', opts)     show one directly.
//
// They never block clicks (pointer-events: none), queue rather than overlap, keep a polite
// gap between appearances, respect prefers-reduced-motion, and `?mascots=0` turns them off.
// @ts-nocheck — a straight port of mascots.js (DOM-only, no React); typed at its edges.
const OFF = new URLSearchParams(location.search).get('mascots') === '0';
const GAP_MS = 5000;        // quiet time before idle chatter
const MOMENT_GAP_MS = 1200; // big moments (Reprisal, ties, game start…) only wait this long
const SHOW_MS = 4200;  // how long one stays up

// --- the cast (flat vector, 120×120) ---------------------------------------------
const SVG = {
  santa: `<svg viewBox="0 0 120 120" aria-hidden="true">
    <path d="M26 52 C30 18 74 8 96 30 L104 50 Z" fill="#d7263d"/>
    <circle cx="104" cy="54" r="9" fill="#fff"/>
    <rect x="20" y="46" width="72" height="14" rx="7" fill="#fff"/>
    <circle cx="56" cy="70" r="24" fill="#f6c7a6"/>
    <path d="M28 72 C30 106 82 106 84 72 C76 86 66 90 56 90 C46 90 36 86 28 72 Z" fill="#fff"/>
    <path d="M40 80 C46 74 52 76 56 80 C60 76 66 74 72 80 C66 86 60 84 56 82 C52 84 46 86 40 80 Z" fill="#f1f1f1" stroke="#e2e2e2"/>
    <circle cx="47" cy="66" r="3" fill="#2b2b2b"/><circle cx="65" cy="66" r="3" fill="#2b2b2b"/>
    <circle cx="40" cy="74" r="4" fill="#f4a3a3" opacity=".7"/><circle cx="72" cy="74" r="4" fill="#f4a3a3" opacity=".7"/>
    <circle cx="56" cy="74" r="4" fill="#e8908a"/>
  </svg>`,
  elf: `<svg viewBox="0 0 120 120" aria-hidden="true">
    <path d="M30 56 C34 24 60 10 92 6 C80 20 82 36 86 56 Z" fill="#1f9d55"/>
    <circle cx="92" cy="8" r="6" fill="#f6c343"/>
    <rect x="28" y="50" width="62" height="12" rx="6" fill="#d7263d"/>
    <path d="M30 70 L12 58 L32 82 Z" fill="#f2c1a0"/><path d="M88 70 L106 58 L86 82 Z" fill="#f2c1a0"/>
    <circle cx="59" cy="76" r="25" fill="#f8d2b5"/>
    <circle cx="50" cy="72" r="3" fill="#2b2b2b"/><circle cx="68" cy="72" r="3" fill="#2b2b2b"/>
    <path d="M49 84 C54 90 64 90 69 84" stroke="#b5533c" stroke-width="3" fill="none" stroke-linecap="round"/>
    <circle cx="44" cy="81" r="4" fill="#f4a3a3" opacity=".7"/><circle cx="74" cy="81" r="4" fill="#f4a3a3" opacity=".7"/>
  </svg>`,
  reindeer: `<svg viewBox="0 0 120 120" aria-hidden="true">
    <g stroke="#7a4b24" stroke-width="5" stroke-linecap="round" fill="none">
      <path d="M44 40 C38 26 30 18 22 10 M34 26 L20 26 M40 32 L30 40"/>
      <path d="M76 40 C82 26 90 18 98 10 M86 26 L100 26 M80 32 L90 40"/>
    </g>
    <ellipse cx="34" cy="50" rx="10" ry="6" fill="#a8693a" transform="rotate(-25 34 50)"/>
    <ellipse cx="86" cy="50" rx="10" ry="6" fill="#a8693a" transform="rotate(25 86 50)"/>
    <path d="M38 50 C38 34 82 34 82 50 L80 84 C78 102 42 102 40 84 Z" fill="#b97a45"/>
    <ellipse cx="60" cy="88" rx="17" ry="13" fill="#e2b98f"/>
    <circle cx="50" cy="60" r="3.5" fill="#2b2b2b"/><circle cx="70" cy="60" r="3.5" fill="#2b2b2b"/>
    <circle cx="60" cy="82" r="8" fill="#e23b3b"/><circle cx="57" cy="79" r="2.5" fill="#ff9c9c"/>
  </svg>`,
};
const NAMES = { santa: 'Santa', elf: 'Jingles the Elf', reindeer: 'Rudy' };

const css = `
  .mascot { position: fixed; bottom: 0; z-index: 2300; pointer-events: none; display: flex; align-items: flex-end; gap: 10px;
    transform: translateY(110%); transition: transform .45s cubic-bezier(.34,1.56,.64,1); }
  .mascot.left { left: 24px; } .mascot.right { right: 24px; flex-direction: row-reverse; }
  .mascot.up { transform: translateY(0); }
  .mascot svg { width: var(--mascot-size, 132px); height: var(--mascot-size, 132px); display: block; filter: drop-shadow(0 8px 14px rgba(0,0,0,.25)); }
  .mascot .bubble { margin-bottom: calc(var(--mascot-size, 132px) * .62); max-width: 300px; padding: 12px 16px; border-radius: 18px;
    background: #fff; color: #161a26; font: 600 15px/1.35 system-ui, sans-serif; box-shadow: 0 10px 30px rgba(15,20,35,.2); position: relative; }
  .mascot .bubble b { display: block; font-size: 10px; letter-spacing: .14em; text-transform: uppercase; color: #8a90a2; margin-bottom: 3px; }
  .mascot.left .bubble { border-bottom-left-radius: 4px; } .mascot.right .bubble { border-bottom-right-radius: 4px; }
  .mascot.hop svg { animation: mascotHop .5s ease-in-out 2; }
  @keyframes mascotHop { 50% { transform: translateY(-14px) rotate(-4deg); } }
  .mascot-kick { animation: mascotKick .45s ease-in-out; }
  @keyframes mascotKick { 20% { transform: translateX(-10px) rotate(-.6deg); } 50% { transform: translateX(8px) rotate(.4deg); } 80% { transform: translateX(-4px); } }
  @media (max-width: 640px) {
    .mascot { --mascot-size: 92px; } .mascot.left { left: 8px; } .mascot.right { right: 8px; }
    .mascot .bubble { max-width: 210px; font-size: 13px; padding: 10px 12px; }
  }
  @media (prefers-reduced-motion: reduce) { .mascot { transition: none; } .mascot.hop svg, .mascot-kick { animation: none; } }
`;

const queue = [];
let busy = false;
let lastShown = 0;

function mount() {
  if (document.getElementById('mascot-style')) return;
  const style = document.createElement('style');
  style.id = 'mascot-style';
  style.textContent = css;
  document.head.appendChild(style);
}

/**
 * Show `who` saying `text`. opts: { side: 'left'|'right', hop, kick: Element, chatter }.
 * `chatter` lines are flavour: dropped if anyone is talking or waiting. Everything
 * else is a game moment: it jumps ahead of queued chatter and waits only briefly.
 */
function say(who, text, opts = {}) {
  if (OFF || !SVG[who]) return;
  if (opts.chatter && (busy || queue.length)) return;
  for (let i = queue.length - 1; i >= 0; i--) if (queue[i].opts.chatter) queue.splice(i, 1);
  if (queue.length >= 3) queue.shift(); // stale moments aren't worth waiting for
  queue.push({ who, text, opts });
  pump();
}

function pump() {
  if (busy || !queue.length) return;
  const wait = Math.max(0, lastShown + (queue[0].opts.chatter ? GAP_MS : MOMENT_GAP_MS) - Date.now());
  if (wait) { setTimeout(pump, wait); return; }
  const { who, text, opts } = queue.shift();
  busy = true;
  mount();
  const el = document.createElement('div');
  el.className = `mascot ${opts.side || (who === 'santa' ? 'left' : 'right')}${opts.hop ? ' hop' : ''}`;
  el.dataset.mascot = who;
  el.setAttribute('role', 'status');
  el.innerHTML = `${SVG[who]}<div class="bubble"><b>${NAMES[who]}</b></div>`;
  el.querySelector('.bubble').append(text); // text node: names are user input
  document.body.appendChild(el);
  requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add('up')));
  if (opts.kick instanceof Element) {
    setTimeout(() => { opts.kick.classList.remove('mascot-kick'); void opts.kick.offsetWidth; opts.kick.classList.add('mascot-kick'); }, 450);
  }
  setTimeout(() => {
    el.classList.remove('up');
    setTimeout(() => { el.remove(); busy = false; lastShown = Date.now(); pump(); }, 500);
  }, SHOW_MS);
}

const pick = (lines) => lines[Math.floor(Math.random() * lines.length)];

// --- reacting to the game ----------------------------------------------------------
let prev = null;
let idleTimer = null;
let idleKey = null;

function observe(state, { me } = {}) {
  if (OFF || !state) return;
  const snap = {
    phase: state.phase,
    turn: state.currentTurn ? { ...state.currentTurn } : null,
    misfits: Object.values(state.gifts || {}).filter((g) => g.ownerId === 'misfit').length,
    players: state.players || {},
  };
  const was = prev;
  prev = snap;
  if (!was) return; // first call: just remember where we are (no pop-ups on page load/rejoin)
  const t = snap.turn;
  const pt = was.turn;
  const name = (id) => snap.players[id]?.name || 'someone';
  const turnChanged = t?.phase !== pt?.phase || t?.giftId !== pt?.giftId;
  if (me) return observePhone(snap, was, me, turnChanged, name);

  if (was.phase === 'lobby' && snap.phase === 'main') {
    say('santa', pick(['Ho ho ho! Let the brawl begin.', 'Ho ho ho! Play nice. Or don\'t — that\'s the game.']), { hop: true });
  }
  if (was.phase !== 'endgame' && snap.phase === 'endgame') say('santa', 'All unwrapped! Time for the Misfit Lottery.', { hop: true });
  if (was.phase !== 'ended' && snap.phase === 'ended') say('santa', 'That\'s a wrap! Santa saw everything. Everything.', { hop: true });
  if (snap.misfits > was.misfits) say('reindeer', pick(['Another one for the Island of Misfit Gifts!', 'Into the Misfit pile you go.']));
  if (!t || !turnChanged) return idle(snap);

  if (t.phase === 'bidding' && Math.random() < 0.3) {
    say('elf', pick(['Sealed bids are secret. Even from me. Especially from me.', 'Ooh, shiny. Who wants it?', 'Bid with your heart. Or your chips. Mostly chips.']), { chatter: true });
  }
  if (t.phase === 'reveal') {
    const bids = Object.values(t.bids || {}).filter((b) => b.amount > 0).map((b) => b.amount);
    const top = Math.max(0, ...bids);
    if (!bids.length) say('elf', `Nobody?! ${name(t.openerId)}, Grinch's Gambit is RIGHT there…`);
    else if (bids.filter((b) => b === top).length >= 2) say('reindeer', 'A tie! This is getting spicy.', { hop: true });
    else if (top >= 4) say('santa', `${top} chips! Somebody really wants this.`, { chatter: true });
  }
  if (t.phase === 'duel' && t.isGambit) say('elf', `${name(t.challengerId)} is going full Grinch!`, { side: 'left' });
  if (t.phase === 'reprisal') {
    say('reindeer', `Reindeer kick back! ${name(t.reprisalPlayerId)} gets a Reprisal.`, {
      hop: true, kick: document.querySelector('#turn-panel, .turn-panel, main'),
    });
  }
  if (pt?.phase === 'duel' && (pt.tax || 0) > 0 && t.phase !== 'duel') say('santa', `Ho ho — ${pt.tax} for the North Pole Fund. Thank you kindly.`);
  idle(snap);
}

/** Phones: only the player's own big moments. */
function observePhone(snap, was, me, turnChanged, name) {
  const t = snap.turn;
  if (!t || !turnChanged) return;
  if (t.phase === 'bidding' && t.openerId === me) say('santa', 'You\'re up! Unwrap it and look pleased.', { side: 'left' });
  if (t.phase === 'reprisal' && t.reprisalPlayerId === me) say('reindeer', 'Lost? Kick back! Pick a target… or let it go.', { hop: true });
  if (t.phase === 'endgame_choice' && t.activePlayerId === me) say('elf', 'Misfit Lottery! Steal a gift, or auction the Misfit?');
}

/** The elf nudges if bidding sits still for a while. */
function idle(snap) {
  const t = snap.turn;
  const key = t?.phase === 'bidding' ? `${t.giftId}|${Object.keys(t.bids || {}).length}` : null;
  if (key === idleKey) return;
  idleKey = key;
  clearTimeout(idleTimer);
  if (key) idleTimer = setTimeout(() => say('elf', 'Psst… those chips won\'t bid themselves.', { side: 'left', chatter: true }), 30000);
}


export { say, observe, SVG as svg };
