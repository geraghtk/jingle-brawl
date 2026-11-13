// Santa, an elf and a reindeer who pop up at game moments with a one-liner.
// Shared by host.html (the TV: table-wide moments) and player.html (only YOUR moments).
//
//   Mascots.observe(gameState, { me })   call on every state change; it diffs against the
//                                        previous call and decides whether anyone pops up.
//   Mascots.say('elf', 'text', opts)     show one directly.
//
// They never block clicks (pointer-events: none), queue rather than overlap, keep a polite
// gap between appearances, and `?mascots=0` turns them off.
// @ts-nocheck — a straight port of mascots.js (DOM-only, no React); typed at its edges.
const OFF = new URLSearchParams(location.search).get('mascots') === '0';
const GAP_MS = 5000;        // quiet time before idle chatter
const MOMENT_GAP_MS = 1200; // big moments (Reprisal, ties, game start…) only wait this long
const SHOW_MS = 4200;  // how long one stays up

// --- the cast (flat vector, 120×120) ---------------------------------------------
const SVG = {
  santa: `<svg viewBox="0 0 120 120" aria-hidden="true">
    <circle cx="56" cy="70" r="24" fill="#f6c7a6"/>
    <path d="M28 72 C30 106 82 106 84 72 C76 86 66 90 56 90 C46 90 36 86 28 72 Z" fill="#fff"/>
    <path d="M40 80 C46 74 52 76 56 80 C60 76 66 74 72 80 C66 86 60 84 56 82 C52 84 46 86 40 80 Z" fill="#f1f1f1" stroke="#e2e2e2"/>
    <circle cx="47" cy="66" r="3" fill="#2b2b2b"/><circle cx="65" cy="66" r="3" fill="#2b2b2b"/>
    <circle cx="40" cy="74" r="4" fill="#f4a3a3" opacity=".7"/><circle cx="72" cy="74" r="4" fill="#f4a3a3" opacity=".7"/>
    <circle cx="56" cy="74" r="4" fill="#e8908a"/>
    <path d="M26 52 C30 18 74 8 96 30 L104 50 Z" fill="#d7263d"/>
    <circle cx="104" cy="54" r="9" fill="#fff"/>
    <rect x="20" y="46" width="72" height="14" rx="7" fill="#fff"/>
  </svg>`,
  elf: `<svg viewBox="0 0 120 120" aria-hidden="true">
    <path d="M30 70 L12 58 L32 82 Z" fill="#f2c1a0"/><path d="M88 70 L106 58 L86 82 Z" fill="#f2c1a0"/>
    <circle cx="59" cy="76" r="25" fill="#f8d2b5"/>
    <circle cx="50" cy="72" r="3" fill="#2b2b2b"/><circle cx="68" cy="72" r="3" fill="#2b2b2b"/>
    <path d="M49 84 C54 90 64 90 69 84" stroke="#b5533c" stroke-width="3" fill="none" stroke-linecap="round"/>
    <circle cx="44" cy="81" r="4" fill="#f4a3a3" opacity=".7"/><circle cx="74" cy="81" r="4" fill="#f4a3a3" opacity=".7"/>
    <path d="M30 56 C34 24 60 10 92 6 C80 20 82 36 86 56 Z" fill="#1f9d55"/>
    <circle cx="92" cy="8" r="6" fill="#f6c343"/>
    <rect x="28" y="50" width="62" height="12" rx="6" fill="#d7263d"/>
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

// --- what they say -----------------------------------------------------------------
// One pool per moment; `line(moment, x)` picks one at random. Lines that need a name or a
// number are functions of it. Keep them speech-bubble short and free of em dashes.
// Santa: knowing, judgy about the naughty list, dad jokes. Rudy: dramatic, competitive,
// devoted to the Misfit pile and carrots. Jingles the Elf: overworked and a gossip.
const chips = (n) => `${n} chip${n === 1 ? '' : 's'}`;
const LINES = {
  start: [ // santa, the game begins
    'Ho ho ho! Let the brawl begin.',
    'Ho ho ho! Play nice. Or don\'t. That\'s the game.',
    'I\'ve checked the list twice. Some of you are on it a LOT.',
    'Welcome! Please keep your elbows inside the sleigh.',
    'Chips out, smiles on, grudges optional. Ho ho ho!',
    'Let the brawl begin! I\'m not saying who\'s naughty. Yet.',
    'I see you when you\'re bidding. I know when you\'re bluffing.',
    'What do elves learn at school? The elf-abet! Ho ho. Let\'s play.',
    'Reading glasses: on. Naughty list: open. Begin!',
    'Ho ho ho! May the best gift-grabber win.',
    'Why is my sack so full? Re-gifts, mostly. Let\'s go!',
  ],
  endgame: [ // santa, every gift is open
    'All unwrapped! Time for the Misfit Lottery.',
    'Every gift is open. Now the Misfits get their moment.',
    'Misfit Lottery! One person\'s re-gift is another\'s treasure.',
    'All unwrapped! Steal or auction, choose wisely. I\'m taking notes.',
    'Final stretch! Even the naughty list is on the edge of its seat.',
    'The Misfit pile is calling. Ho ho, I hope someone answers.',
    'Last chance for glory. Or at least for a better candle.',
    'Endgame! My favourite part, after cookies. And naps.',
  ],
  ended: [ // santa, game over
    'That\'s a wrap! Santa saw everything. Everything.',
    'Game over! I\'ll be updating the naughty list tonight.',
    'That\'s a wrap! It\'s the thought that counts. Mostly.',
    'Well played! A few of you are definitely getting coal.',
    'Ho ho ho! Same time next year? I\'ll bring a longer list.',
    'Game over! Leftover chips don\'t buy cookies, you know.',
    'That\'s a wrap. Say thank you for your gift. Even the candle.',
    'Why did the gift go to therapy? It felt re-wrapped. Goodnight!',
  ],
  misfit: [ // rudy, a gift lands in the Misfit pile
    'Another one for the Island of Misfit Gifts!',
    'Into the Misfit pile you go.',
    'Welcome to the Misfit pile, little buddy. I\'ll guard you with my life.',
    'The Misfit pile grows! My kingdom expands!',
    'Misfit pile, party of one more! I\'ll get the carrots.',
    'Don\'t cry, little gift. The Misfits are the best family.',
    'To the Misfits! I\'ve always said they\'re underrated.',
    'MISFIT PILE! I live for this. Well, this and carrots.',
    'One more for the pile! I call it "my collection".',
    'Another Misfit! Honestly? Best gift on the table.',
  ],
  bidding: [ // jingles, chatter while sealed bids go in
    'Sealed bids are secret. Even from me. Especially from me.',
    'Ooh, shiny. Who wants it?',
    'Bid with your heart. Or your chips. Mostly chips.',
    'I wrapped that one. I\'m sworn to secrecy. And very tired.',
    'Psst. I heard someone\'s been saving chips all game.',
    'Fourteen-hour shift and still wrapping. Bid something, please.',
    'Don\'t look at me. I don\'t know who brought the fruitcake.',
    'Between us? Rudy bids with his nose. It\'s very wet.',
    'Sealed bids! Nobody tells me anything anyway.',
    'I wrapped four thousand gifts this week. Make this one count.',
    'Rumour has it Santa wanted this one. You didn\'t hear it from me.',
  ],
  noBids: [ // jingles, nobody bid; (opener's name)
    (n) => `Nobody?! ${n}, Grinch's Gambit is RIGHT there…`,
    (n) => `Zero bids! ${n}, the Gambit is calling your name.`,
    (n) => `Tumbleweeds. ${n}, maybe go full Grinch?`,
    (n) => `No takers?! ${n}, either it's amazing or it's a candle.`,
    (n) => `Not one chip! ${n}, keep it, or steal something better?`,
    (n) => `Silence! ${n}, that's either respect or fear.`,
    (n) => `Zero bids. ${n}, I'm telling everyone. It's my job.`,
    (n) => `Nobody wants it?! ${n}, I'd gossip, but there's nothing to say.`,
  ],
  tie: [ // rudy, tied top bid
    'A tie! This is getting spicy.',
    'A TIE! Somebody hold my carrots.',
    'Tied bids! I haven\'t been this excited since the Misfit pile.',
    'Same bid?! Great minds think alike. So do greedy ones.',
    'A tie! I\'d bet my antlers on this. Ooh, drama.',
    'Tie! I trained my whole life for a moment like this.',
    'Dead heat! Like the time I raced Dasher. I won. Allegedly.',
    'A tie! Nobody blink. Except me. I\'m very dramatic.',
  ],
  bigBid: [ // santa, a big winning bid; (the amount)
    (n) => `${chips(n)}! Somebody really wants this.`,
    (n) => `${chips(n)}?! Ho ho, that's either love or spite.`,
    (n) => `${chips(n)}. I've seen less effort for a pony.`,
    (n) => `${chips(n)}! Bold. I'm writing that down.`,
    (n) => `${chips(n)}! Leftover chips don't buy cookies, after all.`,
    (n) => `Ho ho! ${chips(n)}. Big spender alert.`,
    (n) => `${chips(n)}! Somebody's been very good. Or very bad.`,
    (n) => `${chips(n)} for this? I'm intrigued and a little worried.`,
  ],
  gambit: [ // jingles, a Grinch's Gambit duel starts; (the challenger)
    (n) => `${n} is going full Grinch!`,
    (n) => `Grinch's Gambit! ${n} has chosen chaos.`,
    (n) => `Ooh! ${n} just took a page from the Grinch's book.`,
    (n) => `${n} goes Grinch! I'm telling everyone. Everyone.`,
    (n) => `Gasp! ${n} is coming for someone's gift!`,
    (n) => `${n}, your heart just shrank three sizes. Respect.`,
    (n) => `Grinch mode: ON. ${n}, bold move!`,
    (n) => `${n} pulls a Gambit! Somebody get me snacks.`,
  ],
  reprisal: [ // rudy, a Reindeer Reprisal; (who gets it)
    (n) => `Reindeer kick back! ${n} gets a Reprisal.`,
    (n) => `Reprisal time! ${n}, aim those hooves.`,
    (n) => `${n} gets a Reprisal! Hold my carrots.`,
    (n) => `Not so fast! ${n} kicks back with a Reprisal.`,
    (n) => `REPRISAL! ${n}, I trained you for this.`,
    (n) => `Kick back, ${n}! Revenge is a dish best served with carrots.`,
    (n) => `${n} gets a Reprisal! Somebody's gift is in danger.`,
    (n) => `Hooves up! ${n} has a Reprisal and nothing to lose.`,
  ],
  tax: [ // santa, duel tax paid; (the amount)
    (n) => `Ho ho. ${n} for the North Pole Fund. Thank you kindly.`,
    (n) => `${chips(n)} to the North Pole Fund. The reindeer need snacks.`,
    (n) => `${chips(n)} in tax. Sleighs don't polish themselves.`,
    (n) => `The North Pole thanks you for your ${chips(n)}. Ho ho.`,
    (n) => `${chips(n)} for the Fund! Elf cocoa isn't free, you know.`,
    (n) => `Tax time! ${chips(n)} to the North Pole. It's for the carrots.`,
    (n) => `Ho ho, ${chips(n)} for the Fund. Beard trimmers aren't cheap.`,
    (n) => `${chips(n)} for the Fund. Consider it a nice-list donation.`,
  ],
  phoneOpener: [ // santa, on your phone: you're the opener
    'You\'re up! Unwrap it and look pleased.',
    'Your turn! Unwrap it like you\'ve never seen paper before.',
    'You\'re up! Whatever it is, say "Oh, I love it!"',
    'Your turn to unwrap! I\'m watching your face. Very closely.',
    'You\'re up! Remember, it\'s the thought that counts. Mostly.',
    'Unwrap time! Practise your surprised face. Ho ho.',
    'You\'re up! Pro tip: shake it first. Unless it sloshes.',
    'Your turn! Rip it or peel it, Santa doesn\'t judge. Much.',
  ],
  phoneReprisal: [ // rudy, on your phone: you've got a Reprisal
    'Lost? Kick back! Pick a target… or let it go.',
    'Kick back! Pick a gift to challenge, or let it go.',
    'You lost? Not on my watch! Pick a target.',
    'Reprisal time! Choose wisely. I\'m rooting for chaos.',
    'Hooves up! Pick a target, or walk away with dignity.',
    'Revenge? In THIS economy? Pick a target. Or don\'t.',
    'Pick a Misfit if you like. They\'re lovely. I\'d know.',
    'Get back in there, champ! Pick a target… or let it go.',
  ],
  phoneEndgame: [ // jingles, on your phone: your Misfit Lottery choice
    'Misfit Lottery! Steal a gift, or auction the Misfit?',
    'Your call! Steal something shiny, or auction the Misfit?',
    'Steal or auction? I know what I\'d do. I\'m not telling.',
    'Misfit Lottery! Choose fast, I\'m on my break.',
    'Your move! Steal a gift, or take your chances at auction.',
    'Psst. The Misfit isn\'t as bad as it looks. Probably. Steal or auction?',
  ],
  idle: [ // jingles, bidding has sat still for 30 seconds
    'Psst… those chips won\'t bid themselves.',
    'Still thinking? I\'ve wrapped three gifts since you started.',
    'Hello? The bid button is the shiny one.',
    'Take your time. I\'ll just be here. Aging.',
    'Tick tock! Even Rudy has decided, and he\'s mostly carrots.',
    'Psst. Leftover chips are worth nothing at the end. Just saying.',
    'Twenty-hour shift over here. Please bid so I can sit down.',
    'I\'d peek at the bids, but I\'m too tired to stand on tiptoe.',
  ],
};
const line = (moment, x) => { const l = pick(LINES[moment]); return typeof l === 'function' ? l(x) : l; };

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

  if (was.phase === 'lobby' && snap.phase === 'main') say('santa', line('start'), { hop: true });
  if (was.phase !== 'endgame' && snap.phase === 'endgame') say('santa', line('endgame'), { hop: true });
  if (was.phase !== 'ended' && snap.phase === 'ended') say('santa', line('ended'), { hop: true });
  if (snap.misfits > was.misfits) say('reindeer', line('misfit'));
  if (!t || !turnChanged) return idle(snap);

  if (t.phase === 'bidding' && Math.random() < 0.3) say('elf', line('bidding'), { chatter: true });
  if (t.phase === 'reveal') {
    const bids = Object.values(t.bids || {}).filter((b) => b.amount > 0).map((b) => b.amount);
    const top = Math.max(0, ...bids);
    if (!bids.length) say('elf', line('noBids', name(t.openerId)));
    else if (bids.filter((b) => b === top).length >= 2) say('reindeer', line('tie'), { hop: true });
    else if (top >= 4) say('santa', line('bigBid', top), { chatter: true });
  }
  if (t.phase === 'duel' && t.isGambit) say('elf', line('gambit', name(t.challengerId)), { side: 'left' });
  if (t.phase === 'reprisal') {
    say('reindeer', line('reprisal', name(t.reprisalPlayerId)), {
      hop: true, kick: document.querySelector('#turn-panel, .turn-panel, main'),
    });
  }
  if (pt?.phase === 'duel' && (pt.tax || 0) > 0 && t.phase !== 'duel') say('santa', line('tax', pt.tax));
  idle(snap);
}

/** Phones: only the player's own big moments. */
function observePhone(snap, was, me, turnChanged, name) {
  const t = snap.turn;
  if (!t || !turnChanged) return;
  if (t.phase === 'bidding' && t.openerId === me) say('santa', line('phoneOpener'), { side: 'left' });
  if (t.phase === 'reprisal' && t.reprisalPlayerId === me) say('reindeer', line('phoneReprisal'), { hop: true });
  if (t.phase === 'endgame_choice' && t.activePlayerId === me) say('elf', line('phoneEndgame'));
}

/** The elf nudges if bidding sits still for a while. */
function idle(snap) {
  const t = snap.turn;
  const key = t?.phase === 'bidding' ? `${t.giftId}|${Object.keys(t.bids || {}).length}` : null;
  if (key === idleKey) return;
  idleKey = key;
  clearTimeout(idleTimer);
  if (key) idleTimer = setTimeout(() => say('elf', line('idle'), { side: 'left', chatter: true }), 30000);
}


export { say, observe, SVG as svg, LINES as lines };
