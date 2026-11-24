// The finale. When the game ends every screen gets the Party MVP moment (ui/MvpReveal.tsx), then the
// night's recap (core/recap.ts): on the TV and the console a show (third, second and first place rise
// onto a podium, the awards are read out one by one, then who got what); on a phone one card to scroll
// (your place first). Phones and the console can save it as a picture (ui/recapPicture.ts).
import { useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import { ordinal, recapOf } from '../core/recap';
import type { Award, PodiumStep, Recap as RecapData, RecapLine } from '../core/recap';
import type { Game, PlayerId } from '../core/types';
import { Avatar } from './avatar';
import { Icon } from './icons';
import type { IconName } from './icons';
import { Confetti, CountUp } from './motion';
import { FitName } from './FittedName';
import { MvpReveal, useMvps } from './MvpReveal';
import { renderPicture, sharePicture, stepNames } from './recapPicture';
import { play } from './sound';
import './recap.css';

const BRAND = 'YuleDuel';
/** How long the TV holds the Party MVP before the show goes on (a console or a phone waits for a tap). */
const MVP_HOLD_MS = 6500;
const INTRO_MS = 1100, RISE_MS = 2200, WINNER_MS = 5600, AWARD_MS = 2600, LAST_AWARD_MS = 4200;
/** A console narrower than this (the Host tab inside a phone) gets the card, not the show. */
const SHOW_MIN_W = 760;

type Where = 'tv' | 'console' | 'phone';

export function Recap({ game, me, where }: { game: Game; me?: PlayerId; where: Where }) {
  const mvps = useMvps(game);
  const recap = useMemo(() => recapOf(game), [game.players, game.gifts, game.minigame]); // eslint-disable-line react-hooks/exhaustive-deps
  const [beat, setBeat] = useState<'mvp' | 'recap' | 'closed'>('mvp');
  const wide = useWide();
  if (!mvps.length) return null;                 // nobody scored a point: no extra prize, no places (as before)
  if (beat === 'mvp') {
    return <MvpReveal game={game} me={me} variant={where === 'phone' ? 'phone' : 'tv'} onClose={() => setBeat('recap')} autoMs={where === 'tv' ? MVP_HOLD_MS : undefined} />;
  }
  if (beat === 'closed') {
    return <button type="button" className={`recap-open on-${where}`} data-testid="recap-open" onClick={() => setBeat('recap')}><Icon name="trophy" /> The night's recap</button>;
  }
  const close = () => setBeat('closed');
  return where === 'phone' || (where === 'console' && !wide)
    ? <RecapCard recap={recap} me={me} where={where} onClose={close} />
    : <RecapShow recap={recap} where={where} onClose={close} />;
}

function useWide() {
  const [wide, setWide] = useState(() => window.innerWidth >= SHOW_MIN_W);
  useEffect(() => {
    const on = () => setWide(window.innerWidth >= SHOW_MIN_W);
    window.addEventListener('resize', on);
    return () => window.removeEventListener('resize', on);
  }, []);
  return wide;
}

const today = () => new Date().toLocaleDateString(undefined, { day: 'numeric', month: 'long', year: 'numeric' });

// --- the picture -----------------------------------------------------------------------------------

/** "Save a picture": made ahead of the tap (a share sheet opens only straight from a tap), shared or downloaded. */
function SaveButton({ recap, className }: { recap: RecapData; className?: string }) {
  const ready = useRef<Blob | null>(null);
  const [state, setState] = useState<'idle' | 'busy' | 'done'>('idle');
  const words = useMemo(() => ({ brand: BRAND, title: 'The final standings', date: today() }), []);
  useEffect(() => {
    let live = true;
    ready.current = null;
    renderPicture(recap, words).then((b) => { if (live) ready.current = b; }, () => {});
    return () => { live = false; };
  }, [recap, words]);
  const save = async () => {
    if (state === 'busy') return;
    setState('busy');
    try {
      const blob = ready.current || await renderPicture(recap, words);
      const how = await sharePicture(blob, `yuleduel-${new Date().toISOString().slice(0, 10)}.png`, `${BRAND}: ${words.title.toLowerCase()}`);
      setState(how === 'cancelled' ? 'idle' : 'done');
    } catch { setState('idle'); }                 // no picture on this browser: say nothing, the recap is still here
  };
  useEffect(() => {
    if (state !== 'done') return;
    const id = window.setTimeout(() => setState('idle'), 2400);
    return () => clearTimeout(id);
  }, [state]);
  return (
    <button type="button" className={`rcp-btn primary${className ? ` ${className}` : ''}`} data-testid="recap-save" disabled={state === 'busy'} onClick={(e) => { e.stopPropagation(); void save(); }}>
      {state === 'busy' ? 'Making your picture' : state === 'done' ? 'Picture saved' : 'Save a picture'}
    </button>
  );
}

// --- pieces both layouts draw ----------------------------------------------------------------------

const PODIUM_ORDER = [2, 1, 3];
const SHOWN_ON_STEP = 3;

function Podium({ steps, up, size }: { steps: PodiumStep[]; up: (place: number) => boolean; size: (place: number) => number }) {
  const order = PODIUM_ORDER.map((place) => steps.find((s) => s.place === place)).filter((s): s is PodiumStep => !!s);
  return (
    <div className="rcp-podium" data-testid="recap-podium">
      {order.map((s) => (
        <div key={s.place} className={`rcp-step p${s.place}${up(s.place) ? ' up' : ''}`} data-testid={`recap-step-${s.place}`} aria-hidden={!up(s.place)}>
          <div className="rcp-step-who">
            <div className={`rcp-step-avs n${Math.min(SHOWN_ON_STEP, s.players.length)}`}>
              {s.place === 1 && <Icon name="crown" className="rcp-crown" />}
              {s.players.slice(0, SHOWN_ON_STEP).map((p) => <Avatar key={p.id} name={p.name} avatar={p.avatar} size={size(s.place)} />)}
            </div>
            <div className="rcp-step-name"><FitName name={stepNames(s.players)} chars={12} lines={2} min={0.5} /></div>
          </div>
          <div className="rcp-step-block">
            <span className="rcp-step-place">{ordinal(s.place)}</span>
            <span className="rcp-step-score">{up(s.place) ? <CountUp value={s.players[0].score} from={0} ms={900} delay={350} /> : 0} Party Score</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function AwardRow({ a, i }: { a: Award; i: number }) {
  return (
    <li className="rcp-award-row mo-up" style={{ ['--i']: i } as CSSProperties} data-testid={`recap-award-${a.id}`}>
      <Icon name={a.icon as IconName} className="rcp-award-icon" />
      <div className="rcp-award-text">
        <div className="rcp-award-title">{a.title}</div>
        <div className="rcp-award-who"><strong>{a.name}</strong><i aria-hidden> · </i><span>{a.detail}</span></div>
      </div>
    </li>
  );
}

function Lines({ recap, me }: { recap: RecapData; me?: PlayerId }) {
  const cols = recap.lines.length > 6 ? 2 : 1;
  const rows = Math.max(1, Math.ceil(recap.lines.length / cols));
  return (
    <ol className={`rcp-list cols${cols}`} style={{ ['--rows']: rows } as CSSProperties} data-testid="recap-list">
      {recap.lines.map((l: RecapLine, i) => (
        <li key={l.id} className={`rcp-line mo-up${l.place > 0 && l.place <= 3 && l.score > 0 ? ' top' : ''}${l.id === me ? ' me' : ''}`} style={{ ['--i']: i } as CSSProperties} data-testid={`recap-line-${l.id}`}>
          <span className="rcp-line-place">{l.place > 0 ? ordinal(l.place) : ''}</span>
          <Avatar name={l.name} avatar={l.avatar} size={36} />
          <span className="rcp-line-who">
            <span className="rcp-line-name">{l.name}</span>
            <span className={`rcp-line-gift${l.gift ? '' : ' none'}`}>{l.gift ? <><Icon name="gift" /> {l.gift.name}</> : 'No gift this year'}</span>
          </span>
          <span className="rcp-line-score">{l.score}</span>
        </li>
      ))}
    </ol>
  );
}

// --- the TV and the console: the show --------------------------------------------------------------

type Step = { scene: 'podium'; shown: number; ms: number } | { scene: 'awards'; shown: number; ms: number } | { scene: 'final'; ms: 0 };

/** The running order: an empty podium, each place from the last to the first, each award, then who got what. */
export function showSteps(recap: Pick<RecapData, 'podium' | 'awards'>): Step[] {
  const steps: Step[] = [];
  const n = recap.podium.length;
  if (n) steps.push({ scene: 'podium', shown: 0, ms: INTRO_MS });
  for (let i = 1; i <= n; i++) steps.push({ scene: 'podium', shown: i, ms: i === n ? WINNER_MS : RISE_MS });
  recap.awards.forEach((_, i) => steps.push({ scene: 'awards', shown: i + 1, ms: i === recap.awards.length - 1 ? LAST_AWARD_MS : AWARD_MS }));
  steps.push({ scene: 'final', ms: 0 });
  return steps;
}

function RecapShow({ recap, where, onClose }: { recap: RecapData; where: Where; onClose: () => void }) {
  const steps = useMemo(() => showSteps(recap), [recap.podium.length, recap.awards.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const [at, setAt] = useState(0);
  const step = steps[Math.min(at, steps.length - 1)];
  const last = at >= steps.length - 1;
  const tv = where === 'tv';
  // the podium fills from the lowest place up: third, second, first
  const rising = useMemo(() => [...recap.podium].sort((a, b) => b.place - a.place).map((s): number => s.place), [recap.podium]);
  const upCount = step.scene === 'podium' ? step.shown : rising.length;
  const winnerUp = step.scene === 'podium' && step.shown === rising.length;

  useEffect(() => {
    if (last) return;
    const id = window.setTimeout(() => setAt((a) => a + 1), step.ms);
    return () => clearTimeout(id);
  }, [at, last]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {                                 // the TV is the room's loudspeaker
    if (!tv) return;
    if (step.scene === 'podium' && step.shown > 0) play(winnerUp ? 'fanfare' : 'thud', winnerUp ? 1 : 0.8 + step.shown * 0.15);
    else if (step.scene === 'awards') play('ding', 0.9 + step.shown * 0.08);
    else if (step.scene === 'final') play('bell');
  }, [at]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className={`recap as-show on-${where} scene-${step.scene}`} id="recap" role="dialog" aria-label="The night's recap" data-testid="recap" data-step={at} data-scene={step.scene}
      onClick={last ? undefined : () => setAt((a) => Math.min(a + 1, steps.length - 1))}>
      {winnerUp && <Confetti n={140} seed={7} />}
      {step.scene === 'final' && <Confetti n={60} seed={11} />}
      {!tv && (
        <div className="rcp-controls">
          {!last && <span className="rcp-skip">Click to skip ahead</span>}
          <SaveButton recap={recap} />
          <button type="button" className="rcp-btn ghost" data-testid="recap-close" onClick={(e) => { e.stopPropagation(); onClose(); }}>Close</button>
        </div>
      )}

      {step.scene === 'podium' && (
        <section className="rcp-scene" key="podium">
          <header className="rcp-head">
            <span className="fz-sticker yel rcp-sticker">{BRAND}</span>
            <h1 className="rcp-title mo-slam">The final standings</h1>
          </header>
          <Podium steps={recap.podium} up={(place) => rising.indexOf(place) < upCount} size={() => 96} />
        </section>
      )}

      {step.scene === 'awards' && (
        <section className="rcp-scene" key="awards">
          <header className="rcp-head">
            <span className="fz-sticker yel rcp-sticker">And now</span>
            <h1 className="rcp-title mo-slam">Tonight's awards</h1>
          </header>
          <ul className={`rcp-cards n${recap.awards.length}`}>
            {recap.awards.map((a, i) => (
              <li key={a.id} className={`rcp-card${i < step.shown ? ' up' : ''}`} aria-hidden={i >= step.shown} data-testid={`recap-card-${a.id}`}>
                <Icon name={a.icon as IconName} className="rcp-card-icon" />
                <div className="rcp-card-title">{a.title}</div>
                <div className="rcp-card-who"><Avatar name={a.name} avatar={recap.lines.find((l) => l.id === a.winner)?.avatar} size={40} /><FitName name={a.name} chars={13} lines={2} min={0.55} /></div>
                <div className="rcp-card-detail">{a.detail}</div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {step.scene === 'final' && (
        <section className="rcp-scene" key="final">
          <header className="rcp-head">
            <span className="fz-sticker yel rcp-sticker">{BRAND} · {today()}</span>
            <h1 className="rcp-title mo-pop">Who got what</h1>
          </header>
          <div className={`rcp-final${recap.awards.length ? '' : ' solo'}${recap.lines.length > 6 ? ' two' : ''}`}>
            {recap.awards.length > 0 && <ul className="rcp-awards" style={{ ['--n']: recap.awards.length } as CSSProperties}>{recap.awards.map((a, i) => <AwardRow key={a.id} a={a} i={i} />)}</ul>}
            <Lines recap={recap} />
          </div>
        </section>
      )}
    </div>
  );
}

// --- a phone (and a narrow console): one card to scroll --------------------------------------------

function RecapCard({ recap, me, where, onClose }: { recap: RecapData; me?: PlayerId; where: Where; onClose: () => void }) {
  const mine = me ? recap.lines.find((l) => l.id === me) : undefined;
  const myAwards = recap.awards.filter((a) => a.winner === me);
  const onTop = !!mine && mine.place === 1;
  return (
    <div className={`recap as-card on-${where}`} id="recap" role="dialog" aria-label="The night's recap" data-testid="recap">
      {onTop && <Confetti n={50} seed={3} />}
      <div className="rcp-scroll">
        <div className="rcp-sheet">
          <header className="rcp-hero mo-pop">
            <span className="fz-sticker yel rcp-sticker">{BRAND} · {today()}</span>
            {mine ? <>
              <div className="rcp-hero-place" data-testid="recap-place">{mine.place > 0 ? `${ordinal(mine.place)} place` : 'Game over'}</div>
              <div className="rcp-hero-score"><CountUp value={mine.score} from={0} ms={900} delay={250} className="rcp-hero-num" /> Party Score</div>
              <div className={`rcp-hero-gift${mine.gift ? '' : ' none'}`}>{mine.gift ? <><Icon name="gift" /> You went home with <strong>{mine.gift.name}</strong></> : 'No gift this year'}</div>
            </> : <div className="rcp-hero-place small">The final standings</div>}
          </header>

          {myAwards.map((a) => (
            <div key={a.id} className="rcp-mine mo-slam" data-testid="recap-my-award">
              <Icon name={a.icon as IconName} className="rcp-mine-icon" />
              <div><div className="rcp-mine-k">Your award</div><div className="rcp-mine-title">{a.title}</div><div className="rcp-mine-detail">{a.detail}</div></div>
            </div>
          ))}

          {recap.podium.length > 0 && (
            <section className="rcp-sec">
              <h2 className="rcp-h">The podium</h2>
              <Podium steps={recap.podium} up={() => true} size={(place) => (place === 1 ? 56 : 46)} />
            </section>
          )}

          {recap.awards.length > 0 && (
            <section className="rcp-sec">
              <h2 className="rcp-h">Tonight's awards</h2>
              <ul className="rcp-awards">{recap.awards.map((a, i) => <AwardRow key={a.id} a={a} i={i} />)}</ul>
            </section>
          )}

          <section className="rcp-sec">
            <h2 className="rcp-h">Who got what</h2>
            <Lines recap={recap} me={me} />
          </section>

          {recap.drawings.length > 0 && (
            <section className="rcp-sec" data-testid="recap-drawings">
              <h2 className="rcp-h">Sketch-Off{recap.prompt ? `: ${recap.prompt}` : ''}</h2>
              <ul className="rcp-gallery">
                {recap.drawings.map((d) => (
                  <li key={d.id}>
                    <img src={d.img} alt={`Drawing by ${d.name}`} />
                    <span className="rcp-gallery-name">{d.name}</span>
                    <a className="rcp-btn ghost small" href={d.img} download={`sketch-off-${d.id}.png`}>Save</a>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      </div>
      <div className="rcp-bar">
        <SaveButton recap={recap} />
        <button type="button" className="rcp-btn ghost" data-testid="recap-close" onClick={onClose}>Done</button>
      </div>
    </div>
  );
}
