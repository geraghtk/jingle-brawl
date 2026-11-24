// The TV's own screen for the main game (tv.html): one 16:9 picture that never scrolls, set to be read
// from across the room. A thin frame (the name, what is happening, the room code, everyone at the table
// along the bottom) around one scene per moment of the turn (host/TvScenes.tsx). The mini-games, the
// duel, the verdict and the finale lie over it full screen as before (ui/tvstage.css sizes the duel and
// the verdict for the room). No controls: the host runs the night from their own device.
import { useEffect } from 'react';
import type { ReactNode } from 'react';
import { duelAppInfo } from '../core/duelgame';
import { GAMES } from '../core/party';
import type { PartyGame } from '../core/party';
import { reprisalTargets, stealTargets, wrappedGifts } from '../core/rules';
import type { Game, PlayerId } from '../core/types';
import { Glyphs } from '../ui/icons';
import { fitText } from '../ui/tvLayout';
import '../ui/motion.css';
import '../ui/tvstage.css';
import { useHost } from './HostContext';
import { MusicToggle } from './Music';
import { GiftTiles, nameOf, Rail } from './TvParts';
import { BetweenScene, BiddingScene, DrawOffScene, EndedScene, EndgameIdleScene, LobbyScene, Paths, RevealScene, SpotScene, TiebreakDuelScene, VersusScene } from './TvScenes';

interface Scene { key: string; badge: string; body: ReactNode; focus?: (PlayerId | undefined)[] }

function sceneOf(game: Game, tiebreakLabel: string): Scene {
  const t = game.currentTurn;
  const nm = (id?: PlayerId | null) => nameOf(game, id);
  if (!t) {
    if (game.phase === 'lobby') return { key: 'lobby', badge: 'LOBBY', body: <LobbyScene /> };
    if (game.phase === 'main') return { key: `between|${wrappedGifts(game)}`, badge: 'BETWEEN TURNS', body: <BetweenScene /> };
    if (game.phase === 'endgame') return { key: 'endgame', badge: '🎰 MISFIT LOTTERY', body: <EndgameIdleScene /> };
    return { key: 'ended', badge: 'GAME OVER', body: <EndedScene /> };
  }
  const gift = t.giftId ? game.gifts?.[t.giftId]?.name : undefined;
  const fightingFor = gift ? <>Fighting for <strong>{gift}</strong></> : undefined;
  switch (t.phase) {
    case 'bidding':
      return { key: `bidding|${t.giftId}`, badge: '🎯 SEALED BIDS', body: <BiddingScene t={t} />, focus: [t.openerId] };
    case 'endgame_bidding':
      return { key: `auction|${t.misfitId}`, badge: '🎰 AUCTION', body: <BiddingScene t={t} />, focus: [t.activePlayerId] };
    case 'reveal':
      return { key: `reveal|${t.giftId}`, badge: '✉ REVEAL', body: <RevealScene t={t} />, focus: [t.openerId] };
    case 'endgame_reveal':
      return { key: `auction-reveal|${t.misfitId}`, badge: '🎰 AUCTION REVEAL', body: <RevealScene t={t} />, focus: [t.activePlayerId] };
    case 'yield_choice':
      return {
        key: 'yield_choice', badge: '🎲 SELECTED', focus: [t.selectedC1, t.selectedC2, t.openerId],
        body: <VersusScene a={t.selectedC1} b={t.selectedC2} mid="&" kicker="SELECTED" title={`Both bid ${t.bids?.[t.selectedC1!]?.amount || 0} chips`}
          foot={<>Waiting for <strong>{nm(t.openerId)}</strong>: fight or yield</>} />,
      };
    case 'duel_pending':
      return {
        key: 'duel_pending', badge: '⚔️ DUEL', focus: [t.challengerId, t.openerId],
        body: <VersusScene a={t.challengerId} b={t.openerId} kicker="THE OPENER FIGHTS" roles={['Challenger', 'Opener']} foot={fightingFor ?? 'The duel is about to start'} />,
      };
    case 'yield_pending':
      return {
        key: 'yield_pending', badge: '🏃 YIELD', focus: [t.yieldC1, t.yieldC2],
        body: <VersusScene a={t.yieldC1} b={t.yieldC2} kicker="THE OPENER YIELDS" title={`${nm(t.openerId)} steps aside`}
          foot={<>{fightingFor ?? 'They fight for the gift'}. Each pays half their bid.</>} />,
      };
    case 'tiebreaker_pending':
      return {
        key: 'tiebreaker', badge: '⚖️ TIEBREAKER',
        body: <SpotScene kicker="A TIE" title={`Starting ${tiebreakLabel}…`} sub="Everyone plays, and the best-placed tied bidder wins the tie." />,
      };
    case 'duel':
      return t.isTiebreak
        ? { key: 'tiebreak-duel', badge: '⚖️ TIEBREAK', body: <TiebreakDuelScene t={t} />, focus: [t.challengerId, t.defenderId] }
        : { key: 'duel', badge: '⚔️ DUEL', focus: [t.challengerId, t.defenderId],     // the duel itself lies over this (DuelModal)
          body: <VersusScene a={t.challengerId} b={t.defenderId} kicker="DUEL" foot={fightingFor} /> };
    case 'reprisal':
      return {
        key: 'reprisal', badge: '🦌 REPRISAL', focus: [t.reprisalPlayerId],
        body: <SpotScene who={t.reprisalPlayerId} kicker="REINDEER REPRISAL" title={`${nm(t.reprisalPlayerId)} may strike back!`}
          sub="Challenge one of these gifts, or let it go." wait={<>{nm(t.reprisalPlayerId)} is choosing on their phone</>}>
          <GiftTiles gifts={reprisalTargets(game, t.excludedGiftId, t.reprisalPlayerId!)} />
        </SpotScene>,
      };
    case 'endgame_choice':
      return {
        key: `endgame_choice|${t.activePlayerId}`, badge: '🎰 MISFIT LOTTERY', focus: [t.activePlayerId],
        body: <SpotScene who={t.activePlayerId} kicker="MISFIT LOTTERY" title={`${nm(t.activePlayerId)}'s turn`}
          sub={<>The Misfit: <strong>{t.misfitName || 'a mystery gift'}</strong></>} wait={<>{nm(t.activePlayerId)} is choosing on their phone</>}>
          <Paths who={nm(t.activePlayerId)} />
        </SpotScene>,
      };
    case 'endgame_steal_select':
      return {
        key: 'steal', badge: '🎯 STEAL', focus: [t.activePlayerId],
        body: <SpotScene who={t.activePlayerId} kicker="STEAL" title={`${nm(t.activePlayerId)} is stealing!`}
          sub="Whose gift? The loser of the duel takes the Misfit." wait={<>{nm(t.activePlayerId)} is picking a gift on their phone</>}>
          <GiftTiles gifts={stealTargets(game, t.activePlayerId!)} />
        </SpotScene>,
      };
    case 'endgame_nobids':
      return {
        key: 'nobids', badge: '🎰 NO BIDS', focus: [t.activePlayerId],
        body: <SpotScene who={t.activePlayerId} kicker="NO BIDS" title="Nobody bid for the Misfit"
          sub={<>The Misfit: <strong>{t.misfitName || 'a mystery gift'}</strong></>} wait={<>{nm(t.activePlayerId)} picks someone to duel</>} />,
      };
    case 'minigame_draw_off':
      return { key: 'draw_off', badge: '🎨 AI DRAW OFF', body: <DrawOffScene />, focus: [t.challengerId, t.defenderId] };
    case 'minigame_party': {
      // the party game itself fills the screen (host/WhackStage.tsx PartyFull); this is what is behind it
      const mg = game.minigame as PartyGame | undefined;
      const g = (mg && GAMES[mg.type]) || { emoji: '🎉', label: 'Party game' };
      return {
        key: 'party', badge: t.lab ? `🧪 GAME LAB: ${g.label.toUpperCase()}` : `${g.emoji} ${g.label.toUpperCase()}`,
        body: <SpotScene kicker={t.lab ? 'GAME LAB' : 'PARTY GAME'} title={g.label} sub={t.lab ? 'Practice: nothing is won or lost.' : 'Everyone plays!'} />,
      };
    }
    case 'minigame_arena': {
      const info = duelAppInfo(game.arenaView?.gameId);
      const label = info?.label ?? 'Mini-game';
      return {
        key: 'arena', badge: t.lab ? `🧪 GAME LAB: ${label.toUpperCase()}` : `${info?.emoji ?? '🎮'} ${label.toUpperCase()}`, focus: [t.challengerId, t.defenderId],
        body: <SpotScene kicker={t.lab ? 'GAME LAB' : 'MINI-GAME'} title={label} sub={t.lab ? 'Practice: nothing is won or lost.' : `${nm(t.challengerId)} vs ${nm(t.defenderId)}`} />,
      };
    }
    default:
      return { key: String(t.phase), badge: String(t.phase).replace(/_/g, ' ').toUpperCase(), body: <SpotScene kicker="ONE MOMENT" title="Waiting…" /> };
  }
}

/** The duel and its verdict lie over the stage as siblings (DuelModal, DuelVerdict): the names in them are set
 *  to fit from here, as sizes on the page (ui/tvstage.css reads them). */
function useOverlayNames(game: Game) {
  const t = game.currentTurn;
  const red = nameOf(game, t?.challengerId), green = nameOf(game, t?.defenderId);
  const winner = nameOf(game, game.lastDuel?.winnerId);
  useEffect(() => {
    const root = document.documentElement;
    const corner = (name: string) => String(fitText(name.toUpperCase(), { width: 62, max: 9, min: 3.2, maxLines: 2, upper: true }).size);
    root.style.setProperty('--tvs-red', corner(red));
    root.style.setProperty('--tvs-green', corner(green));
    root.style.setProperty('--tvs-verdict', String(fitText(winner.toUpperCase(), { width: 150, max: 19, min: 6, maxLines: 2, upper: true }).size));
  }, [red, green, winner]);
}

export function TvStage() {
  const { game, room, cfg } = useHost();
  const method = cfg.tiebreakerMethod;
  const scene = sceneOf(game, method === 'party_mix' || !(method in GAMES) ? 'a party game' : GAMES[method as keyof typeof GAMES].label);
  useOverlayNames(game);
  const lobby = game.phase === 'lobby';
  const focus = new Set((scene.focus || []).filter((id): id is PlayerId => !!id));
  const players = Object.keys(game.players || {}).length;
  return (
    <div id="tv-stage" className="tvs" data-testid="tv-stage" data-scene={scene.key.split('|')[0]}>
      <div className={`tvs-pic${lobby ? ' lobby' : ''}`}>
        <header className="tvs-top">
          <div className="tvs-brand"><img src="ydlogo3.png" alt="" /><span className="f6-disp">YuleDuel</span></div>
          <span key={scene.badge} className="tvs-badge mo-down" id="phase-badge"><Glyphs>{scene.badge}</Glyphs></span>
          <div className="tvs-top-cells">
            {!lobby && <div className="tvs-cell"><span className="k">Gifts to open</span><span className="v f6-num" id="strip-gifts">{wrappedGifts(game)}</span></div>}
            <div className="tvs-cell"><span className="k">Players</span><span className="v f6-num" id="connected-count">{players}</span></div>
            {!lobby && <div className="tvs-cell room"><span className="k">Room code</span><span className="v f6-num" id="strip-room">{room}</span></div>}
            <MusicToggle />
          </div>
        </header>
        <main className="tvs-hero" id="turn-panel" key={scene.key}>{scene.body}</main>
        {!lobby && <Rail focus={focus} />}
      </div>
    </div>
  );
}
