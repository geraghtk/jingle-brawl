// The duel modal on the TV: red corner vs green corner, pot and tax, "X wins" buttons.
import { resolveDuel } from '../core/duel';
import { startDuelApp } from '../core/duelgame';
import { DuelGameStage, useDuelGame } from './DuelGameStage';
import { freshRng, useHost } from './HostContext';
import { useVoteGate, VotePanel } from './VoteGate';
import { Avatar } from '../ui/avatar';

export function DuelModal() {
  const { game, act, takeSnapshot } = useHost();
  const t = game.currentTurn;
  const open = t?.phase === 'duel' && !t.isTiebreak;
  const name = (id?: string) => (id && game.players?.[id]?.name) || 'Unknown';
  const gift = t?.giftId ? game.gifts?.[t.giftId] : undefined;
  const pot = t?.pot || 0;
  const tax = t?.tax || 0;
  const { locked: votesPending, now } = useVoteGate();
  const { ready, choice } = useDuelGame();
  const locked = votesPending || !ready;
  const app = ready && choice?.kind === 'app' ? choice : null; // not before the wheel stops (spoiler)
  const win = (id: string) => { if (locked) return; takeSnapshot(); act((g) => resolveDuel(g, id, now())); };
  return (
    <div className={`modal-overlay${open ? ' active' : ''}`} id="duel-modal-overlay">
      <div className="modal duel-modal" id="duel-modal-content">
        {open && t && (
          <>
            <div className="duel-v6">
              <div className="duel-corner red" data-duel-corner={t.challengerId}>
                <div className="fz-halftone corner-halftone" />
                <span className="fz-sticker ink corner-sticker">RED CORNER</span>
                <div className="corner-avatar"><Avatar name={name(t.challengerId)} avatar={game.players?.[t.challengerId!]?.avatar} size={96} /></div>
                <div className="corner-name f6-disp">{name(t.challengerId).toUpperCase()}</div>
                <div className="corner-role">Challenger · put chips on the line</div>
              </div>
              <div className="duel-vs-strip"><span className="duel-vs-text f6-disp fz-riso f6-wiggle">VS</span></div>
              <div className="duel-corner green" data-duel-corner={t.defenderId}>
                <div className="fz-halftone corner-halftone" />
                <span className="fz-sticker ink corner-sticker">GREEN CORNER</span>
                <div className="corner-avatar"><Avatar name={name(t.defenderId)} avatar={game.players?.[t.defenderId!]?.avatar} size={96} /></div>
                <div className="corner-name f6-disp">{name(t.defenderId).toUpperCase()}</div>
                <div className="corner-role">{t.isYieldDuel ? 'Challenger · put chips on the line' : 'Defender · defends for free'}</div>
              </div>
            </div>
            <DuelGameStage />
            {ready && <VotePanel />}
            <div className="duel-footer">
              {gift && <div className="cell"><span className="k">Fighting for</span><span className="v">{gift.name}</span></div>}
              <div className="cell" data-duel-pot><span className="k">Pot</span><span className="v f6-num">{pot + tax}</span></div>
              {tax > 0 && <div className="cell"><span className="k">Santa tax</span><span className="v f6-num">−{tax}</span></div>}
              <div className="cell"><span className="k">To winner</span><span className="v f6-num">{pot}</span></div>
              <div className="cell duel-actions">
                <span className="k">{!ready ? 'First: which duel?' : votesPending ? (app ? 'The game starts once the votes are in' : 'Winner buttons unlock once the votes are in') : app ? `${app.name} decides it` : 'Duel it out, then click the winner'}</span>
                {app ? (
                  <div className="duel-actions-row">
                    <button className="duel-win-btn duel-app-btn" disabled={locked} data-testid="start-duel-app"
                      onClick={() => { takeSnapshot(); act((g) => startDuelApp(g, { now: now(), rng: freshRng() })); }}>▶ Start {app.name}</button>
                  </div>
                ) : (
                <div className="duel-actions-row">
                  <button className="duel-win-btn red" disabled={locked} data-testid={`modal-winner-${t.challengerId}`} onClick={() => win(t.challengerId!)}>{name(t.challengerId)} wins</button>
                  <button className="duel-win-btn green" disabled={locked} data-testid={`modal-winner-${t.defenderId}`} onClick={() => win(t.defenderId!)}>{name(t.defenderId)} wins</button>
                </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
