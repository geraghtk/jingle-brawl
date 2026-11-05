// I lost and may take a Reindeer Reprisal: pick a target (or skip).
import * as D from '../core/duel';
import { reprisalTargets } from '../core/rules';
import { minCost } from '../core/settings';
import { MISFIT } from '../core/types';
import { usePlayer } from './PlayerContext';
import { Glyphs, Icon } from '../ui/icons';

export function ReprisalCard() {
  const { game, me, player, cfg, act } = usePlayer();
  const t = game.currentTurn!;
  const targets = reprisalTargets(game, t.excludedGiftId, me).sort(([, a], [, b]) => {
    const am = a.ownerId === MISFIT, bm = b.ownerId === MISFIT;
    if (am !== bm) return am ? 1 : -1;
    return minCost(cfg, a.naughtyLevel) - minCost(cfg, b.naughtyLevel);
  });
  const affordable = targets.filter(([, g]) => player.chips >= minCost(cfg, g.naughtyLevel));
  const skip = (label: string, extra = {}) => (
    <button className="btn btn-secondary" data-testid="skip-reprisal" onClick={() => act(D.skipReprisal)} style={extra}>{label}</button>
  );

  return (
    <div className="card action-card" id="action-card">
      <div className="card-title"><Icon name="zap" /> Your Turn!</div>
      <div id="action-title" style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.5rem' }}><Icon name="reprisal" /> Reindeer Reprisal!</div>
      <div id="action-desc" style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
        You lost the duel but get a <strong>revenge chance</strong>!<br />
        Challenge any gift (except "{game.gifts?.[t.excludedGiftId!]?.name || 'the one you lost'}").
      </div>
      <div id="action-help">
        <div className="help-box warning">
          <div className="help-title"><Icon name="idea" /> Reprisal Tips</div>
          <div className="help-content">
            <strong>Cost:</strong> Pay Minimum Cost ({cfg.baseCost} + Naughty Level) to the Bank.<br />
            <strong>Misfits:</strong> You can claim unclaimed Misfit gifts too!<br />
            <strong>Chain limit:</strong> Max {cfg.maxReprisalDepth} chained reprisals.
          </div>
        </div>
      </div>
      <div className="action-options" id="action-options">
        {!targets.length ? (
          <>
            <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '1rem' }}>No gifts available to challenge</div>
            {skip('Skip Reprisal')}
          </>
        ) : !affordable.length ? (
          <>
            <div style={{ textAlign: 'center', color: 'var(--accent-gold)', padding: '1rem' }}>
              <Icon name="sad" /> Not enough chips! You need at least {Math.min(...targets.map(([, g]) => minCost(cfg, g.naughtyLevel)))} chip(s).
            </div>
            {skip('Skip Reprisal')}
          </>
        ) : (
          <>
            {targets.map(([id, gift]) => {
              const misfit = gift.ownerId === MISFIT;
              const cost = minCost(cfg, gift.naughtyLevel);
              const afford = player.chips >= cost;
              return (
                <div key={id} className={`action-option ${afford ? '' : 'disabled'} ${misfit ? 'misfit' : ''}`} data-testid={`reprisal-target-${id}`}
                  onClick={afford ? () => act((g) => D.chooseReprisal(g, id, Date.now())) : undefined}>
                  <div className="option-header">
                    <div className="option-title">{gift.name}</div>
                    {gift.naughtyLevel > 0 && <span className="naughty-badge"><Icon name="naughty" /> {gift.naughtyLevel}</span>}
                  </div>
                  <div className="option-desc"><Glyphs>{misfit ? '🎯 Auto-claim (no duel!)' : `⚔️ Duel ${game.players?.[gift.ownerId]?.name || 'Unknown'}`}</Glyphs></div>
                  <div className={`option-cost ${afford ? '' : 'cant-afford'}`}>
                    <Icon name="moneybag" /> {cost} chip{cost > 1 ? 's' : ''}{!afford && <span>(need {cost - player.chips} more)</span>}
                  </div>
                </div>
              );
            })}
            {skip('⏭️ Skip Reprisal (Keep what you have)', { marginTop: '1rem' })}
          </>
        )}
      </div>
    </div>
  );
}
