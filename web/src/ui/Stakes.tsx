// What a duel was for, as an in-app game's ending says it (so nothing announces the duel again afterwards):
// what the winner takes, what it means for the loser, the small print. The words are core/stakes.ts, worked
// out by the room from resolveDuel itself; the ceremony card and AI Draw Off's results both show this.
import type { CSSProperties } from 'react';
import { stakesWords } from '../core/stakes';
import type { DuelStakes } from '../core/stakes';
import type { Game } from '../core/types';
import './ceremony.css';

export function StakesLines({ stakes, game, big, style }: { stakes: DuelStakes | null | undefined; game: Game; big?: boolean; style?: CSSProperties }) {
  if (!stakes) return null;
  const name = (id: string) => game.players?.[id]?.name || 'Someone';
  const w = stakesWords(stakes, name);
  if (!w.prize && !w.loser.length && !w.foot.length) return null;
  return (
    <div className={`cer-stakes mo-up${big ? ' big' : ''}`} style={style} data-testid="ceremony-stakes">
      {w.prize && <p className="cer-prize" data-testid="ceremony-prize">{w.prize}</p>}
      {w.loser.length > 0 && <p className="cer-loser" data-testid="ceremony-loser"><b>{name(stakes.loserId)}</b> {w.loser.join(' · ')}</p>}
      {w.foot.length > 0 && <p className="cer-foot">{w.foot.join(' · ')}</p>}
    </div>
  );
}
