// Sealed bid on the opened gift (main game) or the Misfit (endgame auction).
import { useState } from 'react';
import * as A from '../core/actions';
import { avatarColor } from '../ui/avatar';
import { dividendNote, taxNote } from '../ui/notes';
import { usePlayer } from './PlayerContext';

export function BidPanel() {
  const { game, me, player, cfg, act } = usePlayer();
  const t = game.currentTurn!;
  const endgame = t.phase === 'endgame_bidding';
  const locked = t.bids?.[me];
  const [editing, setEditing] = useState(false);
  const max = Math.min(player.chips, cfg.maxBid);
  const [bid, setBid] = useState(locked?.amount ?? 0);
  const amount = Math.min(bid, max);
  const sealed = !!locked && !editing;

  const gift = t.giftId ? game.gifts?.[t.giftId] : undefined;
  const giftName = endgame ? t.misfitName || 'Misfit Gift' : gift?.name || 'Mystery Gift';
  const naughty = Math.max(0, Math.min(5, gift?.naughtyLevel ?? 0));
  const sitter = endgame ? t.activePlayerId : t.openerId;
  const bidsIn = Object.keys(t.bids || {}).length;
  const eligible = Object.keys(game.players || {}).filter((id) => id !== sitter).length;

  const seal = async () => {
    if (await act((g) => A.submitBid(g, me, amount, Date.now()))) setEditing(false);
  };

  return (
    <div className="card bid-panel" id="bid-panel" data-testid="bid-panel">
      <div className="bid-kicker f6-cap">Sealed bid</div>
      <div className="bid-title f6-disp">Bid it. Mean it.</div>
      <div className="gift-being-bid">
        <div className="f6-gift mini" id="bid-gift-tile" aria-hidden="true" style={{ ['--gift-color' as string]: avatarColor(giftName) }}>
          <span className="ribbon-v" /><span className="ribbon-h" /><span className="bow" />
        </div>
        <div className="gift-lines">
          <div className="name" id="bid-gift-name">{giftName}</div>
          <div className="opener-info" id="bid-opener-info">
            {endgame ? <strong>🎰 Misfit Auction!</strong> : <>Opened by <strong>{game.players?.[t.openerId!]?.name || 'Player'}</strong> · Naughty {naughty}</>}
          </div>
        </div>
        <span className="f6-chip naughty-chip" id="bid-naughty-chip">{endgame ? '' : '●'.repeat(naughty) + '○'.repeat(5 - naughty)}</span>
      </div>

      <div className="bid-slider-container">
        <span className={`fz-sticker allin-sticker${max > 0 && amount === max ? '' : ' hidden'}`} id="allin-sticker">ALL-IN!</span>
        <div className="bid-amount-display">
          <div className="f6-cap">Your bid</div>
          <div className="amount f6-disp fz-riso f6-num" id="bid-amount-display">{amount}</div>
          <div className="label" id="bid-of-label">{max > 0 ? `of ${max} chips` : 'chips'}</div>
        </div>
        <input type="range" className="bid-slider" id="bid-slider" min={0} max={max} value={amount} disabled={sealed}
          onChange={(e) => setBid(parseInt(e.target.value))} />
        <div className="bid-quick-buttons" id="bid-quick-buttons">
          {Array.from({ length: max + 1 }, (_, i) => (
            <button key={i} data-testid={`bid-${i}`} className={amount === i ? 'selected' : ''} disabled={sealed} onClick={() => setBid(i)}>
              {i === 0 ? 'Pass' : i}
            </button>
          ))}
        </div>
      </div>

      <button className="btn btn-primary seal-btn" id="submit-bid-btn" data-testid="seal-bid" disabled={sealed} onClick={seal}>
        {sealed ? '✅ Bid Submitted' : 'Seal it — no takebacks 🔒'}
      </button>

      {sealed && (
        <div className="bid-status submitted" id="bid-status">
          <div className="sealed-state">
            <div className="sealed-title f6-disp">{locked!.amount > 0 ? <>Locked at <span style={{ color: 'var(--accent)' }}>{locked!.amount}</span>.</> : 'Passed.'}</div>
            <div className="sealed-disc"><span className="sealed-emoji">✉️</span><span className="fz-sticker mint sealed-sticker">SEALED</span></div>
            <div className="sealed-flinch f6-disp">Don't flinch.</div>
            <div className="sealed-sub">Everyone else is still sweating. Good.</div>
            <div className="final-stats" style={{ width: '100%', marginTop: 12 }}>
              <div className="fstat"><div className="fstat-v f6-disp f6-num" id="sealed-bids-in">{bidsIn}/{eligible}</div><div className="fstat-k">bids in</div></div>
              <div className="fstat"><div className="fstat-v f6-disp f6-num">{Math.max(0, player.chips - locked!.amount)} <span className="f6-coin" /></div><div className="fstat-k">chips left</div></div>
            </div>
            <button className="btn btn-secondary" data-testid="change-bid" onClick={() => { setBid(locked!.amount); setEditing(true); }} style={{ marginTop: '0.75rem', fontSize: '0.85rem' }}>🔄 Change Bid</button>
          </div>
        </div>
      )}

      <details className="howto bid-help" id="bid-help-text">
        <summary>{endgame ? 'How the Misfit auction works' : 'What your bid means'}</summary>
        {endgame ? (
          <dl className="howto-list">
            <dt>Who bids</dt><dd>Everyone, even players who already have a gift.</dd>
            <dt>Top bid</dt><dd>Duels {game.players?.[t.activePlayerId!]?.name || 'the active player'} for this Misfit.</dd>
            <dt>Win</dt><dd>You take the Misfit. If you had a gift, they may get it.</dd>
          </dl>
        ) : (
          <dl className="howto-list">
            <dt>Pass</dt><dd>Sit this gift out.</dd>
            <dt>Top bid</dt><dd>You duel the opener for it.</dd>
            <dt>Win</dt><dd>The gift is yours, and you take the pot.</dd>
            <dt>Santa Tax</dt><dd>{taxNote(cfg).replace(/^Santa Tax: /, '')}</dd>
            <dt>Lose</dt><dd>{dividendNote(cfg, false)}.</dd>
          </dl>
        )}
      </details>
    </div>
  );
}
