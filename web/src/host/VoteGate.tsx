// The spectators' vote on the TV: who has called the duel, who is still deciding, and how long
// the host waits. The winner buttons stay locked until the table has had its say.
import { voteStatus } from '../core/voting';
import { Avatar } from '../ui/avatar';
import { useSecondsLeft } from '../ui/PartyResults';
import { useServerOffset } from '../net/hooks';
import { useHost } from './HostContext';

export function useVoteGate() {
  const { game } = useHost();
  const offset = useServerOffset();
  const endsAt = game.currentTurn?.voteEndsAt;
  const secondsLeft = useSecondsLeft(endsAt, offset); // re-renders while the clock runs
  const status = voteStatus(game, Date.now() + offset);
  return { status, secondsLeft, locked: status.blocking, now: () => Date.now() + offset };
}

export function VotePanel() {
  const { game, send, tv } = useHost();
  const { status, secondsLeft, locked } = useVoteGate();
  if (!status.eligible.length) return null;
  const t = game.currentTurn!;
  const pick = (id: string) => t.duelVotes?.[id]?.votedFor;
  return (
    <div className={`vote-gate${locked ? ' open' : ''}`} data-testid="vote-gate">
      <div className="vote-gate-head">
        <span className="vote-gate-title">
          {locked ? 'The table is calling it' : !status.waiting.length ? 'Votes are in' : status.accepting ? 'Voting open' : 'Voting closed'}
        </span>
        <span className="vote-gate-count f6-num">{status.voted.length}/{status.eligible.length}</span>
        {locked && status.endsAt != null && <span className="vote-gate-clock f6-num" aria-label={`${secondsLeft} seconds left`}>{secondsLeft}s</span>}
      </div>
      <div className="vote-gate-seats">
        {status.eligible.map((id) => (
          <span key={id} className={`vote-seat${pick(id) ? ' in' : ''}`} data-voter={id}>
            <Avatar name={game.players?.[id]?.name || '?'} avatar={game.players?.[id]?.avatar} size={26} />
            <span className="vote-seat-name">{game.players?.[id]?.name}</span>
            {/* who they backed stays secret until the duel is called */}
            <span className="vote-seat-state">{pick(id) ? '✓' : '…'}</span>
          </span>
        ))}
      </div>
      {locked && !tv && (
        <button className="vote-gate-close" data-testid="close-voting" onClick={() => send({ type: 'close_voting' })}>
          Close voting now
        </button>
      )}
    </div>
  );
}
