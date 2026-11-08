// What the duellists play, on the TV: the defender picking (with a host override), the wheel
// spinning, then the duel itself. The winner buttons wait until it's known.
import { DUEL_APPS, duelGameStep, duelOptions, duelPicker, SPIN_MS, UNDER } from '../core/duelgame';
import { useServerOffset } from '../net/hooks';
import { DuelWheel } from '../ui/DuelWheel';
import { useSecondsLeft } from '../ui/PartyResults';
import { useHost } from './HostContext';
import { Glyphs, Icon } from '../ui/icons';

export function useDuelGame() {
  const { game } = useHost();
  const offset = useServerOffset();
  const t = game.currentTurn;
  const step = duelGameStep(game);
  const spunAt = typeof t?.wheel?.spunAt === 'number' ? t.wheel.spunAt : undefined;
  const spinLeft = useSecondsLeft(spunAt != null ? spunAt + SPIN_MS : undefined, offset); // re-render at the stop
  const spinning = spunAt != null && spinLeft > 0;
  return { step, choice: t?.duelGame, spinning, ready: !step && !spinning, offset };
}

export function DuelGameStage({ compact = false }: { compact?: boolean }) {
  const { game, send } = useHost();
  const t = game.currentTurn;
  const { step, choice, spinning, offset } = useDuelGame();
  const name = (id?: string) => (id && game.players?.[id]?.name) || 'Someone';
  if (!t || (!step && !choice)) return null; // duel picking is off

  const picker = duelPicker(game);
  const youngPick = picker != null && (game.players?.[picker]?.age ?? 99) < UNDER;
  if (step === 'picker') {
    return (
      <div className="duel-game-stage picking" data-testid="duel-game-stage">
        <div className="dgs-title"><Icon name="target" /> {name(picker!)} is picking the duel{youngPick && <span className="dgs-rule"> · under-16 rule</span>}</div>
        <div className="dgs-options">
          {duelOptions(game).map((o) => <span key={o.name} className="dgs-option">{o.kind === 'app' && o.app && <><Glyphs>{DUEL_APPS[o.app].emoji}</Glyphs> </>}{o.name}</span>)}
        </div>
        <button className="vote-gate-close" data-testid="host-spin" onClick={() => send({ type: 'host_spin' })}>
          Spin the wheel for them
        </button>
      </div>
    );
  }
  if (step === 'wheel' || (spinning && t.wheel)) {
    return (
      <div className="duel-game-stage wheel" data-testid="duel-game-stage">
        {t.wheel ? <DuelWheel spin={t.wheel} offset={offset} size={compact ? 240 : 300} /> : <div className="dgs-title"><Icon name="wheel" /> Spinning the wheel…</div>}
      </div>
    );
  }
  return (
    <div className="duel-game-stage chosen" data-testid="duel-game-stage">
      <span className="dgs-k">The duel</span>
      <span className="dgs-game f6-disp" data-testid="duel-game-name">{choice!.kind === 'app' && choice!.app && <><Glyphs>{DUEL_APPS[choice!.app].emoji}</Glyphs> </>}{choice!.name}</span>
      <span className="dgs-by">{choice!.via === 'wheel' ? 'the wheel picked' : `${name(picker ?? t.defenderId)} picked it`}</span>
    </div>
  );
}
