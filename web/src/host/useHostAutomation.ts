// What the host page still does on its own: the mascots react to the game. Every automatic game
// step (party-game phases, the duel vote, the wheel, duel games, ties, the AI critic) runs on the
// room server (core/hostauto.ts, server/src/room/actor.ts).
import { useEffect } from 'react';
import { observe as mascotsObserve } from '../ui/mascots';
import { useHost } from './HostContext';

export function useHostAutomation() {
  const { game } = useHost();
  useEffect(() => { mascotsObserve(game); }, [game]);
}
