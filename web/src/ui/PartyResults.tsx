// A party game's ticking clock. (Its results are the shared ending now: ui/Ceremony.tsx PartyCeremony.)
import { useEffect, useState } from 'react';

/** Seconds left in the current phase, ticking. */
export function useSecondsLeft(endsAt: number | undefined, offset: number): number {
  const calc = () => Math.max(0, Math.ceil(((endsAt || 0) - (Date.now() + offset)) / 1000));
  const [s, setS] = useState(calc);
  useEffect(() => {
    setS(calc());
    const id = setInterval(() => setS(calc()), 250);
    return () => clearInterval(id);
  }, [endsAt, offset]); // eslint-disable-line react-hooks/exhaustive-deps
  return s;
}
