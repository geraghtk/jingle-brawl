// A player's or a gift's name drawn large, at a size that fits it (ui/nameFit.ts). It goes INSIDE the
// element that sets the place's font size and colour: <h1 className="verdict-name"><FitName ... /></h1>.
import { fitName } from './nameFit';
import type { FitSpec } from './nameFit';
import './fitName.css';

export function FitName({ name, chars, lines, min, upper, inline, className, testid }: FitSpec & {
  name: string;
  /** Drawn in capitals (by css: the text itself stays as typed), so measured as capitals. */
  upper?: boolean;
  /** Stay in the line of text around it (default: a block of its own). */
  inline?: boolean;
  className?: string;
  testid?: string;
}) {
  const fit = fitName(upper ? name.toUpperCase() : name, { chars, lines, min });
  return (
    <span className={`fit-name${inline ? ' fit-inline' : ''}${upper ? ' fit-upper' : ''}${fit.whole ? '' : ' fit-break'}${className ? ` ${className}` : ''}`}
      style={{ fontSize: `${fit.scale}em` }} data-fit={fit.scale} data-testid={testid}>{name}</span>
  );
}
