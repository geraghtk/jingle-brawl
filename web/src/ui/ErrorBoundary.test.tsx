import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { Crashed, ErrorBoundary } from './ErrorBoundary';

describe('a screen that crashes while drawing', () => {
  it('is caught: the boundary switches to its own screen instead of leaving the page blank', () => {
    expect(ErrorBoundary.getDerivedStateFromError()).toEqual({ failed: true });
    const ok = new ErrorBoundary({ where: 'phone', children: 'the game' });
    expect(ok.render()).toBe('the game');
    ok.state = { failed: true };
    expect(renderToStaticMarkup(<>{ok.render()}</>)).toContain('data-testid="crashed"');
  });

  it('says the game is safe and offers a reload, in words for that screen', () => {
    const phone = renderToStaticMarkup(<Crashed where="phone" onReload={() => {}} />);
    expect(phone).toContain('your seat is kept');
    expect(phone).toContain('data-testid="crashed-reload"');
    expect(renderToStaticMarkup(<Crashed where="tv" onReload={() => {}} />)).toContain('the TV picks up where it was');
    expect(renderToStaticMarkup(<Crashed where="host" onReload={() => {}} />)).toContain('your console picks up');
  });

  it('every page is wrapped in it', () => {
    for (const page of ['host', 'player', 'tv']) expect(readFileSync(resolve(__dirname, `../${page}/main.tsx`), 'utf8'), page).toMatch(/<ErrorBoundary where="(host|phone|tv)"><App \/><\/ErrorBoundary>/);
  });
});
