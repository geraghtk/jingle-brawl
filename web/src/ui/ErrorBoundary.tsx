// If a screen's code throws while drawing, React takes the whole page away and leaves it blank: at a
// party that is a phone or a TV that has "died" with no way back. This catches it and says what to do
// instead. The game itself lives on the room server, so reloading the page puts the screen straight back
// where it was (the phone keeps its seat). Wrapped round each page's app in its main.tsx.
import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import './errorBoundary.css';

/** What the stuck screen says (exported for the test). */
export function Crashed({ where, onReload }: { where: 'phone' | 'host' | 'tv'; onReload: () => void }) {
  return (
    <div className="crashed" role="alert" data-testid="crashed">
      <div className="crashed-card">
        <div className="crashed-face" aria-hidden>:(</div>
        <h1>This screen tripped over.</h1>
        <p>{where === 'tv' ? 'The game is safe. Reload this page and the TV picks up where it was.'
          : where === 'host' ? 'The game is safe on the server. Reload and your console picks up where it was.'
          : 'The game is safe and your seat is kept. Reload and you are straight back in.'}</p>
        <button type="button" className="crashed-btn" data-testid="crashed-reload" onClick={onReload}>Reload</button>
      </div>
    </div>
  );
}

interface Props { where: 'phone' | 'host' | 'tv'; children: ReactNode; onError?: (e: Error) => void }

export class ErrorBoundary extends Component<Props, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError(): { failed: boolean } { return { failed: true }; }
  componentDidCatch(error: Error, info: ErrorInfo): void {
    // for whoever looks at the console; nothing about the game or its players is sent anywhere
    console.error('A screen crashed while drawing:', error, info.componentStack);
    this.props.onError?.(error);
  }
  render(): ReactNode {
    if (!this.state.failed) return this.props.children;
    return <Crashed where={this.props.where} onReload={() => window.location.reload()} />;
  }
}
