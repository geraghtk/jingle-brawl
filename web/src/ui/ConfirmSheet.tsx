// The phone's own "are you sure?": a sheet that comes up from the bottom of the screen, in the
// app's look (no native confirm() box). Escape or a tap outside it is "no".
import { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { Glyphs } from './icons';
import './motion.css';
import './confirmSheet.css';

export function ConfirmSheet({ title, body, confirm, cancel = 'Cancel', danger, onConfirm, onClose, testid = 'sheet' }: {
  title: string; body?: ReactNode;
  /** The button that goes ahead, e.g. "Leave game". */
  confirm: string; cancel?: string;
  /** Hard to take back: the button is red and the other one has the focus. */
  danger?: boolean;
  onConfirm: () => void; onClose: () => void; testid?: string;
}) {
  const safe = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    safe.current?.focus();
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); onClose(); } };
    document.addEventListener('keydown', key);
    return () => document.removeEventListener('keydown', key);
  }, [onClose]);
  return (
    <div className="sheet-veil" data-testid={`${testid}-veil`} onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className={`sheet mo-up${danger ? ' sheet-danger' : ''}`} role="alertdialog" aria-modal="true" aria-label={title} data-testid={testid}>
        <span className="sheet-grip" aria-hidden />
        <h2 className="sheet-title"><Glyphs>{title}</Glyphs></h2>
        {body && <p className="sheet-body">{body}</p>}
        <div className="sheet-actions">
          <button type="button" className={`btn ${danger ? 'sheet-go-danger' : 'btn-primary'}`} data-testid={`${testid}-confirm`} onClick={onConfirm}>{confirm}</button>
          <button type="button" className="btn btn-secondary" data-testid={`${testid}-cancel`} ref={safe} onClick={onClose}>{cancel}</button>
        </div>
      </div>
    </div>
  );
}
