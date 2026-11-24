// The console's own dialogs (no native prompt / confirm / alert boxes): a confirm, a notice, and
// the gift's name, asked every turn. They are shown in the host's modal (Modal.tsx).
import { useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent, ReactNode, RefObject } from 'react';
import { Avatar } from '../ui/avatar';
import { Glyphs } from '../ui/icons';
import { useHost } from './HostContext';
import { useModal } from './Modal';
import './dialogs.css';

/** The longest gift name the box takes: what fits a gift card on a phone. */
export const GIFT_NAME_MAX = 40;

/** Escape closes the dialog (a click outside it already does: Modal.tsx). */
function useEscape(onClose: () => void) {
  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); onClose(); } };
    document.addEventListener('keydown', key);
    return () => document.removeEventListener('keydown', key);
  }, [onClose]);
}

/** Put the focus in the dialog as it opens. The modal fades in from hidden, and a hidden thing
 *  cannot take the focus, so try again each frame until it has it (a few frames at most). */
function useFocus(ref: RefObject<HTMLElement>) {
  useEffect(() => {
    let frame = 0, tries = 0;
    const go = () => {
      ref.current?.focus();
      if (document.activeElement !== ref.current && tries++ < 40) frame = requestAnimationFrame(go);
    };
    go();
    return () => cancelAnimationFrame(frame);
  }, [ref]);
}

export interface ConfirmOptions {
  title: string;
  body?: ReactNode;
  /** The button that goes ahead, e.g. "End the game". */
  confirm: string;
  cancel?: string;
  /** Cannot be taken back: the button is red and Cancel has the focus. */
  danger?: boolean;
  onConfirm: () => void;
}

export function ConfirmDialog({ title, body, confirm, cancel = 'Cancel', danger, onConfirm, onClose }: ConfirmOptions & { onClose: () => void }) {
  const first = useRef<HTMLButtonElement>(null);
  useEscape(onClose);
  useFocus(first);
  return (
    <div className={`dlg${danger ? ' dlg-danger' : ''}`} role="alertdialog" aria-modal="true" aria-label={title} data-testid="dialog">
      <h2 className="dlg-title"><Glyphs>{title}</Glyphs></h2>
      {body && <p className="dlg-body">{body}</p>}
      <div className="dlg-actions">
        <button type="button" className="btn btn-secondary" data-testid="dialog-cancel" ref={danger ? first : undefined} onClick={onClose}>{cancel}</button>
        <button type="button" className={`btn ${danger ? 'btn-danger' : 'btn-primary'}`} data-testid="dialog-confirm" ref={danger ? undefined : first}
          onClick={() => { onClose(); onConfirm(); }}>{confirm}</button>
      </div>
    </div>
  );
}

/** Something the host should know; nothing to decide. */
export function NoticeDialog({ title, body, onClose }: { title: string; body?: ReactNode; onClose: () => void }) {
  const ok = useRef<HTMLButtonElement>(null);
  useEscape(onClose);
  useFocus(ok);
  return (
    <div className="dlg" role="alertdialog" aria-modal="true" aria-label={title} data-testid="dialog">
      <h2 className="dlg-title"><Glyphs>{title}</Glyphs></h2>
      {body && <p className="dlg-body">{body}</p>}
      <div className="dlg-actions">
        <button type="button" className="btn btn-primary" data-testid="dialog-ok" ref={ok} onClick={onClose}>OK</button>
      </div>
    </div>
  );
}

/** The gift somebody just drew: its name, then the turn starts. `seed` is the draw the room will
 *  make, so the player named here is the one it draws. */
export function GiftDialog({ pid, seed, onClose }: { pid: string | undefined; seed: number; onClose: () => void }) {
  const { game, send } = useHost();
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  useEscape(onClose);
  useFocus(input);
  const player = pid ? game.players?.[pid] : undefined;
  const who = player?.name || 'Someone';
  const clean = name.trim().slice(0, GIFT_NAME_MAX);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!clean || busy) return;
    setBusy(true);
    setFailed(false);
    if (await send({ type: 'draw', giftName: clean, seed })) return onClose();
    setBusy(false);
    setFailed(true);
  };
  return (
    <form className="dlg dlg-gift" role="dialog" aria-modal="true" aria-label={`${who} was drawn`} data-testid="gift-dialog" onSubmit={submit}>
      <div className="dlg-who mo-pop">
        <Avatar name={who} avatar={player?.avatar} size={72} />
      </div>
      <h2 className="dlg-title" data-testid="gift-drawn"><span className="dlg-name">{who}</span> was drawn!</h2>
      <label className="dlg-body" htmlFor="gift-name">What gift are they opening?</label>
      <div className="dlg-field">
        <input ref={input} id="gift-name" data-testid="gift-name" className="input" type="text" autoComplete="off" enterKeyHint="go"
          placeholder="Fuzzy socks" maxLength={GIFT_NAME_MAX} value={name} onChange={(e) => setName(e.target.value)} />
        {name.length >= GIFT_NAME_MAX - 10 && <span className="dlg-count">{GIFT_NAME_MAX - name.length}</span>}
      </div>
      {failed && <p className="dlg-error" data-testid="gift-error">That did not go through. Try again.</p>}
      <div className="dlg-actions">
        <button type="button" className="btn btn-secondary" data-testid="gift-cancel" onClick={onClose}>Cancel</button>
        <button type="submit" className="btn btn-gold" data-testid="gift-submit" disabled={!clean || busy}>Start bidding</button>
      </div>
    </form>
  );
}

/** `confirm(...)` and `notice(...)` for a host component: they open in the console's modal. */
export function useDialogs() {
  const setModal = useModal();
  return useMemo(() => {
    const close = () => setModal(null);
    return {
      close,
      confirm: (o: ConfirmOptions) => setModal(<ConfirmDialog key={Date.now()} {...o} onClose={close} />),
      notice: (title: string, body?: ReactNode) => setModal(<NoticeDialog key={Date.now()} title={title} body={body} onClose={close} />),
    };
  }, [setModal]);
}
