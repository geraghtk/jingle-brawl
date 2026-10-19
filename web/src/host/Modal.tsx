// The host's general-purpose modal (manual overrides, Add Gift …).
import { createContext, useContext } from 'react';
import type { ReactNode } from 'react';

export const ModalCtx = createContext<(content: ReactNode | null) => void>(() => {});
export const useModal = () => useContext(ModalCtx);

export function Modal({ content, onClose }: { content: ReactNode | null; onClose: () => void }) {
  return (
    <div className={`modal-overlay${content ? ' active' : ''}`} id="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal" id="modal-content">{content}</div>
    </div>
  );
}

/** A list of choices in the modal (Gambit / Reprisal targets, defenders, steals). */
export function OptionList({ title, intro, options, onCancel }: {
  title: string; intro: ReactNode;
  options: { key: string; title: string; desc?: string; onPick: () => void; testid?: string }[];
  onCancel: () => void;
}) {
  return (
    <>
      <h2>{title}</h2>
      <p style={{ color: 'var(--text-secondary)' }}>{intro}</p>
      <div className="options">
        {options.map((o) => (
          <div key={o.key} className="option-btn" data-testid={o.testid} onClick={o.onPick}>
            <div className="option-title">{o.title}</div>
            {o.desc && <div className="option-desc">{o.desc}</div>}
          </div>
        ))}
      </div>
      <button className="btn btn-secondary" onClick={onCancel} style={{ marginTop: '1rem' }}>Cancel</button>
    </>
  );
}
