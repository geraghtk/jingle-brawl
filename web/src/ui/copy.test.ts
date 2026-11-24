// House style: no em dashes in anything players see. Every page, rule doc and AI prompt is
// scanned with its comments stripped (comments are for developers and may keep theirs).
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = resolve(__dirname, '../../..');

const ORIGINAL_PAGES = [
  'web/public/rules.html', 'web/public/quick-reference.html', 'web/public/privacy.html', 'web/public/pages.css',
  'Jingle Brawl V2.md', 'ai_art_critic.py', 'server/src/critic/critic.ts',
  'web/host.html', 'web/player.html', 'web/tv.html',
];

/** The React app: everything under web/src except the simulator and tests. */
function reactSources(dir = join(ROOT, 'web/src')): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) return f === 'sim' ? [] : reactSources(p);
    if (!/\.(tsx?|css)$/.test(f) || /\.test\.tsx?$/.test(f)) return [];
    return [relative(ROOT, p).split(sep).join('/')];
  });
}

/** Drop comments (block, line, HTML, Python) so only shipped text is left. Errs on stripping too much. */
function stripComments(src: string, file: string): string {
  let s = src;
  if (file.endsWith('.md')) return s;
  if (file.endsWith('.py')) return s.replace(/(^|\s)#.*$/gm, '$1');
  const blank = (m: string) => m.replace(/[^\n]/g, ''); // keep line numbers
  if (file.endsWith('.html')) s = s.replace(/<!--[\s\S]*?-->/g, blank);
  s = s.replace(/\/\*[\s\S]*?\*\//g, blank);
  if (!file.endsWith('.css')) s = s.replace(/(^|[^:\\'"`])\/\/.*$/gm, '$1');
  return s;
}

// An em dash, or an en dash spaced like one (ranges like 1–5 are fine)
const DASHES = new RegExp(`${String.fromCharCode(0x2014)}|\\s${String.fromCharCode(0x2013)}\\s`);

describe('player-facing copy has no em dashes', () => {
  const files = [...ORIGINAL_PAGES, ...reactSources()];
  it('scans the React app as well as the original pages', () => {
    expect(files).toContain('web/src/host/TurnPanel.tsx');
    expect(files.some((f) => f.startsWith('web/src/sim/'))).toBe(false);
  });
  it.each(files)('%s', (file) => {
    const text = stripComments(readFileSync(join(ROOT, file), 'utf8'), file);
    const hits = text.split(/\r?\n/).map((l, i) => [i + 1, l.trim()] as const).filter(([, l]) => DASHES.test(l));
    expect(hits.map(([n, l]) => `${n}: ${l.slice(0, 120)}`)).toEqual([]);
  });
  it('the dash check and comment stripper catch what they should and keep strings and URLs', () => {
    const em = String.fromCharCode(0x2014);
    expect(DASHES.test(`a ${em} b`)).toBe(true);
    expect(DASHES.test(`a ${String.fromCharCode(0x2013)} b`)).toBe(true);
    expect(DASHES.test(`pages 1${String.fromCharCode(0x2013)}5`)).toBe(false);
    expect(stripComments(`say('a ${em} b'); // c ${em} d`, 'x.js')).toContain(em);
    expect(stripComments(`say('a'); // c ${em} d`, 'x.js')).not.toContain(em);
    expect(stripComments(`fetch("http://x") /* ${em} */`, 'x.ts')).toBe('fetch("http://x") ');
    expect(stripComments(`<p>ok</p><!-- ${em} -->`, 'x.html')).toBe('<p>ok</p>');
    expect(stripComments(`x = "a" # ${em}`, 'x.py')).not.toContain(em);
  });
});

// The product is YuleDuel (one word; the owner, 2026-11-23). The names it had before must not come
// back on a screen, a page or in what the critic says. (Storage keys like jingleBrawl_host_room and
// the rules file's name are not the product's name: neither has the space.)
const OLD_NAMES = /jingle brawl|yuel|yule duel(?!l)/i;

describe('the product is called YuleDuel everywhere a player looks', () => {
  const files = [...ORIGINAL_PAGES, ...reactSources()].filter((f) => f !== 'web/src/ui/copy.test.ts');
  it.each(files)('%s', (file) => {
    const text = stripComments(readFileSync(join(ROOT, file), 'utf8'), file);
    const hits = text.split(/\r?\n/).map((l, i) => [i + 1, l.trim()] as const).filter(([, l]) => OLD_NAMES.test(l.slice(0, 2000)));
    expect(hits.map(([n, l]) => `${n}: ${l.slice(0, 120)}`)).toEqual([]);
  });
  it('the pages say the name', () => {
    for (const page of ['web/host.html', 'web/player.html', 'web/tv.html', 'web/public/rules.html', 'web/public/quick-reference.html', 'web/public/privacy.html']) {
      expect(readFileSync(join(ROOT, page), 'utf8'), page).toMatch(/<title>YuleDuel\b/);
    }
  });
  it('catches the old names and lets the new one, the award and the storage keys through', () => {
    for (const old of ['Jingle Brawl', 'Yuel Duel', 'Yule Duel', 'yuel-duel']) expect(OLD_NAMES.test(old)).toBe(true);
    for (const ok of ['YuleDuel', 'Yule Dueller', 'jingleBrawl_host_room', 'Jingle Chips']) expect(OLD_NAMES.test(ok)).toBe(false);
  });
  it('the retired Google Docs export is gone and nothing links to it', () => {
    expect(readdirSync(join(ROOT, 'web/public'))).not.toContain('JingleBrawlV2.html');
    for (const page of ['rules.html', 'quick-reference.html', 'privacy.html']) {
      expect(readFileSync(join(ROOT, 'web/public', page), 'utf8')).not.toContain('JingleBrawlV2');
    }
  });
  it('the rule pages share one stylesheet and carry no styles or scripts of their own', () => {
    for (const page of ['rules.html', 'quick-reference.html', 'privacy.html']) {
      const html = readFileSync(join(ROOT, 'web/public', page), 'utf8');
      expect(html, page).toContain('<link rel="stylesheet" href="pages.css">');
      expect(html, page).not.toMatch(/<style|style="|<script(?![^>]*src=)|\son[a-z]+=/);
    }
  });
});
