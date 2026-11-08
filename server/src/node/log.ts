// What the server says about itself: one JSON object per line on stdout (any log collector reads
// that), and counters for /metrics. Never names, drawings, AI keys or client addresses: room codes,
// counts and error messages only.

export type Fields = Record<string, unknown>;
export interface Logger {
  info(event: string, fields?: Fields): void;
  warn(event: string, fields?: Fields): void;
  error(event: string, fields?: Fields): void;
}

export function jsonLogger(write: (line: string) => void = (l) => process.stdout.write(l + '\n'), now: () => number = Date.now): Logger {
  const at = (level: string) => (event: string, fields: Fields = {}) => {
    try { write(JSON.stringify({ t: new Date(now()).toISOString(), level, event, ...fields })); } catch { /* never let logging break a request */ }
  };
  return { info: at('info'), warn: at('warn'), error: at('error') };
}

export const silentLogger: Logger = { info() {}, warn() {}, error() {} };

/** Counters and gauges, rendered in Prometheus' text format. */
export class Metrics {
  private counters = new Map<string, number>();
  private gauges = new Map<string, () => number>();

  inc(name: string, labels: Record<string, string> = {}, by = 1): void {
    const key = name + fmt(labels);
    this.counters.set(key, (this.counters.get(key) || 0) + by);
  }

  gauge(name: string, read: () => number): void {
    this.gauges.set(name, read);
  }

  get(name: string, labels: Record<string, string> = {}): number {
    return this.counters.get(name + fmt(labels)) || 0;
  }

  render(): string {
    const lines: string[] = [];
    const typed = new Set<string>();
    for (const [key, v] of [...this.counters].sort()) {
      const name = key.split('{')[0];
      if (!typed.has(name)) { lines.push(`# TYPE ${name} counter`); typed.add(name); }
      lines.push(`${key} ${v}`);
    }
    for (const [name, read] of this.gauges) lines.push(`# TYPE ${name} gauge`, `${name} ${read()}`);
    return lines.join('\n') + '\n';
  }
}

const fmt = (labels: Record<string, string>) => {
  const e = Object.entries(labels);
  return e.length ? `{${e.map(([k, v]) => `${k}="${String(v).replace(/["\\\n]/g, '_')}"`).join(',')}}` : '';
};
