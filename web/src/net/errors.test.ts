// A page's error report: what is kept, what is dropped (the server side: server/src/node/observability.test.ts).
import { describe, expect, it } from 'vitest';
import { toReport } from './errors';

describe('the page side of error reports', () => {
  it('keeps the message and stack, drops cross-origin noise', () => {
    const err = new Error('boom');
    expect(toReport('host', err)).toMatchObject({ page: 'host', message: 'boom' });
    expect(toReport('host', err)!.stack).toContain('boom');
    expect(toReport('tv', null, 'Script error.')).toBeNull();
    expect(toReport('tv', 'plain rejection')).toEqual({ page: 'tv', message: 'plain rejection' });
  });
});
