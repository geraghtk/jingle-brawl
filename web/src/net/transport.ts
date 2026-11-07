// Which back end the screens talk to: the room server (the default) or Firebase (being retired).
// Fixed for the page's life: ?transport=firebase|server, else VITE_TRANSPORT.
const asked = new URLSearchParams(location.search).get('transport') ?? import.meta.env.VITE_TRANSPORT;
export const TRANSPORT: 'server' | 'firebase' = asked === 'firebase' ? 'firebase' : 'server';
export const onServer = TRANSPORT === 'server';
