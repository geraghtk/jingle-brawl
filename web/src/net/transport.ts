// Which back end the screens talk to: the room server ('server', the future) or Firebase (until
// the switch). Fixed for the page's life: ?transport=server|firebase, else VITE_TRANSPORT.
const asked = new URLSearchParams(location.search).get('transport') ?? import.meta.env.VITE_TRANSPORT;
export const TRANSPORT: 'server' | 'firebase' = asked === 'server' ? 'server' : 'firebase';
export const onServer = TRANSPORT === 'server';
