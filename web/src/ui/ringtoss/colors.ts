// Ring Toss: each player's ring colour, the same on the TV, the console and their phone (by their
// place among the game's participants).
export const RING_COLORS = ['#ff3366', '#18b88a', '#8a7bff', '#ffb000', '#38b6e6', '#ff7a59', '#c04fd6', '#2f6fde', '#7cc242', '#e0457b', '#00a3a3', '#9b6b3c'];
/** Which colour (its index: the scene's gradients are numbered by it). */
export const colorIndex = (participants: string[], pid: string) => Math.max(0, participants.indexOf(pid)) % RING_COLORS.length;
export const ringColor = (participants: string[], pid: string) => RING_COLORS[colorIndex(participants, pid)];
