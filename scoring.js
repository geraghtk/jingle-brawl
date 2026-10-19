// Party Score + spectator bets — pure rules shared by host.html, player.html and Node tests
// (scoring.test.js).
//
//   Party Score = party points (3/2/1 from party mini-games) + 1 per correct duel call.
//   The highest Party Score at the final tally is the Party MVP (ties share it).
//
//   Spectator bets (host setting, experimental): each spectator vote stakes 1 chip. The
//   wrong callers' chips are split evenly between the right callers (who also keep their
//   own stake); an odd chip left over goes to the North Pole Fund. Nobody right → all to
//   the Fund. A vote placed with 0 chips stakes nothing (it still counts for Party Score).
(function (root) {
  const partyScore = (player) => (player?.partyPoints || 0) + (player?.votingWins || 0);

  /** Ids of the Party MVP(s) — only if someone actually scored. */
  function partyMvps(players) {
    const scores = Object.entries(players || {}).map(([id, p]) => [id, partyScore(p)]);
    const best = Math.max(0, ...scores.map(([, s]) => s));
    return best > 0 ? scores.filter(([, s]) => s === best).map(([id]) => id) : [];
  }

  /**
   * Settle one duel's bets. votes = { voterId: { votedFor, stake } }.
   * Returns { delta: { voterId: chipChange }, pool, toBank, winners }.
   */
  function settleBets(votes, winnerId) {
    const delta = {};
    const staked = Object.entries(votes || {}).filter(([, v]) => (v?.stake || 0) > 0);
    const right = staked.filter(([, v]) => v.votedFor === winnerId).map(([id]) => id);
    const wrong = staked.filter(([, v]) => v.votedFor !== winnerId).map(([id]) => id);
    const pool = wrong.length; // 1 chip each
    for (const id of wrong) delta[id] = -1;
    let toBank = pool;
    if (right.length) {
      const share = Math.floor(pool / right.length);
      for (const id of right) delta[id] = share;
      toBank = pool - share * right.length;
    }
    return { delta, pool, toBank, winners: right.length };
  }

  const api = { partyScore, partyMvps, settleBets };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.Scoring = api;
})(typeof window !== 'undefined' ? window : globalThis);
