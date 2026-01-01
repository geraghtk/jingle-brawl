// Firebase Configuration for Jingle Brawl
// 
// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyDOPkYyZ6qFNoBB7iVbZRtyjk3W_uf7POA",
    authDomain: "jingle-brawl-e8308.firebaseapp.com",
    databaseURL: "https://jingle-brawl-e8308-default-rtdb.firebaseio.com",
    projectId: "jingle-brawl-e8308",
    storageBucket: "jingle-brawl-e8308.firebasestorage.app",
    messagingSenderId: "778188643750",
    appId: "1:778188643750:web:be573c1c5f88cd61a48367"
};

// Game configuration defaults
const GAME_CONFIG = {
    // Whether players can see each other's chip counts
    publicChips: false,
    
    // Starting chips per player (auto-calculated based on player count)
    getStartingChips: (playerCount) => playerCount <= 10 ? 10 : 12,
    
    // Base challenge cost (for Gambits, Reprisals, Endgame steals)
    baseCost: 1,
    
    // Minimum cost formula (uses baseCost setting)
    getMinCost: function(naughtyLevel) { 
        return (this.baseCost || 1) + naughtyLevel; 
    },
    
    // Max chips winner receives from pot (rest goes to bank as Santa Tax)
    maxWinnerPayout: 2,
    
    // Santa Tax kicks in at this pot size
    taxThreshold: 2,
    
    // Loser dividend amount
    loserDividend: 1,
    
    // Who gets loser dividend: 'everyone', 'defenders', 'nobody'
    loserDividendRule: 'defenders',
    
    // Max reprisal chain depth (0=none, 1=single, 2=can chain once)
    maxReprisalDepth: 1,
    
    // Maximum bid amount
    maxBid: 4
};
