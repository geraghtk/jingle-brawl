// Firebase Configuration for Yuel Duel
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

// On localhost both the database and auth go to the local emulators (see firebase.json)
// so testing never touches production data OR creates real anonymous users in the
// production Auth project. Add ?prod=1 to the URL to opt out.
function useLocalEmulators() {
    const isLocalhost = ['localhost', '127.0.0.1'].includes(location.hostname);
    return isLocalhost && !new URLSearchParams(location.search).has('prod');
}

// Initialize Firebase and return the Realtime Database handle.
function connectGameDatabase() {
    firebase.initializeApp(firebaseConfig);
    const db = firebase.database();
    if (useLocalEmulators()) {
        db.useEmulator('127.0.0.1', 9000);
        console.info('Firebase: using local emulators (start them with "npm run emulator"; add ?prod=1 for production)');
    }
    return db;
}

// Return the Auth handle (call after connectGameDatabase, before signing in).
function connectGameAuth() {
    const auth = firebase.auth();
    if (useLocalEmulators()) {
        auth.useEmulator('http://127.0.0.1:9099', { disableWarnings: true });
    }
    return auth;
}

// Anonymous sign-in for the security rules. Returns signedIn(): call it right before
// a write; it resolves to the user (or null) and retries a failed sign-in (busy
// server, flaky network) instead of leaving the page signed out for good.
function anonymousSignIn(auth, persistence) {
    const ready = auth.setPersistence(persistence).catch(() => {});
    let pending = null;
    return function signedIn() {
        pending = pending || (async () => {
            await ready;
            for (let attempt = 0; ; attempt++) {
                if (auth.currentUser) return auth.currentUser;
                try {
                    const cred = await auth.signInAnonymously();
                    return (cred && cred.user) || auth.currentUser || null;
                } catch (err) {
                    if (attempt >= 3) {
                        console.error('Anonymous auth failed (enable the Anonymous provider in the Firebase console):', err && (err.code || err.message || err));
                        return null;
                    }
                    await new Promise((r) => setTimeout(r, 400 * 2 ** attempt));
                }
            }
        })().finally(() => { pending = null; });
        return pending;
    };
}

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
        return (this.baseCost ?? 1) + naughtyLevel; // ?? not ||: "0 + Naughty" is a real setting
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
