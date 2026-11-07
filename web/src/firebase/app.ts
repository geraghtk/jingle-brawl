// Firebase app, database and anonymous auth. On localhost everything goes to the local
// emulators (database :9000, auth :9099) so testing never touches production; ?prod=1 opts out.
import { initializeApp } from 'firebase/app';
import {
  browserLocalPersistence, browserSessionPersistence, connectAuthEmulator, getAuth,
  setPersistence, signInAnonymously,
} from 'firebase/auth';
import { connectDatabaseEmulator, getDatabase } from 'firebase/database';

const firebaseConfig = {
  apiKey: 'AIzaSyDOPkYyZ6qFNoBB7iVbZRtyjk3W_uf7POA',
  authDomain: 'jingle-brawl-e8308.firebaseapp.com',
  databaseURL: 'https://jingle-brawl-e8308-default-rtdb.firebaseio.com',
  projectId: 'jingle-brawl-e8308',
  storageBucket: 'jingle-brawl-e8308.firebasestorage.app',
  messagingSenderId: '778188643750',
  appId: '1:778188643750:web:be573c1c5f88cd61a48367',
};

import { testId } from '../net/storage';

const params = new URLSearchParams(location.search);
export const usingEmulators = ['localhost', '127.0.0.1'].includes(location.hostname) && !params.has('prod');

export const app = initializeApp(firebaseConfig);
export const db = getDatabase(app);
export const auth = getAuth(app);
if (usingEmulators) {
  connectDatabaseEmulator(db, '127.0.0.1', 9000);
  connectAuthEmulator(auth, 'http://127.0.0.1:9099', { disableWarnings: true });
}

const persistence = setPersistence(auth, testId ? browserSessionPersistence : browserLocalPersistence).catch(() => undefined);
let pending: Promise<string | null> | null = null;

/**
 * Signed in anonymously before any write (the rules require auth; players own their
 * node via ownerUid). Test tabs use per-tab persistence so each tab is its own user.
 * Call it at the moment you need the uid: a failed sign-in (busy server, flaky network)
 * is retried here rather than leaving the page signed out for good.
 */
export function signedIn(): Promise<string | null> {
  pending ??= (async () => {
    await persistence;
    await auth.authStateReady(); // a returning host keeps their uid (= hostUid)
    for (let attempt = 0; ; attempt++) {
      if (auth.currentUser) return auth.currentUser.uid;
      try {
        return (await signInAnonymously(auth)).user.uid;
      } catch (err) {
        if (attempt >= 3) {
          console.error('Anonymous auth failed (enable the Anonymous provider in the Firebase console):', (err as { code?: string })?.code || err);
          return null;
        }
        await new Promise((r) => setTimeout(r, 400 * 2 ** attempt));
      }
    }
  })().finally(() => { pending = null; });
  return pending;
}
signedIn(); // start early so it is usually done before anyone taps a button

export { storageKey, testId } from '../net/storage';
