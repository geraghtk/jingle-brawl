# Firebase Setup Guide for Jingle Brawl

## Step 1: Create a Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Add project"
3. Name it something like "jingle-brawl"
4. Disable Google Analytics (not needed)
5. Click "Create project"

## Step 2: Create a Realtime Database

1. In your Firebase project, go to **Build → Realtime Database**
2. Click "Create Database"
3. Choose a location (any region works)
4. Start in **Test mode** (we'll secure it later)
5. Click "Enable"

## Step 3: Get Your Configuration

1. Go to **Project Settings** (gear icon)
2. Scroll down to "Your apps"
3. Click the web icon (`</>`)
4. Register app with nickname "jingle-brawl-web"
5. Copy the `firebaseConfig` object

## Step 4: Add Your Config

Create or edit `firebase-config.js` in your project folder:

```javascript
// firebase-config.js
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  databaseURL: "https://YOUR_PROJECT-default-rtdb.firebaseio.com",
  projectId: "YOUR_PROJECT",
  storageBucket: "YOUR_PROJECT.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID"
};
```

Replace the values with your actual Firebase config.

## Step 5: Security Rules (Optional but Recommended)

In Firebase Console → Realtime Database → Rules, use:

```json
{
  "rules": {
    "games": {
      "$roomCode": {
        ".read": true,
        ".write": true
      }
    }
  }
}
```

For production, you'd want stricter rules, but this works for party games.

## Step 6: Test Your Setup

1. Open `host.html` in a browser
2. Create a new game
3. Open `player.html` on a phone
4. Enter the room code
5. You should see the player appear on the host screen!

## Troubleshooting

- **"Firebase is not defined"**: Make sure firebase-config.js has the correct values
- **"Permission denied"**: Check your database rules are in test mode
- **Players not syncing**: Ensure all devices are on the internet

## Hosting (Optional)

For the best experience, host these files on:
- **GitHub Pages** (free)
- **Firebase Hosting** (free tier)
- **Netlify** (free)

This way everyone can access via a URL instead of local files.

