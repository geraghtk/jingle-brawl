# Deploying Yuel Duel

One Node process serves the app and runs every room (WebSockets, SQLite). It needs one machine, one
disk for `rooms.db`, and HTTPS in front of it (phones' browsers want `wss://` on a public site).

## What to set

| Setting | Why |
|---|---|
| a volume at `/data` (`JB_DB=/data/rooms.db`, the image's default) | rooms survive restarts and deploys |
| `JB_TRUST_PROXY=1` (or `--trust-proxy`) | behind a reverse proxy: the per-address limits use the client's address from `X-Forwarded-For` (the proxy's last hop). Leave it off when nothing is in front, or anyone could pick their own address. |
| `JB_TOKEN_SECRET` (optional) | signs the host's and players' session tokens. Without it the server makes one and keeps it in `rooms.db`; set it to rotate (everyone rejoins). |
| `JB_METRICS_TOKEN` (optional) | `/metrics` answers this machine only; with this set, also `Authorization: Bearer <token>` (behind a proxy, only the token). |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` (optional) | a default AI Draw Off critic for your own parties. Hosts can bring their own key per game instead (memory only). |
| `PORT` | default 8080 in the image, 8787 otherwise |

Never run production with `--debug-api` (it also switches off the abuse limits).

## Docker

```bash
docker build -t yuel-duel .
docker run -d --name yuel-duel -p 8080:8080 -v yuel-data:/data -e JB_TRUST_PROXY=1 yuel-duel
```

Put a TLS proxy in front (Caddy is the least work: `yourdomain.example { reverse_proxy 127.0.0.1:8080 }`).
WebSockets pass through on `/ws`; nothing else to configure.

## Fly.io (example)

```bash
fly launch --no-deploy          # pick a name and a region near your parties; keep the Dockerfile
fly volumes create data --size 1
```

Then in `fly.toml`:

```toml
[env]
  JB_TRUST_PROXY = "1"

[mounts]
  source = "data"
  destination = "/data"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = "off"     # rooms live in memory between moves: don't stop mid-party
  min_machines_running = 1

[[http_service.checks]]
  path = "/healthz"
  interval = "30s"
  timeout = "3s"
```

`fly deploy`. Run exactly one machine: rooms are in that process's memory (a second machine would
split a party across two servers).

## What it looks after by itself

- **Abuse limits** per client address: making rooms, joining, looking up room codes, opening sockets
  (generous: a household shares one address), at most 150 open sockets per address, 2000 live rooms,
  50 players a game, a per-socket message rate and a 1 MB frame cap. Over a limit: `429` with
  `Retry-After`. Full: `503`.
- **Room lifetime**: rooms leave memory after 2 h idle (forgetting any AI key) and are deleted from
  `rooms.db` 2 h after the game ends or 24 h after anyone last touched them (checked every 10 min).
- **Headers** on everything: a Content-Security-Policy (our scripts only; Google Fonts),
  `Referrer-Policy: no-referrer` (room codes ride in URLs), `X-Content-Type-Options: nosniff`, no framing.

## Watching it

- Logs: one JSON object per line on stdout (`room.created`, `rooms.purged`, `client.error`,
  `intent.crash`, `rooms.full`, `seat.reissued`, `crash`). No names, drawings, keys or addresses.
- `/metrics` (Prometheus text): `jb_rooms_live`, `jb_sockets_open`, `jb_rooms_created_total`,
  `jb_joins_total`, `jb_intents_total{ok}`, `jb_intent_crashes_total`, `jb_rate_limited_total{kind}`,
  `jb_client_errors_total{page}`, `jb_rooms_refused_total`, `jb_sockets_refused_total{why}`.
- `/healthz` for the load balancer.
- Browsers report their uncaught errors to `/api/client-error` (page, message, stack; no URL).

## Backups

`rooms.db` holds only games in progress (nothing older than a day), so a lost disk loses tonight's
parties, not history. If that matters, Litestream can stream it to object storage (planned, P1).
