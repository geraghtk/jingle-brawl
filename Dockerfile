# The room server and the app in one small image: `docker build -t yuel-duel .` then
# `docker run -p 8080:8080 -v yuel-data:/data yuel-duel`. Rooms live in /data/rooms.db (keep it on a
# volume). Behind a reverse proxy (Fly, Caddy, nginx) set JB_TRUST_PROXY=1. See DEPLOY.md.

FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts
COPY tsconfig.json vite.config.mts ./
COPY web ./web
COPY server ./server
RUN npm run build

FROM node:22-slim
ENV NODE_ENV=production PORT=8080 JB_DB=/data/rooms.db
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts && npm cache clean --force
COPY --from=build /app/dist ./dist
COPY --from=build /app/dist-server ./dist-server
RUN mkdir -p /data && chown node:node /data
USER node
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD node -e "fetch('http://127.0.0.1:'+process.env.PORT+'/healthz').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "--disable-warning=ExperimentalWarning", "dist-server/index.mjs"]
