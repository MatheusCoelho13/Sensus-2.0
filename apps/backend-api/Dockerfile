# Build context is the monorepo root so packages/shared-contracts resolves:
#   docker build -f apps/backend-api/Dockerfile .

# Stage 1: build (shared-contracts is bundled into dist/)
FROM node:22-alpine AS builder
WORKDIR /app
COPY packages/shared-contracts ./packages/shared-contracts
COPY apps/backend-api ./apps/backend-api
# Minimal workspace root: only what the backend needs
RUN echo '{"private":true,"workspaces":["apps/backend-api","packages/shared-contracts"]}' > package.json \
 && npm install --no-audit --no-fund \
 && npm run build -w apps/backend-api \
 && npm prune --omit=dev --no-audit --no-fund

# Stage 2: runtime
FROM node:22-alpine
ENV NODE_ENV=production
WORKDIR /app/apps/backend-api
COPY --from=builder /app/node_modules /app/node_modules
COPY --from=builder /app/apps/backend-api/dist ./dist
COPY --from=builder /app/apps/backend-api/migrations ./migrations
USER node
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD wget -qO- http://localhost:${PORT:-8080}/api/health || exit 1
CMD ["node", "dist/server.js"]
