import pg from 'pg'
import { buildApp } from './app'
import { createAnaliseService } from './analise-service'
import { loadConfig } from './config'
import { createIAClient } from './ia-client'
import { migrate } from './migrate'
import { RateLimiter } from './rate-limiter'
import { createPgRepository } from './repository'
import { createElevenLabsTts } from './tts'

const cfg = loadConfig()

// No DB configured (or persistence off) → run stateless; /api/analises returns empty.
const pool = cfg.persistence && cfg.databaseUrl ? new pg.Pool({ connectionString: cfg.databaseUrl, max: 10 }) : null
if (pool) await migrate(pool)

const ia = createIAClient({ baseUrl: cfg.iaBaseUrl, timeoutMs: cfg.iaTimeoutMs })
const app = await buildApp({
  ia,
  analise: createAnaliseService({
    ia,
    repo: pool ? createPgRepository(pool) : null,
    relevantClasses: cfg.relevantClasses,
    warn: (m) => app.log.warn(m),
  }),
  limiter: new RateLimiter(cfg.rateLimitFps),
  tts: createElevenLabsTts(cfg.elevenlabs),
  wsMaxPayload: cfg.wsMaxPayload,
  logLevel: cfg.logLevel,
})

for (const sig of ['SIGINT', 'SIGTERM'] as const) {
  process.on(sig, async () => {
    await app.close()
    await pool?.end()
    process.exit(0)
  })
}

await app.listen({ host: '0.0.0.0', port: cfg.port })
