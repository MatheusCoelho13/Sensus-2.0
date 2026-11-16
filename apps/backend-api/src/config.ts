export type Config = {
  port: number
  databaseUrl: string | null
  iaBaseUrl: string
  iaTimeoutMs: number
  rateLimitFps: number
  persistence: boolean
  relevantClasses: string[]
  wsMaxPayload: number
  logLevel: string
  elevenlabs: { apiKey: string; voiceId: string; model: string }
}

const DEFAULT_CLASSES = ['person', 'chair', 'dining table', 'door', 'stairs', 'car', 'cell phone', 'backpack']

const int = (v: string | undefined, d: number) => (v && Number.isFinite(Number(v)) ? Number(v) : d)

/** DATABASE_URL wins; falls back to legacy SPRING_DATASOURCE_URL + DB_USER/DB_PASS so existing deploys keep working. */
export function resolveDatabaseUrl(env: NodeJS.ProcessEnv): string | null {
  if (env.DATABASE_URL) return env.DATABASE_URL
  const jdbc = env.SPRING_DATASOURCE_URL
  if (!jdbc) return null
  const url = new URL(jdbc.replace(/^jdbc:/, ''))
  if (env.DB_USER) url.username = env.DB_USER
  if (env.DB_PASS) url.password = env.DB_PASS
  return url.toString()
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  return {
    port: int(env.PORT, 8080),
    databaseUrl: resolveDatabaseUrl(env),
    iaBaseUrl: (env.IA_SERVICE_URL ?? 'http://ia-service:8000').replace(/\/$/, ''),
    iaTimeoutMs: int(env.IA_TIMEOUT_MS, 5000),
    rateLimitFps: int(env.RATE_LIMIT_FPS, 5),
    persistence: env.PERSISTENCE_ENABLED !== 'false',
    relevantClasses: env.RELEVANT_CLASSES ? env.RELEVANT_CLASSES.split(',').map((s) => s.trim()) : DEFAULT_CLASSES,
    wsMaxPayload: int(env.WEBSOCKET_MAX_PAYLOAD_SIZE, 512 * 1024),
    logLevel: env.LOG_LEVEL?.toLowerCase() ?? 'info',
    elevenlabs: {
      apiKey: env.ELEVENLABS_API_KEY ?? '',
      voiceId: env.ELEVENLABS_VOICE_ID ?? 'pNInz6obpgDQGcFmaJgB',
      model: env.ELEVENLABS_MODEL ?? 'eleven_multilingual_v2',
    },
  }
}
