import { randomUUID } from 'node:crypto'
import Fastify, { type FastifyInstance } from 'fastify'
import websocket from '@fastify/websocket'
import { z } from 'zod'
import type { AnaliseService } from './analise-service'
import type { IAClient } from './ia-client'
import type { RateLimiter } from './rate-limiter'
import type { Tts } from './tts'
import { createWsHandler } from './ws-handler'
import { AnaliseNotFoundError, IAServiceError, InvalidImageError, erro, ok } from './types'

export type AppDeps = {
  analise: AnaliseService
  ia: IAClient
  limiter: RateLimiter
  tts: Tts
  wsMaxPayload: number
  logLevel?: string
}

const BASE64 = /^[A-Za-z0-9+/\r\n]*={0,2}$/

const AnalisarBody = z.object({
  imagemBase64: z.string().trim().min(1, 'imagemBase64 é obrigatório'),
  formato: z.string().optional(),
})
const TtsBody = z.object({ text: z.string().trim().min(1).max(1000), isClose: z.boolean().default(false) })
const PageQuery = z.object({
  page: z.coerce.number().int().min(0).default(0),
  size: z.coerce.number().int().min(1).max(100).default(20),
})
const IdParam = z.object({ id: z.coerce.number().int().positive() })

export async function buildApp(deps: AppDeps): Promise<FastifyInstance> {
  const app = Fastify({ logger: { level: deps.logLevel ?? 'info' }, bodyLimit: 15 * 1024 * 1024 })
  await app.register(websocket, { options: { maxPayload: deps.wsMaxPayload } })

  // ── Realtime (SDD 02/04): /ws/realtime ────────────────────────────────────
  const sockets = new Map<string, import('ws').WebSocket>()
  const handler = createWsHandler({
    analise: deps.analise,
    limiter: deps.limiter,
    log: app.log.child({ ctx: 'ws' }) as never,
    send: (id, payload) => {
      const s = sockets.get(id)
      if (s?.readyState === s?.OPEN) s?.send(JSON.stringify(payload))
    },
  })

  app.get('/ws/realtime', { websocket: true }, (socket) => {
    const id = randomUUID()
    sockets.set(id, socket)
    socket.on('message', (data: Buffer, isBinary: boolean) => {
      void handler.onMessage(id, data, isBinary)
    })
    socket.on('error', (e) => app.log.error({ err: e }, '[WEBSOCKET] transport error'))
    socket.on('close', () => {
      sockets.delete(id)
      handler.onClose(id)
    })
  })

  // ── Health ────────────────────────────────────────────────────────────────
  // Liveness never depends on IA (docker/Render healthcheck); IA state is reported alongside.
  const health = async () => ({
    status: 'UP',
    ia: (await deps.ia.healthy()) ? 'UP' : 'DOWN',
    websocket: { activeSessions: sockets.size },
  })
  app.get('/api/health', health)
  app.get('/actuator/health', health)

  app.get('/api/status', async () =>
    ok({
      backend: 'UP',
      iaService: (await deps.ia.healthy()) ? 'UP' : 'DOWN',
      timestamp: Math.floor(Date.now() / 1000),
    }),
  )

  // ── Análises ──────────────────────────────────────────────────────────────
  app.post('/api/analisar', async (req, reply) => {
    const body = AnalisarBody.parse(req.body)
    if (!BASE64.test(body.imagemBase64)) return reply.code(400).send(erro('Imagem base64 inválida'))
    return ok(await deps.analise.processImage(Buffer.from(body.imagemBase64, 'base64')))
  })

  app.get('/api/analises', async (req) => {
    const { page, size } = PageQuery.parse(req.query)
    return deps.analise.list(page, size)
  })

  app.get('/api/analises/:id', async (req) => ok(await deps.analise.findById(IdParam.parse(req.params).id)))

  // ── TTS ───────────────────────────────────────────────────────────────────
  app.post('/api/tts', async (req, reply) => {
    const body = TtsBody.parse(req.body)
    try {
      const audio = await deps.tts(body.text, body.isClose)
      return reply.type('audio/mpeg').send(audio)
    } catch (e) {
      req.log.error({ err: e }, 'TTS failed')
      return reply.code(500).send()
    }
  })

  // ── Errors ────────────────────────────────────────────────────────────────
  app.setErrorHandler((error, req, reply) => {
    const err = error as Error & { statusCode?: number }
    if (err instanceof z.ZodError) {
      return reply.code(400).send(erro(err.issues.map((i) => i.message).join(', ')))
    }
    if (err instanceof InvalidImageError) return reply.code(400).send(erro(err.message))
    if (err instanceof AnaliseNotFoundError) return reply.code(404).send(erro('Análise não encontrada'))
    if (err instanceof IAServiceError) return reply.code(503).send(erro('Serviço de IA indisponível'))
    const status = err.statusCode
    if (status && status >= 400 && status < 500) return reply.code(status).send(erro(err.message))
    req.log.error({ err }, 'unhandled error')
    return reply.code(500).send(erro('Erro interno do servidor'))
  })

  return app
}
