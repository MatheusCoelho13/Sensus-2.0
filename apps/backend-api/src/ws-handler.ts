import { IAServiceError, type AnaliseResponseDTO } from './types'
import type { RateLimiter } from './rate-limiter'

type Deps = {
  analise: { processImage(image: Buffer): Promise<AnaliseResponseDTO> }
  limiter: RateLimiter
  send: (sessionId: string, payload: unknown) => void
  log: { debug(msg: string): void; warn(msg: string): void }
}

// Error carries both the SDD ErrorResponse fields and legacy `erro` (frontend checks `'erro' in data`).
const errorPayload = (code: string, message: string) => ({ type: 'error', code, message, timestamp: Date.now(), erro: message })

/**
 * Transport-agnostic realtime pipeline (SDD 02): validate → heartbeat → rate-limit → IA → reply.
 * Never throws; one in-flight frame per session, newer frames are dropped (no queue buildup).
 */
export function createWsHandler({ analise, limiter, send, log }: Deps) {
  const busy = new Set<string>()

  async function process(sessionId: string, image: Buffer) {
    if (image.length === 0) return
    if (busy.has(sessionId) || !limiter.allow(sessionId)) return
    busy.add(sessionId)
    try {
      send(sessionId, await analise.processImage(image))
    } catch (e) {
      if (e instanceof IAServiceError) send(sessionId, errorPayload('INFERENCE_ERROR', 'IA indisponível'))
      else log.warn(`[WEBSOCKET] Erro ao processar frame, mantendo sessão: sessionId=${sessionId} erro=${(e as Error).message}`)
    } finally {
      busy.delete(sessionId)
    }
  }

  return {
    async onMessage(sessionId: string, data: Buffer, isBinary: boolean): Promise<void> {
      if (isBinary) return process(sessionId, data)

      let msg: unknown
      try {
        msg = JSON.parse(data.toString('utf8'))
      } catch {
        return send(sessionId, errorPayload('INVALID_PAYLOAD', 'JSON inválido'))
      }
      if (typeof msg !== 'object' || msg === null) return send(sessionId, errorPayload('INVALID_PAYLOAD', 'Payload inválido'))

      const { type, dados } = msg as { type?: unknown; dados?: unknown }
      if (type === 'heartbeat') return log.debug(`[WEBSOCKET] Heartbeat: sessionId=${sessionId}`)
      if (typeof dados !== 'string' || dados.trim() === '') return
      return process(sessionId, Buffer.from(dados, 'base64'))
    },

    onClose(sessionId: string) {
      limiter.remove(sessionId)
      busy.delete(sessionId)
    },
  }
}
