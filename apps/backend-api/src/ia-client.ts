import { randomUUID } from 'node:crypto'
import type { DetectedObject } from '@sensus/shared-contracts'
import { IAServiceError } from './types'

export type IAResult = { timestamp: number; objects: DetectedObject[] }

export type IAClient = {
  infer(image: Buffer): Promise<IAResult>
  healthy(): Promise<boolean>
}

const RETRY_DELAY_MS = 500

export function createIAClient(opts: { baseUrl: string; timeoutMs: number; fetch?: typeof fetch }): IAClient {
  const doFetch = opts.fetch ?? fetch

  async function inferOnce(body: string): Promise<IAResult> {
    const res = await doFetch(`${opts.baseUrl}/infer`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body,
      signal: AbortSignal.timeout(opts.timeoutMs),
    })
    // HTTP errors are definitive (no retry); only transport errors/timeouts retry.
    if (!res.ok) throw new IAServiceError(`Serviço de IA retornou erro: ${res.status}`)
    const json = (await res.json()) as { objects?: DetectedObject[] }
    return { timestamp: 0, objects: Array.isArray(json?.objects) ? json.objects : [] }
  }

  return {
    async infer(image) {
      const timestamp = Date.now()
      const body = JSON.stringify({ frameId: randomUUID(), timestamp, image: image.toString('base64') })
      for (let attempt = 0; ; attempt++) {
        try {
          return { ...(await inferOnce(body)), timestamp }
        } catch (e) {
          if (e instanceof IAServiceError) throw e
          if (attempt >= 1) throw new IAServiceError(`Falha ao comunicar com serviço de IA: ${(e as Error).message}`)
          await new Promise((r) => setTimeout(r, RETRY_DELAY_MS))
        }
      }
    },

    async healthy() {
      try {
        const res = await doFetch(`${opts.baseUrl}/health`, { signal: AbortSignal.timeout(3000) })
        return res.ok
      } catch {
        return false
      }
    },
  }
}
