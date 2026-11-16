import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import type { FastifyInstance } from 'fastify'
import { buildApp } from '../src/app'
import { RateLimiter } from '../src/rate-limiter'
import { AnaliseNotFoundError, IAServiceError } from '../src/types'

const dto = { id: 1, timestamp: 5, objetos: [] }
const analise = {
  processImage: vi.fn().mockResolvedValue(dto),
  list: vi.fn().mockResolvedValue({ content: [dto], totalElements: 1, totalPages: 1, size: 20, number: 0 }),
  findById: vi.fn().mockImplementation(async (id: number) => {
    if (id === 1) return dto
    throw new AnaliseNotFoundError('nf')
  }),
}
let app: FastifyInstance

beforeAll(async () => {
  app = await buildApp({
    analise,
    ia: { infer: vi.fn(), healthy: vi.fn().mockResolvedValue(true) },
    limiter: new RateLimiter(5),
    tts: vi.fn().mockResolvedValue(Buffer.from('mp3')),
    wsMaxPayload: 1024,
    logLevel: 'silent',
  })
})
afterAll(() => app.close())

describe('REST', () => {
  it('GET /api/status wraps backend + ia status', async () => {
    const r = await app.inject('/api/status')
    expect(r.json()).toMatchObject({ sucesso: true, dados: { backend: 'UP', iaService: 'UP' } })
  })

  it('GET /api/health and /actuator/health', async () => {
    for (const url of ['/api/health', '/actuator/health']) {
      expect((await app.inject(url)).json()).toMatchObject({ status: 'UP', ia: 'UP' })
    }
  })

  it('POST /api/analisar ok / blank / bad base64', async () => {
    const post = (payload: unknown) => app.inject({ method: 'POST', url: '/api/analisar', payload: payload as never })
    const ok = await post({ imagemBase64: 'AQID' })
    expect(ok.json()).toMatchObject({ sucesso: true, dados: dto })
    expect((await post({ imagemBase64: '  ' })).statusCode).toBe(400)
    expect((await post({ imagemBase64: '@@@' })).json()).toMatchObject({ sucesso: false, mensagem: 'Imagem base64 inválida' })
  })

  it('IA failure → 503', async () => {
    analise.processImage.mockRejectedValueOnce(new IAServiceError('x'))
    const r = await app.inject({ method: 'POST', url: '/api/analisar', payload: { imagemBase64: 'AQID' } })
    expect(r.statusCode).toBe(503)
  })

  it('GET /api/analises paginates (Spring Page shape); bad size → 400', async () => {
    const r = await app.inject('/api/analises?page=0&size=20')
    expect(r.json()).toMatchObject({ content: [dto], totalElements: 1, number: 0 })
    expect((await app.inject('/api/analises?size=0')).statusCode).toBe(400)
  })

  it('GET /api/analises/:id found / 404 / 400', async () => {
    expect((await app.inject('/api/analises/1')).json()).toMatchObject({ sucesso: true, dados: dto })
    expect((await app.inject('/api/analises/2')).statusCode).toBe(404)
    expect((await app.inject('/api/analises/abc')).statusCode).toBe(400)
  })

  it('POST /api/tts returns audio/mpeg; invalid body 400', async () => {
    const r = await app.inject({ method: 'POST', url: '/api/tts', payload: { text: 'olá', isClose: true } })
    expect(r.headers['content-type']).toContain('audio/mpeg')
    expect(r.body).toBe('mp3')
    expect((await app.inject({ method: 'POST', url: '/api/tts', payload: {} })).statusCode).toBe(400)
  })
})
