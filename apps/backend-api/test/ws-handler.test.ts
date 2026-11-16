import { describe, expect, it, vi } from 'vitest'
import { RateLimiter } from '../src/rate-limiter'
import { IAServiceError, type AnaliseResponseDTO } from '../src/types'
import { createWsHandler } from '../src/ws-handler'

const result: AnaliseResponseDTO = { id: null, timestamp: 1, objetos: [] }

function setup(processImage = vi.fn().mockResolvedValue(result), fps = 5) {
  const send = vi.fn()
  const limiter = new RateLimiter(fps)
  const h = createWsHandler({ analise: { processImage }, limiter, send, log: { debug: vi.fn(), warn: vi.fn() } })
  const text = (o: unknown) => h.onMessage('s1', Buffer.from(typeof o === 'string' ? o : JSON.stringify(o)), false)
  return { h, send, processImage, text }
}

describe('ws handler', () => {
  it('ignores heartbeat (even with extra fields): no inference, no reply', async () => {
    const { text, send, processImage } = setup()
    await text({ type: 'heartbeat', timestamp: 1, extra: 'x' })
    expect(processImage).not.toHaveBeenCalled()
    expect(send).not.toHaveBeenCalled()
  })

  it('is case-sensitive on heartbeat and ignores blank/missing dados', async () => {
    const { text, processImage } = setup()
    await text({ type: 'Heartbeat', timestamp: 1 })
    await text({ tipo: 'imagem', dados: '' })
    expect(processImage).not.toHaveBeenCalled()
  })

  it('decodes base64 dados, runs analysis, replies with result', async () => {
    const { text, send, processImage } = setup()
    await text({ tipo: 'imagem', dados: Buffer.from('jpeg-bytes').toString('base64') })
    expect(processImage).toHaveBeenCalledWith(Buffer.from('jpeg-bytes'))
    expect(send).toHaveBeenCalledWith('s1', result)
  })

  it('accepts binary frames', async () => {
    const { h, send } = setup()
    await h.onMessage('s1', Buffer.from([1, 2, 3]), true)
    expect(send).toHaveBeenCalledWith('s1', result)
  })

  it('replies INVALID_PAYLOAD on malformed JSON and keeps going', async () => {
    const { text, send } = setup()
    await text('{nope')
    expect(send.mock.calls[0]![1]).toMatchObject({ type: 'error', code: 'INVALID_PAYLOAD' })
  })

  it('maps IAServiceError to legacy+SDD error payload; other errors are swallowed', async () => {
    const ia = setup(vi.fn().mockRejectedValue(new IAServiceError('down')))
    await ia.text({ dados: 'AQID' })
    expect(ia.send.mock.calls[0]![1]).toMatchObject({ erro: 'IA indisponível', code: 'INFERENCE_ERROR' })

    const other = setup(vi.fn().mockRejectedValue(new Error('boom')))
    await other.text({ dados: 'AQID' })
    expect(other.send).not.toHaveBeenCalled()
  })

  it('drops frames over the rate limit and while one is in flight', async () => {
    let release!: (v: AnaliseResponseDTO) => void
    const slow = vi.fn(() => new Promise<AnaliseResponseDTO>((r) => (release = r)))
    const { text, processImage, send } = setup(slow)
    const first = text({ dados: 'AQID' })
    await text({ dados: 'AQID' }) // in flight → dropped
    expect(processImage).toHaveBeenCalledTimes(1)
    release(result)
    await first
    expect(send).toHaveBeenCalledTimes(1)

    const limited = setup(undefined, 1)
    await limited.text({ dados: 'AQID' })
    await limited.text({ dados: 'AQID' })
    expect(limited.processImage).toHaveBeenCalledTimes(1)
  })
})
