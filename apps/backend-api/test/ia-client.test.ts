import { describe, expect, it, vi } from 'vitest'
import { createIAClient } from '../src/ia-client'
import { IAServiceError } from '../src/types'

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })

describe('IAClient', () => {
  it('posts base64 image to /infer and returns objects', async () => {
    const f = vi.fn().mockResolvedValue(json({ objects: [{ name: 'person' }] }))
    const ia = createIAClient({ baseUrl: 'http://ia', timeoutMs: 100, fetch: f as never })
    const r = await ia.infer(Buffer.from('abc'))
    expect(r.objects).toHaveLength(1)
    const [url, init] = f.mock.calls[0]!
    expect(url).toBe('http://ia/infer')
    expect(JSON.parse(init.body)).toMatchObject({ image: Buffer.from('abc').toString('base64') })
  })

  it('HTTP error → IAServiceError, no retry', async () => {
    const f = vi.fn().mockResolvedValue(json({}, 500))
    const ia = createIAClient({ baseUrl: 'http://ia', timeoutMs: 100, fetch: f as never })
    await expect(ia.infer(Buffer.from('a'))).rejects.toBeInstanceOf(IAServiceError)
    expect(f).toHaveBeenCalledTimes(1)
  })

  it('transport error retries once then fails as IAServiceError', async () => {
    const f = vi.fn().mockRejectedValue(new TypeError('fetch failed'))
    const ia = createIAClient({ baseUrl: 'http://ia', timeoutMs: 100, fetch: f as never })
    await expect(ia.infer(Buffer.from('a'))).rejects.toBeInstanceOf(IAServiceError)
    expect(f).toHaveBeenCalledTimes(2)
  })

  it('healthy() is false when unreachable', async () => {
    const ia = createIAClient({ baseUrl: 'http://ia', timeoutMs: 100, fetch: vi.fn().mockRejectedValue(new Error('x')) as never })
    expect(await ia.healthy()).toBe(false)
  })
})
