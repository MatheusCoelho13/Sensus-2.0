import { describe, expect, it } from 'vitest'
import { RateLimiter } from '../src/rate-limiter'

describe('RateLimiter', () => {
  it('allows up to maxFps then rejects, per session', () => {
    const rl = new RateLimiter(3, () => 0)
    expect([1, 2, 3].map(() => rl.allow('a'))).toEqual([true, true, true])
    expect(rl.allow('a')).toBe(false)
    expect(rl.allow('b')).toBe(true)
  })

  it('resets window after 1s and after remove()', () => {
    let t = 0
    const rl = new RateLimiter(1, () => t)
    rl.allow('a')
    expect(rl.allow('a')).toBe(false)
    t = 1000
    expect(rl.allow('a')).toBe(true)
    expect(rl.allow('a')).toBe(false)
    rl.remove('a')
    expect(rl.allow('a')).toBe(true)
  })
})
