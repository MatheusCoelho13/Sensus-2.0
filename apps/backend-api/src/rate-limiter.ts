/** Fixed 1s window per session. Single-threaded Node → no locking needed. */
export class RateLimiter {
  private windows = new Map<string, { start: number; count: number }>()

  constructor(
    private maxFps: number,
    private now: () => number = Date.now,
  ) {}

  allow(sessionId: string): boolean {
    const t = this.now()
    const w = this.windows.get(sessionId)
    if (!w || t - w.start >= 1000) {
      this.windows.set(sessionId, { start: t, count: 1 })
      return true
    }
    if (w.count < this.maxFps) {
      w.count++
      return true
    }
    return false
  }

  remove(sessionId: string): void {
    this.windows.delete(sessionId)
  }
}
