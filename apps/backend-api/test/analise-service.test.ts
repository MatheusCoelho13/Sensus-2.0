import type { DetectedObject } from '@sensus/shared-contracts'
import { describe, expect, it, vi } from 'vitest'
import { createAnaliseService } from '../src/analise-service'
import { AnaliseNotFoundError, InvalidImageError } from '../src/types'

const obj = (name: string, o: Partial<DetectedObject> = {}): DetectedObject => ({
  name, confidence: 0.9, x: 0.1, y: 0.2, width: 0.3, height: 0.4, distance: 'longe', isClose: false, ...o,
})

function setup(objects: DetectedObject[], relevantClasses: string[] = [], repo: any = null) {
  const ia = { infer: vi.fn().mockResolvedValue({ timestamp: 123, objects }), healthy: vi.fn() }
  return { ia, svc: createAnaliseService({ ia, repo, relevantClasses }) }
}

describe('AnaliseService', () => {
  it('rejects empty and oversized images', async () => {
    const { svc } = setup([])
    await expect(svc.processImage(Buffer.alloc(0))).rejects.toBeInstanceOf(InvalidImageError)
    await expect(svc.processImage(Buffer.alloc(10 * 1024 * 1024 + 1))).rejects.toBeInstanceOf(InvalidImageError)
  })

  it('filters by relevant classes, sorts close first then by name, maps to legacy DTO', async () => {
    const { svc } = setup(
      [obj('person'), obj('zebra'), obj('car', { distance: 'perto', isClose: true }), obj('chair', { distance: 'perto' })],
      ['person', 'car', 'chair'],
    )
    const r = await svc.processImage(Buffer.from('x'))
    expect(r.objetos.map((o) => o.nome)).toEqual(['car', 'chair', 'person'])
    expect(r.objetos[1]).toMatchObject({ isClose: true, distancia: 'perto', bboxX: 0.1, bboxHeight: 0.4 }) // perto ⇒ isClose
    expect(r.timestamp).toBe(123)
    expect(r.id).toBeNull()
  })

  it('empty filter list keeps everything', async () => {
    const { svc } = setup([obj('zebra')])
    expect((await svc.processImage(Buffer.from('x'))).objetos).toHaveLength(1)
  })

  it('persists and returns id; persistence failure does not fail the request', async () => {
    const save = vi.fn().mockResolvedValue(7)
    expect((await setup([obj('person')], [], { save }).svc.processImage(Buffer.from('x'))).id).toBe(7)

    const warn = vi.fn()
    const ia = { infer: vi.fn().mockResolvedValue({ timestamp: 1, objects: [] }), healthy: vi.fn() }
    const svc = createAnaliseService({ ia, repo: { save: vi.fn().mockRejectedValue(new Error('db')) } as any, relevantClasses: [], warn })
    expect((await svc.processImage(Buffer.from('x'))).id).toBeNull()
    expect(warn).toHaveBeenCalled()
  })

  it('findById throws not-found', async () => {
    const { svc } = setup([], [], { findById: vi.fn().mockResolvedValue(null) })
    await expect(svc.findById(1)).rejects.toBeInstanceOf(AnaliseNotFoundError)
  })
})
