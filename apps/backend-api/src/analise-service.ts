import type { IAClient } from './ia-client'
import type { AnaliseRepository } from './repository'
import {
  AnaliseNotFoundError,
  InvalidImageError,
  type AnaliseResponseDTO,
  type ObjetoDetectadoDTO,
  type Page,
} from './types'

const MAX_IMAGE_BYTES = 10 * 1024 * 1024

export type AnaliseService = ReturnType<typeof createAnaliseService>

export function createAnaliseService(deps: {
  ia: IAClient
  repo: AnaliseRepository | null
  relevantClasses: string[]
  warn?: (msg: string) => void
}) {
  const { ia, repo, relevantClasses } = deps

  return {
    async processImage(image: Buffer): Promise<AnaliseResponseDTO> {
      if (image.length === 0) throw new InvalidImageError('Imagem inválida ou vazia')
      if (image.length > MAX_IMAGE_BYTES) throw new InvalidImageError('Imagem muito grande (máximo 10MB)')

      const { timestamp, objects } = await ia.infer(image)

      const all: ObjetoDetectadoDTO[] = objects.map((o) => ({
        nome: o.name,
        confidence: o.confidence,
        distancia: o.distance,
        isClose: o.isClose || o.distance?.toLowerCase() === 'perto',
        bboxX: o.x,
        bboxY: o.y,
        bboxWidth: o.width,
        bboxHeight: o.height,
      }))

      // close objects first, then by name
      const objetos = (relevantClasses.length ? all.filter((o) => relevantClasses.includes(o.nome)) : all).sort(
        (a, b) => Number(b.isClose) - Number(a.isClose) || (a.nome < b.nome ? -1 : a.nome > b.nome ? 1 : 0),
      )

      let id: number | null = null
      if (repo) {
        try {
          id = await repo.save({ timestamp, resultado: JSON.stringify({ objetos: all, timestamp }), objetos })
        } catch (e) {
          deps.warn?.(`[ANALISE] Falha ao persistir análise: ${(e as Error).message}`)
        }
      }
      return { id, timestamp, objetos }
    },

    async list(page: number, size: number): Promise<Page<AnaliseResponseDTO>> {
      return repo ? repo.list(page, size) : { content: [], totalElements: 0, totalPages: 0, size, number: page }
    },

    async findById(id: number): Promise<AnaliseResponseDTO> {
      const found = repo ? await repo.findById(id) : null
      if (!found) throw new AnaliseNotFoundError(`Análise não encontrada: id=${id}`)
      return found
    },
  }
}
