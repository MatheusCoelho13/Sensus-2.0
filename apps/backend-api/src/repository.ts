import type { Pool } from 'pg'
import type { AnaliseResponseDTO, ObjetoDetectadoDTO, Page } from './types'

export type AnaliseRepository = {
  save(a: { timestamp: number; resultado: string; objetos: ObjetoDetectadoDTO[] }): Promise<number>
  list(page: number, size: number): Promise<Page<AnaliseResponseDTO>>
  findById(id: number): Promise<AnaliseResponseDTO | null>
}

type ObjetoRow = {
  analise_id: string
  nome: string
  confidence: number | null
  distancia: string | null
  is_close: boolean | null
  bbox_x: number | null
  bbox_y: number | null
  bbox_width: number | null
  bbox_height: number | null
}

const toObjeto = (r: ObjetoRow): ObjetoDetectadoDTO => ({
  nome: r.nome,
  confidence: r.confidence,
  distancia: r.distancia,
  isClose: r.is_close ?? false,
  bboxX: r.bbox_x,
  bboxY: r.bbox_y,
  bboxWidth: r.bbox_width,
  bboxHeight: r.bbox_height,
})

export function createPgRepository(pool: Pool): AnaliseRepository {
  async function hydrate(rows: { id: string; timestamp: string }[]): Promise<AnaliseResponseDTO[]> {
    if (rows.length === 0) return []
    const ids = rows.map((r) => r.id)
    const { rows: objs } = await pool.query<ObjetoRow>(
      'SELECT * FROM objeto_detectado WHERE analise_id = ANY($1::bigint[]) ORDER BY id',
      [ids],
    )
    return rows.map((r) => ({
      id: Number(r.id),
      timestamp: Number(r.timestamp),
      objetos: objs.filter((o) => o.analise_id === r.id).map(toObjeto),
    }))
  }

  return {
    async save({ timestamp, resultado, objetos }) {
      const client = await pool.connect()
      try {
        await client.query('BEGIN')
        const { rows } = await client.query<{ id: string }>(
          'INSERT INTO analise (timestamp, resultado) VALUES ($1, $2) RETURNING id',
          [timestamp, resultado],
        )
        const id = rows[0]!.id
        for (const o of objetos) {
          await client.query(
            `INSERT INTO objeto_detectado
               (nome, confidence, distancia, is_close, bbox_x, bbox_y, bbox_width, bbox_height, analise_id)
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`,
            [o.nome, o.confidence, o.distancia, o.isClose, o.bboxX, o.bboxY, o.bboxWidth, o.bboxHeight, id],
          )
        }
        await client.query('COMMIT')
        return Number(id)
      } catch (e) {
        await client.query('ROLLBACK').catch(() => {})
        throw e
      } finally {
        client.release()
      }
    },

    async list(page, size) {
      const [{ rows }, { rows: count }] = await Promise.all([
        pool.query<{ id: string; timestamp: string }>(
          'SELECT id, timestamp FROM analise ORDER BY timestamp DESC, id DESC LIMIT $1 OFFSET $2',
          [size, page * size],
        ),
        pool.query<{ n: string }>('SELECT count(*) AS n FROM analise'),
      ])
      const total = Number(count[0]!.n)
      return {
        content: await hydrate(rows),
        totalElements: total,
        totalPages: Math.ceil(total / size),
        size,
        number: page,
      }
    },

    async findById(id) {
      const { rows } = await pool.query<{ id: string; timestamp: string }>(
        'SELECT id, timestamp FROM analise WHERE id = $1',
        [id],
      )
      return (await hydrate(rows))[0] ?? null
    },
  }
}
