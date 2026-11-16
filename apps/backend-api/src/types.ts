// Wire shape the current frontend consumes (Portuguese, see frontend-xr/src/types/AnaliseResponse.ts).
// Domain input from the IA service is the shared-contracts DetectedObject.
export type ObjetoDetectadoDTO = {
  nome: string
  confidence: number | null
  distancia: string | null
  isClose: boolean
  bboxX: number | null
  bboxY: number | null
  bboxWidth: number | null
  bboxHeight: number | null
}

export type AnaliseResponseDTO = {
  id: number | null
  timestamp: number
  objetos: ObjetoDetectadoDTO[]
}

export type Page<T> = {
  content: T[]
  totalElements: number
  totalPages: number
  size: number
  number: number
}

export type ApiResponse<T> = { sucesso: boolean; mensagem: string; dados: T | null; timestamp: number }

const nowSec = () => Math.floor(Date.now() / 1000)
export const ok = <T>(dados: T): ApiResponse<T> => ({ sucesso: true, mensagem: 'OK', dados, timestamp: nowSec() })
export const erro = (mensagem: string): ApiResponse<never> => ({ sucesso: false, mensagem, dados: null, timestamp: nowSec() })

export class IAServiceError extends Error {}
export class InvalidImageError extends Error {}
export class AnaliseNotFoundError extends Error {}
