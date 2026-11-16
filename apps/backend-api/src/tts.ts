export type Tts = (text: string, isClose: boolean) => Promise<Buffer>

export function createElevenLabsTts(cfg: { apiKey: string; voiceId: string; model: string }): Tts {
  return async (text, isClose) => {
    if (!cfg.apiKey) throw new Error('ElevenLabs API Key is not configured.')
    const res = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(cfg.voiceId)}`, {
      method: 'POST',
      headers: { 'xi-api-key': cfg.apiKey, 'content-type': 'application/json', accept: 'audio/mpeg' },
      body: JSON.stringify({
        text,
        model_id: cfg.model,
        voice_settings: {
          stability: isClose ? 0.35 : 0.55,
          similarity_boost: isClose ? 0.85 : 0.75,
          speed: isClose ? 1.15 : 1.0,
        },
      }),
      signal: AbortSignal.timeout(10_000),
    })
    if (!res.ok) throw new Error(`ElevenLabs API error: ${res.status}`)
    return Buffer.from(await res.arrayBuffer())
  }
}
