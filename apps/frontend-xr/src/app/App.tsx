import { useCallback, useEffect, useRef, useState } from 'react'
import { CameraView } from '../components/CameraView'
import { ConnectionGuard } from '../components/ConnectionGuard'
import { useWebSocket } from '../hooks/useWebSocket'
import { useFrameSender } from '../hooks/useFrameSender'
import { useAudioAnnouncer } from '../hooks/useAudioAnnouncer'
import { webSocketService } from '../services/websocket.service'
import type { AnaliseResponse } from '../types/AnaliseResponse'
import type { ObjetoDetectado } from '../types/ObjetoDetectado'
import type { WebSocketError } from '../types/WebSocketPayload'
import '../styles/globals.css'

function isAnaliseResponse(data: unknown): data is AnaliseResponse {
  return typeof data === 'object' && data !== null && 'objetos' in data && Array.isArray((data as any).objetos)
}
function isWebSocketError(data: unknown): data is WebSocketError {
  return typeof data === 'object' && data !== null && 'erro' in data
}

const ptLabel: Record<string, string> = {
  person: 'pessoa', bicycle: 'bicicleta', car: 'carro', motorcycle: 'moto',
  bus: 'ônibus', train: 'trem', truck: 'caminhão', boat: 'barco',
  bench: 'banco', cat: 'gato', dog: 'cachorro', backpack: 'mochila',
  bottle: 'garrafa', cup: 'xícara', fork: 'garfo', knife: 'faca',
  banana: 'banana', apple: 'maçã', sandwich: 'sanduíche', orange: 'laranja',
  pizza: 'pizza', donut: 'rosquinha', cake: 'bolo', chair: 'cadeira',
  couch: 'sofá', bed: 'cama', toilet: 'vaso', tv: 'tv',
  laptop: 'notebook', mouse: 'mouse', remote: 'controle', keyboard: 'teclado',
  'cell phone': 'celular', sink: 'pia', refrigerator: 'geladeira',
  book: 'livro', clock: 'relógio', scissors: 'tesoura', toothbrush: 'escova',
}
const ptOf = (l: string) => ptLabel[l.toLowerCase()] || l

// A distância é o que importa para quem se desloca, então ela (e não a categoria) dá a cor.
type Dist = 'perto' | 'medio' | 'longe'
const DISTS: Dist[] = ['perto', 'medio', 'longe']
const DIST_TEXT: Record<Dist, string> = { perto: 'perto', medio: 'médio', longe: 'longe' }
const distOf = (o: ObjetoDetectado): Dist => (o.isClose ? 'perto' : o.distancia ?? 'longe')

// Canvas não lê CSS vars: manter em sincronia com --signal / --fg / --bg em globals.css.
const DIST: Record<Dist, { stroke: string; fill: string; w: number; dash: number[]; tabBg: string; tabFg: string }> = {
  perto: { stroke: '#FFC800', fill: 'transparent', w: 3, dash: [], tabBg: '#FFC800', tabFg: '#15140F' },
  medio: { stroke: '#F2EFE6', fill: 'transparent', w: 2, dash: [], tabBg: '#0F0F0D', tabFg: '#F2EFE6' },
  longe: { stroke: 'rgba(242,239,230,0.7)', fill: 'transparent', w: 1.5, dash: [6, 5], tabBg: '#0F0F0D', tabFg: '#F2EFE6' },
}

function groupByDist(objs: ObjetoDetectado[]) {
  return DISTS.map(d => ({ d, objs: objs.filter(o => distOf(o) === d) }))
}
function tally(objs: ObjetoDetectado[]) {
  const m = new Map<string, number>()
  objs.forEach(o => m.set(o.nome, (m.get(o.nome) ?? 0) + 1))
  return [...m].map(([n, c]) => (c > 1 ? `${ptOf(n)} ×${c}` : ptOf(n))).join(', ')
}

const fmtTime = (ts: number) => new Date(ts).toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
const fmtDate = (ts: number) => new Date(ts).toLocaleDateString('pt-BR', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })

function useLocalState<T>(key: string, initial: T): [T, React.Dispatch<React.SetStateAction<T>>] {
  const [v, setV] = useState<T>(() => {
    try { const s = localStorage.getItem(key); return s ? JSON.parse(s) : initial } catch { return initial }
  })
  useEffect(() => { try { localStorage.setItem(key, JSON.stringify(v)) } catch {} }, [key, v])
  return [v, setV]
}

// Três barras que encurtam: a régua de distância é o logo.
function Mark({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size * 0.75} viewBox="0 0 24 18" aria-hidden="true">
      <rect x="1" y="1" width="22" height="4" fill="#FFC800" stroke="currentColor" strokeWidth="1.5"/>
      <rect x="0" y="7" width="15" height="4" fill="currentColor"/>
      <rect x="0" y="14" width="8" height="4" fill="currentColor"/>
    </svg>
  )
}

function getBbox(obj: ObjetoDetectado) {
  return {
    x: obj.bboxX ?? (obj as any).bbox_x,
    y: obj.bboxY ?? (obj as any).bbox_y,
    w: obj.bboxWidth ?? (obj as any).bbox_width,
    h: obj.bboxHeight ?? (obj as any).bbox_height,
  }
}

function drawBboxes(
  canvas: HTMLCanvasElement,
  objetos: ObjetoDetectado[],
  showLabels: boolean,
  showDistance: boolean,
  video?: HTMLVideoElement | null,
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const dpr = window.devicePixelRatio || 1
  const cw = canvas.clientWidth, ch = canvas.clientHeight
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
    canvas.width = cw * dpr; canvas.height = ch * dpr
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cw, ch)

  // O vídeo usa object-fit:cover (preenche o palco e corta o excesso).
  // Mesma escala do CSS, com offset negativo no eixo cortado, para a caixa cair sobre o objeto.
  const vw = video?.videoWidth || cw, vh = video?.videoHeight || ch
  const scale = Math.max(cw / vw, ch / vh)
  const activeW = vw * scale, activeH = vh * scale
  const offsetX = (cw - activeW) / 2, offsetY = (ch - activeH) / 2

  // longe primeiro, perto por último: o que está perto nunca fica coberto.
  const ordered = [...objetos].sort((a, b) => DISTS.indexOf(distOf(b)) - DISTS.indexOf(distOf(a)))
  ordered.forEach(obj => {
    const { x, y, w, h } = getBbox(obj)
    if (x == null || w == null || h == null) return

    const rx = offsetX + x * activeW
    const ry = offsetY + y * activeH
    const rw = w * activeW
    const rh = h * activeH
    const d = distOf(obj)
    const st = DIST[d]

    ctx.fillStyle = st.fill
    ctx.fillRect(rx, ry, rw, rh)
    ctx.strokeStyle = st.stroke
    ctx.lineWidth = st.w
    ctx.setLineDash(st.dash)
    ctx.strokeRect(rx, ry, rw, rh)
    ctx.setLineDash([])

    if (showLabels) {
      const label = showDistance ? `${ptOf(obj.nome)} · ${DIST_TEXT[d]}` : ptOf(obj.nome)
      ctx.font = "700 15px 'Atkinson Hyperlegible Next', system-ui, sans-serif"
      const tw = ctx.measureText(label).width
      const th = 24
      const ty = ry - th >= 0 ? ry - th : ry // sem espaço acima: a etiqueta entra na caixa
      ctx.fillStyle = st.tabBg
      ctx.fillRect(rx, ty, tw + 14, th)
      ctx.fillStyle = st.tabFg
      ctx.textBaseline = 'middle'
      ctx.fillText(label, rx + 7, ty + th / 2 + 1)
    }
  })
}

const CONN_TEXT = { OPEN: 'Conectado', CONNECTING: 'Conectando…', CLOSED: 'Sem conexão' } as const

// Equivale ao que a voz diz: o grupo mais próximo, em letra grande.
function Caption({ objetos, online }: { objetos: ObjetoDetectado[]; online: boolean }) {
  const near = groupByDist(objetos).find(g => g.objs.length)
  const rest = near ? objetos.length - near.objs.length : 0
  return (
    <div className="caption">
      {near ? (
        <>
          <span className="tag" data-d={near.d}>{DIST_TEXT[near.d]}</span>
          <span className="caption-text">{tally(near.objs)}</span>
          {rest > 0 && <span className="more">+{rest} além</span>}
        </>
      ) : (
        <span className="caption-text dim">{online ? 'Nada à vista.' : 'Aguardando conexão…'}</span>
      )}
    </div>
  )
}

function Header({ view, setView, onXR, sessions, wsState }: any) {
  const items = [
    { id: 'live', label: 'Ao vivo' },
    { id: 'sessions', label: 'Sessões', n: sessions.length },
    { id: 'gallery', label: 'Galeria' },
    { id: 'settings', label: 'Ajustes' },
  ]
  return (
    <>
      <div className="brand"><Mark size={26}/> Sensus</div>
      <nav className="nav" aria-label="Principal">
        {items.map(it => (
          <button key={it.id} onClick={() => setView(it.id)} aria-current={view === it.id ? 'page' : undefined}>
            {it.label}{it.n ? <span className="n">{it.n}</span> : null}
          </button>
        ))}
        <button className="xr" onClick={onXR}>Modo XR</button>
      </nav>
      <div className="conn" data-s={wsState} role="status"><i/>{CONN_TEXT[wsState as keyof typeof CONN_TEXT]}</div>
    </>
  )
}

function Page({ title, lede, children }: { title: string; lede: string; children: React.ReactNode }) {
  return (
    <div className="page">
      <h1>{title}</h1>
      <p className="lede">{lede}</p>
      {children}
    </div>
  )
}

function Empty({ title, sub }: { title: string; sub: string }) {
  return <div className="empty">{title}<p>{sub}</p></div>
}

function Ruler({ analise, history, wsState }: { analise: AnaliseResponse | null; history: any[]; wsState: string }) {
  const objetos = analise?.objetos ?? []
  const groups = groupByDist(objetos)
  return (
    <aside className="ruler" aria-label="Detecções por distância">
      <div className="ruler-head">
        <span className="eyebrow">Régua</span>
        <span>{wsState === 'OPEN' ? `${objetos.length} à vista` : 'sem conexão'}</span>
      </div>
      <ol className="bands">
        {groups.map(({ d, objs }) => (
          <li key={d} className="band" data-d={d} data-on={objs.length > 0}>
            <span className="band-name">{DIST_TEXT[d]}</span>
            {objs.length ? (
              <>
                <span className="band-n">{objs.length}</span>
                <span className="band-what">{tally(objs)}</span>
              </>
            ) : <span className="band-none">livre</span>}
          </li>
        ))}
      </ol>
      <div className="ruler-head"><span className="eyebrow">Registro</span></div>
      {history.length === 0 ? (
        <p className="log-empty">Entra aqui quando um objeto novo aparece ou muda de distância.</p>
      ) : (
        <ul className="log">
          {history.slice(0, 12).map((e: any, i: number) => (
            <li key={i}>
              <time>{fmtTime(e.ts)}</time>
              <span>{ptOf(e.nome)}</span>
              <span className="d" data-d={distOf(e)}>{DIST_TEXT[distOf(e)]}</span>
            </li>
          ))}
        </ul>
      )}
    </aside>
  )
}

function LiveView({ analise, history, addSnapshot, settings, wsState }: any) {
  const { wsState: ws } = useWebSocket()
  const [video, setVideo] = useState<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useFrameSender(video, ws === 'OPEN')

  useEffect(() => {
    if (!canvasRef.current || !analise) return
    drawBboxes(canvasRef.current, analise.objetos, settings.showLabels, settings.showDistance, video)
  }, [analise, settings.showLabels, settings.showDistance, video])

  const takeSnapshot = useCallback(() => {
    if (!video) return
    const cap = document.createElement('canvas')
    cap.width = video.videoWidth; cap.height = video.videoHeight
    const ctx = cap.getContext('2d')!
    ctx.drawImage(video, 0, 0)
    addSnapshot({
      id: `snap-${Date.now()}`,
      ts: Date.now(),
      thumb: cap.toDataURL('image/jpeg', 0.78),
      objetos: analise?.objetos ?? [],
    })
  }, [video, analise, addSnapshot])

  return (
    <div className="live">
      <h1 className="sr-only">Ao vivo</h1>
      <div className="stage">
        <ConnectionGuard wsState={wsState}>
          <CameraView onVideoReady={setVideo} />
          <canvas ref={canvasRef} aria-hidden="true"/>
        </ConnectionGuard>
        <div className="foot">
          <Caption objetos={analise?.objetos ?? []} online={wsState === 'OPEN'}/>
          <button className="shutter" onClick={takeSnapshot} disabled={wsState !== 'OPEN'}>Capturar</button>
        </div>
      </div>
      <Ruler analise={analise} history={history} wsState={wsState}/>
    </div>
  )
}

function SessionsView({ sessions, setSessions }: any) {
  const remove = (id: string) => setSessions(sessions.filter((s: any) => s.id !== id))
  return (
    <Page title="Sessões" lede="Cada captura feita ao vivo, com o que estava à vista naquele momento.">
      {sessions.length === 0 ? (
        <Empty title="Nada registrado ainda." sub="Use Capturar na tela Ao vivo." />
      ) : (
        <ul className="rows">
          {sessions.map((s: any) => (
            <li key={s.id}>
              <span className="when">{fmtDate(s.ts)}</span>
              <span className="what">{s.objetos?.length ? tally(s.objetos) : 'nada à vista'}</span>
              <button className="btn danger" onClick={() => remove(s.id)} aria-label={`Excluir sessão de ${fmtDate(s.ts)}`}>Excluir</button>
            </li>
          ))}
        </ul>
      )}
    </Page>
  )
}

function GalleryView({ snapshots, setSnapshots }: any) {
  const [sel, setSel] = useState<any>(null)
  const remove = (id: string) => setSnapshots(snapshots.filter((s: any) => s.id !== id))
  useEffect(() => {
    if (!sel) return
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') setSel(null) }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [sel])
  return (
    <Page title="Galeria" lede="Os quadros que você capturou, guardados só neste aparelho.">
      {snapshots.length === 0 ? (
        <Empty title="Galeria vazia." sub="Capturas feitas ao vivo aparecem aqui." />
      ) : (
        <div className="grid">
          {snapshots.map((s: any) => (
            <button key={s.id} className="shot" onClick={() => setSel(s)}>
              <img src={s.thumb} alt={`Captura de ${fmtDate(s.ts)}, ${s.objetos?.length ?? 0} objetos`}/>
              <span><b>{fmtDate(s.ts)}</b><span>{s.objetos?.length ?? 0} obj.</span></span>
            </button>
          ))}
        </div>
      )}
      {sel && (
        <div className="modal" onClick={() => setSel(null)}>
          <div className="modal-card" role="dialog" aria-modal="true" aria-label="Captura ampliada" onClick={(e: any) => e.stopPropagation()}>
            <img src={sel.thumb} alt={`Captura de ${fmtDate(sel.ts)}`}/>
            <div className="modal-foot">
              <b>{fmtDate(sel.ts)}</b>
              <div>
                <a className="btn" href={sel.thumb} download={`sensus-${sel.id}.jpg`}>Baixar</a>
                <button className="btn danger" onClick={() => { remove(sel.id); setSel(null) }}>Excluir</button>
                <button className="btn" onClick={() => setSel(null)} autoFocus>Fechar</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </Page>
  )
}

function Switch({ label, hint, value, onChange }: { label: string; hint: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="row">
      <div>
        <div className="row-label">{label}</div>
        <div className="row-hint">{hint}</div>
      </div>
      <button className="switch" role="switch" aria-checked={value} aria-label={label} onClick={() => onChange(!value)}>
        {value ? 'Ligado' : 'Desligado'}
      </button>
    </div>
  )
}

function SettingsView({ settings, setSettings }: any) {
  const set = (k: string, v: any) => setSettings({ ...settings, [k]: v })
  return (
    <Page title="Ajustes" lede="O que aparece na tela e o que o Sensus fala em voz alta.">
      <section className="sect">
        <h2>Na tela</h2>
        <Switch label="Nome do objeto" hint="Etiqueta em cada caixa desenhada sobre a imagem." value={settings.showLabels} onChange={v => set('showLabels', v)} />
        <Switch label="Distância na etiqueta" hint="Acrescenta perto, médio ou longe ao nome." value={settings.showDistance} onChange={v => set('showDistance', v)} />
      </section>
      <section className="sect">
        <h2>Na voz</h2>
        <Switch label="Narração" hint="Descreve o ambiente em voz alta, do mais próximo ao mais distante." value={settings.ttsEnabled} onChange={v => set('ttsEnabled', v)} />
      </section>
      <section className="sect">
        <h2>Sobre</h2>
        <p className="about">Sensus é uma camada de percepção espacial para quem enxerga pouco ou usa dispositivos XR. Projeto da disciplina de Oficina de IA.</p>
      </section>
    </Page>
  )
}

function XRView({ onExit, analise, wsState, settings }: any) {
  const { wsState: ws } = useWebSocket()
  const [video, setVideo] = useState<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useFrameSender(video, ws === 'OPEN')
  useEffect(() => {
    if (!canvasRef.current || !analise) return
    drawBboxes(canvasRef.current, analise.objetos, true, settings.showDistance, video)
  }, [analise, video, settings.showDistance])
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onExit() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onExit])
  return (
    <div className="xr-view">
      <ConnectionGuard wsState={wsState}>
        <CameraView onVideoReady={setVideo}/>
        <canvas ref={canvasRef} aria-hidden="true"/>
      </ConnectionGuard>
      <div className="xr-brand"><Mark size={22}/> Sensus XR</div>
      <div className="foot">
        <Caption objetos={analise?.objetos ?? []} online={wsState === 'OPEN'}/>
        <button className="exit" onClick={onExit}>Sair · Esc</button>
      </div>
    </div>
  )
}

export function App() {
  const { wsState } = useWebSocket()
  const [view, setView] = useState('live')
  const [xrMode, setXrMode] = useState(false)
  const [analise, setAnalise] = useState<AnaliseResponse | null>(null)
  const [history, setHistory] = useState<any[]>([])
  const [sessions, setSessions] = useLocalState<any[]>('sensus.sessions', [])
  const [snapshots, setSnapshots] = useLocalState<any[]>('sensus.snapshots', [])
  const [settings, setSettings] = useLocalState('sensus.settings', {
    showLabels: true, showDistance: true, ttsEnabled: true,
  })

  const { announce } = useAudioAnnouncer()

  const seenRef = useRef<Map<string, number>>(new Map())
  const addHistory = useCallback((evt: any) => {
    // Registra quando algo muda (objeto novo ou outra distância), não a cada quadro.
    const key = `${evt.nome}|${distOf(evt)}`
    const last = seenRef.current.get(key) ?? 0
    if (Date.now() - last < 15000) return
    seenRef.current.set(key, Date.now())
    setHistory(h => [evt, ...h].slice(0, 200))
  }, [])

  const addSnapshot = useCallback((snap: any) => {
    setSnapshots((s: any[]) => [snap, ...s].slice(0, 60))
    setSessions((s: any[]) => [snap, ...s].slice(0, 100))
  }, [setSnapshots, setSessions])

  useEffect(() => {
    const handler = (data: unknown) => {
      if (isWebSocketError(data)) return
      if (isAnaliseResponse(data)) {
        setAnalise(data)
        const seen = new Set<string>()
        data.objetos.forEach(obj => {
          if (!seen.has(obj.nome)) {
            seen.add(obj.nome)
            addHistory({ ...obj, ts: Date.now() })
          }
        })
        announce(data.objetos, (settings as any).ttsEnabled ?? false)
      }
    }
    webSocketService.onMessage(handler)
    return () => webSocketService.offMessage(handler)
  }, [addHistory, announce, settings])

  if (xrMode) return <XRView onExit={() => setXrMode(false)} analise={analise} wsState={wsState} settings={settings}/>

  return (
    <div className="app">
      <Header view={view} setView={setView} sessions={sessions} wsState={wsState} onXR={() => {
        if (typeof window !== 'undefined' && window.speechSynthesis) {
          // Desbloqueia o Autoplay Policy do navegador enviando um áudio mudo
          const u = new SpeechSynthesisUtterance('');
          u.volume = 0;
          window.speechSynthesis.speak(u);
        }
        setXrMode(true);
      }}/>
      <main className="main">
        {view === 'live' && <LiveView analise={analise} history={history} addSnapshot={addSnapshot} settings={settings} wsState={wsState}/>}
        {view === 'sessions' && <SessionsView sessions={sessions} setSessions={setSessions}/>}
        {view === 'gallery' && <GalleryView snapshots={snapshots} setSnapshots={setSnapshots}/>}
        {view === 'settings' && <SettingsView settings={settings} setSettings={setSettings}/>}
      </main>
    </div>
  )
}
