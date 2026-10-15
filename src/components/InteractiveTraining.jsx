import { useEffect, useMemo, useRef, useState } from 'react'

const INITIAL_SETTINGS = { learningRate: '0.0001', epochs: 20, hiddenSize: 32 }

function MetricChart({ history, field, title, format }) {
  const width = 520
  const height = 150
  const pad = 25
  const values = history.flatMap((entry) => [entry.train[field], entry.validation[field]])
  const minimum = field === 'accuracy' ? 0 : Math.min(...values, 0)
  const maximum = field === 'accuracy' ? 1 : Math.max(...values, 1e-6)
  const span = Math.max(maximum - minimum, 1e-6)
  const points = (split) => history.map((entry, index) => {
    const x = pad + (index / Math.max(history.length - 1, 1)) * (width - pad * 2)
    const y = pad + (1 - (entry[split][field] - minimum) / span) * (height - pad * 2)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  const latest = history.at(-1)
  return (
    <div className="sandbox-chart-card">
      <div className="sandbox-chart-title"><span>{title}</span>{latest && <strong>{format(latest.validation[field])} validation</strong>}</div>
      {history.length ? (
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} during interactive training`}>
          {[0, 0.5, 1].map((ratio) => <line key={ratio} x1={pad} x2={width - pad} y1={pad + ratio * (height - pad * 2)} y2={pad + ratio * (height - pad * 2)} />)}
          <polyline className="sandbox-train-line" points={points('train')} />
          <polyline className="sandbox-validation-line" points={points('validation')} />
        </svg>
      ) : <div className="sandbox-chart-empty">Start training to create this chart.</div>}
    </div>
  )
}

export default function InteractiveTraining({ data }) {
  const [settings, setSettings] = useState(INITIAL_SETTINGS)
  const [history, setHistory] = useState([])
  const [status, setStatus] = useState('idle')
  const [sampleCounts, setSampleCounts] = useState(null)
  const [bestEpoch, setBestEpoch] = useState(null)
  const [error, setError] = useState('')
  const workerRef = useRef(null)

  useEffect(() => () => workerRef.current?.terminate(), [])

  const latest = history.at(-1)
  const progress = latest ? latest.epoch / Number(settings.epochs) : 0
  const statusText = useMemo(() => {
    if (status === 'preparing') return 'Preparing compact samples…'
    if (status === 'training') return `Training epoch ${latest?.epoch ?? 0} of ${settings.epochs}`
    if (status === 'complete') return `Training complete · best epoch ${bestEpoch}`
    if (status === 'stopped') return 'Training stopped'
    return 'Ready to train in this browser'
  }, [status, latest, settings.epochs, bestEpoch])

  const startTraining = () => {
    if (!data || status === 'preparing' || status === 'training') return
    workerRef.current?.terminate()
    const worker = new Worker(new URL('../interactiveTrainingWorker.js', import.meta.url), { type: 'module' })
    workerRef.current = worker
    setHistory([])
    setBestEpoch(null)
    setSampleCounts(null)
    setError('')
    setStatus('preparing')
    worker.onmessage = ({ data: message }) => {
      if (message.type === 'ready') {
        setSampleCounts({ training: message.trainingSamples, validation: message.validationSamples })
        setStatus('training')
      } else if (message.type === 'epoch') {
        setHistory((current) => [...current, message])
        setBestEpoch(message.bestEpoch)
      } else if (message.type === 'complete' || message.type === 'stopped') {
        setBestEpoch(message.bestEpoch)
        setStatus(message.type)
        worker.terminate()
        workerRef.current = null
      } else if (message.type === 'error') {
        setError(message.message)
        setStatus('idle')
        worker.terminate()
        workerRef.current = null
      }
    }
    worker.onerror = (event) => {
      setError(event.message || 'Browser training failed.')
      setStatus('idle')
      worker.terminate()
      workerRef.current = null
    }
    const pointsBuffer = data.points.slice().buffer
    const signalsBuffer = data.signals.slice().buffer
    worker.postMessage({
      type: 'train',
      payload: {
        pointsBuffer,
        signalsBuffer,
        pointCount: data.metadata.pointCount,
        timeCount: data.metadata.timeCount,
        learningRate: Number(settings.learningRate),
        epochs: Number(settings.epochs),
        hiddenSize: Number(settings.hiddenSize),
      },
    }, [pointsBuffer, signalsBuffer])
  }

  const stopTraining = () => {
    workerRef.current?.postMessage({ type: 'stop' })
  }

  const updateSetting = (field, value) => setSettings((current) => ({ ...current, [field]: value }))
  const active = status === 'preparing' || status === 'training'
  return (
    <div className="interactive-training">
      <div className="sandbox-heading">
        <div><span className="eyebrow">Bonus · Interactive training</span><h3>Train a small model in your browser</h3></div>
        <span className={`sandbox-status ${active ? 'active' : ''}`}><i />{statusText}</span>
      </div>
      <p className="sandbox-notice"><strong>Training sandbox:</strong> this uses a compact, balanced sample from case_0004 for a quick experiment. It does not replace the formal five-case HPC model or its reported test results.</p>
      <div className="sandbox-controls">
        <label><span>Learning rate</span><select value={settings.learningRate} disabled={active} onChange={(event) => updateSetting('learningRate', event.target.value)}><option value="0.00003">0.00003</option><option value="0.0001">0.0001</option><option value="0.0003">0.0003</option><option value="0.001">0.001</option></select></label>
        <label><span>Epochs</span><input type="number" min="5" max="30" step="1" value={settings.epochs} disabled={active} onChange={(event) => updateSetting('epochs', Math.max(5, Math.min(30, Number(event.target.value))))} /></label>
        <label><span>Hidden units</span><select value={settings.hiddenSize} disabled={active} onChange={(event) => updateSetting('hiddenSize', Number(event.target.value))}><option value="16">16</option><option value="32">32</option><option value="64">64</option></select></label>
        {!active ? <button className="sandbox-start" onClick={startTraining}>Start training</button> : <button className="sandbox-stop" onClick={stopTraining}>Stop</button>}
      </div>
      <div className="sandbox-progress"><i style={{ width: `${progress * 100}%` }} /></div>
      {sampleCounts && <p className="sandbox-samples">One hidden layer · 44 inputs · {sampleCounts.training} training samples · {sampleCounts.validation} validation samples</p>}
      {error && <div className="sandbox-error">{error}</div>}
      <div className="sandbox-live-metrics">
        <div><span>Epoch</span><strong>{latest?.epoch ?? '—'} / {settings.epochs}</strong></div>
        <div><span>Train loss</span><strong>{latest ? latest.train.loss.toFixed(4) : '—'}</strong></div>
        <div><span>Validation loss</span><strong>{latest ? latest.validation.loss.toFixed(4) : '—'}</strong></div>
        <div><span>Train accuracy</span><strong>{latest ? `${(latest.train.accuracy * 100).toFixed(1)}%` : '—'}</strong></div>
        <div><span>Validation accuracy</span><strong>{latest ? `${(latest.validation.accuracy * 100).toFixed(1)}%` : '—'}</strong></div>
        <div><span>Best epoch</span><strong>{bestEpoch ?? '—'}</strong></div>
      </div>
      <div className="sandbox-charts">
        <MetricChart history={history} field="loss" title="Loss" format={(value) => value.toFixed(4)} />
        <MetricChart history={history} field="accuracy" title="Accuracy" format={(value) => `${(value * 100).toFixed(1)}%`} />
      </div>
      <div className="sandbox-legend"><span><i className="train" />Training</span><span><i className="validation" />Validation</span></div>
    </div>
  )
}
