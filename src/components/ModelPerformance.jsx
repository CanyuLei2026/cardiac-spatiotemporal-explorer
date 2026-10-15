import { useEffect, useState } from 'react'
import InteractiveTraining from './InteractiveTraining.jsx'

function percent(value) {
  return `${(value * 100).toFixed(1)}%`
}

function LossChart({ training }) {
  const epochCount = training.trainLoss.length
  const [epoch, setEpoch] = useState(epochCount)
  const [playing, setPlaying] = useState(false)
  useEffect(() => {
    setEpoch(epochCount)
    setPlaying(false)
  }, [epochCount])
  useEffect(() => {
    if (!playing) return undefined
    const timer = window.setInterval(() => {
      setEpoch((current) => {
        if (current >= epochCount) {
          setPlaying(false)
          return current
        }
        return current + 1
      })
    }, 450)
    return () => window.clearInterval(timer)
  }, [playing, epochCount])

  const width = 520
  const height = 170
  const pad = 24
  const train = training.trainLoss
  const validation = training.validationLoss
  const values = [...train, ...validation]
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  const span = Math.max(maximum - minimum, 1e-6)
  const points = (series) => series.slice(0, epoch).map((value, index) => {
    const x = pad + (index / Math.max(series.length - 1, 1)) * (width - pad * 2)
    const y = pad + (1 - (value - minimum) / span) * (height - pad * 2)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  const epochX = pad + ((epoch - 1) / Math.max(epochCount - 1, 1)) * (width - pad * 2)
  const replay = () => {
    if (playing) {
      setPlaying(false)
      return
    }
    if (epoch >= epochCount) setEpoch(1)
    setPlaying(true)
  }
  return (
    <div className="loss-chart-wrap">
      <div className="monitor-heading"><div><span>Training monitor</span><strong>Recorded epoch replay</strong></div><button onClick={replay}>{playing ? 'Pause' : 'Replay'}</button></div>
      <div className="epoch-readout">
        <div><span>Epoch</span><strong>{epoch} / {epochCount}</strong></div>
        <div><span>Training loss</span><strong>{train[epoch - 1].toFixed(4)}</strong></div>
        <div><span>Validation loss</span><strong>{validation[epoch - 1].toFixed(4)}</strong></div>
      </div>
      <svg className="loss-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training and validation loss by epoch">
        {[0, 0.5, 1].map((ratio) => <line key={ratio} x1={pad} x2={width - pad} y1={pad + ratio * (height - pad * 2)} y2={pad + ratio * (height - pad * 2)} />)}
        <polyline className="loss-train" points={points(train)} />
        <polyline className="loss-validation" points={points(validation)} />
        <line className="epoch-cursor" x1={epochX} x2={epochX} y1={pad} y2={height - pad} />
      </svg>
      <input className="epoch-slider" type="range" min="1" max={epochCount} value={epoch} onChange={(event) => { setPlaying(false); setEpoch(Number(event.target.value)) }} aria-label="Training epoch" />
      <div className="chart-legend"><span><i className="train" />Training</span><span><i className="validation" />Validation</span><b>Best epoch {training.bestEpoch}</b></div>
    </div>
  )
}

function SplitMetrics({ metrics }) {
  const rows = [
    ['Training', metrics.training],
    ['Validation', metrics.validation],
    ['Test', metrics.test],
  ]
  return (
    <div className="split-metrics-card">
      <div className="monitor-heading"><div><span>Final split metrics</span><strong>Best saved MLP checkpoint</strong></div></div>
      <div className="split-metrics-scroll">
        <table className="split-metrics-table">
          <thead><tr><th>Dataset</th><th>Accuracy</th><th>Error</th><th>F1</th><th>Recall</th><th>AUROC</th><th>Samples</th></tr></thead>
          <tbody>
            {rows.map(([label, values]) => (
              <tr key={label}><th>{label}</th><td>{percent(values.accuracy)}</td><td>{percent(values.errorRate)}</td><td>{percent(values.f1)}</td><td>{percent(values.recall)}</td><td>{values.auroc.toFixed(3)}</td><td>{values.sampleCount.toLocaleString()}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function ModelPerformance({ model, data }) {
  if (!model) return null
  const mlp = model.metrics.mlp
  const splitMetrics = model.metrics.mlpBySplit
  const baseline = model.metrics.logisticRegression
  const matrix = mlp.confusionMatrix
  return (
    <section className="model-panel panel">
      <div className="model-panel-heading">
        <div><span className="eyebrow">HW2 · Model report</span><h2>Activation classification performance</h2></div>
        <p>MLP 44 → 64 → 32 → 2 · next 20 frames</p>
      </div>
      {model.prototype && <div className="prototype-notice"><strong>Development model</strong><span>This local prototype uses a point split from one sample case. It will be replaced by the five-case held-out model before submission.</span></div>}
      <div className="case-split">
        <span><b>Training</b>{model.caseSplit.training.join(', ')}</span>
        <span><b>Validation</b>{model.caseSplit.validation.join(', ')}</span>
        <span><b>Test</b>{model.caseSplit.test.join(', ')}</span>
      </div>
      <div className="model-report-grid">
        <div className="metric-stack">
          <div className="metric-grid">
            <div><span>Test accuracy</span><strong>{percent(mlp.accuracy)}</strong></div>
            <div><span>F1 score</span><strong>{percent(mlp.f1)}</strong></div>
            <div><span>Recall</span><strong>{percent(mlp.recall)}</strong></div>
            <div><span>AUROC</span><strong>{mlp.auroc.toFixed(3)}</strong></div>
          </div>
          <div className="baseline-comparison">
            <span>Held-out error rate</span>
            <div><b>MLP</b><strong>{percent(mlp.errorRate)}</strong></div>
            <div><b>Linear baseline</b><strong>{percent(baseline.errorRate)}</strong></div>
          </div>
        </div>
        <LossChart training={model.training} />
        <div className="confusion-card">
          <span>Confusion matrix</span>
          <div className="confusion-axis">Predicted</div>
          <div className="confusion-grid">
            <i /> <b>No</b><b>Yes</b>
            <b>No</b><strong>{matrix[0][0]}</strong><strong>{matrix[0][1]}</strong>
            <b>Yes</b><strong>{matrix[1][0]}</strong><strong>{matrix[1][1]}</strong>
          </div>
          <small>Rows show actual labels. Columns show predictions.</small>
        </div>
      </div>
      {splitMetrics && <SplitMetrics metrics={splitMetrics} />}
      <InteractiveTraining data={data} />
    </section>
  )
}
