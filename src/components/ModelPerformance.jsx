function percent(value) {
  return `${(value * 100).toFixed(1)}%`
}

function LossChart({ training }) {
  const width = 520
  const height = 170
  const pad = 24
  const train = training.trainLoss
  const validation = training.validationLoss
  const values = [...train, ...validation]
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  const span = Math.max(maximum - minimum, 1e-6)
  const points = (series) => series.map((value, index) => {
    const x = pad + (index / Math.max(series.length - 1, 1)) * (width - pad * 2)
    const y = pad + (1 - (value - minimum) / span) * (height - pad * 2)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  return (
    <div className="loss-chart-wrap">
      <svg className="loss-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training and validation loss by epoch">
        {[0, 0.5, 1].map((ratio) => <line key={ratio} x1={pad} x2={width - pad} y1={pad + ratio * (height - pad * 2)} y2={pad + ratio * (height - pad * 2)} />)}
        <polyline className="loss-train" points={points(train)} />
        <polyline className="loss-validation" points={points(validation)} />
      </svg>
      <div className="chart-legend"><span><i className="train" />Training</span><span><i className="validation" />Validation</span><b>Best epoch {training.bestEpoch}</b></div>
    </div>
  )
}

export default function ModelPerformance({ model }) {
  if (!model) return null
  const mlp = model.metrics.mlp
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
    </section>
  )
}
