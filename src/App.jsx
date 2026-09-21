import { useEffect, useMemo, useRef, useState } from 'react'
import HeartViewer from './components/HeartViewer.jsx'
import SignalPlot from './components/SignalPlot.jsx'
import { loadCardiacCase, signalAt, traceFor } from './data.js'

const DEFAULT_FILTERS = { region: 'all', pointType: 'all', component: 'all' }
const LABELS = ['Normal', 'Abnormal', 'Landmark']
const REGION_LABELS = {
  0: 'Myocardial interior',
  1: 'Outer surface',
  2: 'LV endocardium',
  3: 'RV endocardium',
}
const POINT_TYPE_LABELS = {
  1: 'Interior sample',
  2: 'Endocardial surface',
  3: 'Epicardial surface',
  4: 'Full-boundary sample',
}
const COMPONENT_LABELS = { 1: 'Main connected component' }

function FilterSelect({ label, value, values, labels, onChange }) {
  const options = Array.isArray(values) ? values : [values]
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="all">All</option>
        {options.map((item) => <option key={item} value={item}>{item} — {labels[item] || `Group ${item}`}</option>)}
      </select>
    </label>
  )
}

function App() {
  const [data, setData] = useState(null)
  const [status, setStatus] = useState({ loading: true, error: '' })
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [colorMode, setColorMode] = useState('adaptive')
  const [selected, setSelected] = useState(0)
  const [filters, setFilters] = useState(DEFAULT_FILTERS)
  const [resetVersion, setResetVersion] = useState(0)
  const [label, setLabel] = useState('Landmark')
  const [note, setNote] = useState('')
  const importRef = useRef()
  const [annotations, setAnnotations] = useState(() => {
    try { return JSON.parse(localStorage.getItem('cardiac-annotations') || '[]') } catch { return [] }
  })

  useEffect(() => {
    loadCardiacCase()
      .then((loaded) => {
        setData(loaded)
        setStatus({ loading: false, error: '' })
      })
      .catch((error) => setStatus({ loading: false, error: error.message }))
  }, [])

  useEffect(() => {
    localStorage.setItem('cardiac-annotations', JSON.stringify(annotations))
  }, [annotations])

  useEffect(() => {
    if (!playing || !data) return undefined
    const timer = window.setInterval(() => {
      setFrame((current) => (current + 1) % data.metadata.timeCount)
    }, Math.max(20, 90 / speed))
    return () => window.clearInterval(timer)
  }, [playing, speed, data])

  const selectedTrace = useMemo(() => (data && selected !== null ? traceFor(data, selected) : null), [data, selected])
  const currentValue = data && selected !== null ? signalAt(data, selected, frame) : null
  const currentAnnotation = annotations.find((item) => item.pointIndex === selected)
  const colorRange = useMemo(() => {
    if (!data || colorMode === 'fixed') return data ? [data.metadata.signalMin, data.metadata.signalMax] : [0, 1]
    const values = new Array(data.metadata.pointCount)
    for (let index = 0; index < values.length; index += 1) values[index] = signalAt(data, index, frame)
    values.sort((a, b) => a - b)
    const low = values[Math.floor((values.length - 1) * 0.02)]
    const high = values[Math.ceil((values.length - 1) * 0.98)]
    return high - low > 0.015 ? [low, high] : [data.metadata.signalMin, data.metadata.signalMax]
  }, [data, frame, colorMode])

  const saveAnnotation = () => {
    if (selected === null) return
    const next = {
      pointIndex: selected,
      originalPointIndex: data.originalIndices[selected],
      label,
      note: note.trim(),
    }
    setAnnotations((items) => [...items.filter((item) => item.pointIndex !== selected), next])
    setNote('')
  }

  const exportAnnotations = () => {
    const payload = JSON.stringify({ caseId: data.metadata.caseId, annotations }, null, 2)
    const url = URL.createObjectURL(new Blob([payload], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${data.metadata.caseId}-annotations.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const importAnnotations = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    try {
      const payload = JSON.parse(await file.text())
      if (!Array.isArray(payload.annotations)) throw new Error('Missing annotations array')
      setAnnotations(payload.annotations)
    } catch (error) {
      window.alert(`Could not import annotations: ${error.message}`)
    } finally {
      event.target.value = ''
    }
  }

  if (status.loading) {
    return <main className="state-screen"><div className="pulse" /><h1>Loading cardiac case…</h1><p>Preparing geometry and 801 signal frames.</p></main>
  }

  if (status.error) {
    return <main className="state-screen error"><h1>Dataset could not be loaded</h1><p>{status.error}</p><p>Run this application through the Vite development server.</p></main>
  }

  const metadata = data.metadata
  const filteredCount = Array.from({ length: metadata.pointCount }, (_, index) => index).filter((index) =>
    (filters.region === 'all' || data.regions[index] === Number(filters.region)) &&
    (filters.pointType === 'all' || data.pointTypes[index] === Number(filters.pointType)) &&
    (filters.component === 'all' || data.components[index] === Number(filters.component))
  ).length

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">DS7400 · HW1</div>
          <h1>Cardiac Spatiotemporal Explorer</h1>
        </div>
        <div className="case-badge"><span className="live-dot" />{metadata.caseId}<small>{metadata.pointCount.toLocaleString()} sampled points</small></div>
      </header>

      <section className="workspace">
        <aside className="control-panel panel">
          <div className="panel-heading"><span>01</span><h2>Explore</h2></div>
          <div className="summary-grid">
            <div><strong>{metadata.originalPointCount.toLocaleString()}</strong><span>source points</span></div>
            <div><strong>{metadata.timeCount}</strong><span>time frames</span></div>
          </div>

          <div className="control-group">
            <h3>Point filters</h3>
            <FilterSelect label="Surface region" value={filters.region} values={metadata.regions} labels={REGION_LABELS} onChange={(region) => setFilters({ ...filters, region })} />
            <FilterSelect label="Point type" value={filters.pointType} values={metadata.pointTypes} labels={POINT_TYPE_LABELS} onChange={(pointType) => setFilters({ ...filters, pointType })} />
            <FilterSelect label="Graph component" value={filters.component} values={metadata.components} labels={COMPONENT_LABELS} onChange={(component) => setFilters({ ...filters, component })} />
            <button className="text-button" onClick={() => setFilters(DEFAULT_FILTERS)}>Clear filters</button>
            <p className="muted">Showing {filteredCount.toLocaleString()} of {metadata.pointCount.toLocaleString()} sampled points</p>
            <div className="filter-key">
              <strong>Label key</strong>
              <p><b>Point type:</b> 1 interior; 2 endocardial; 3 epicardial; 4 full boundary.</p>
              <p><b>Surface region:</b> 0 interior; 1 outer surface; 2 LV endocardium; 3 RV endocardium.</p>
              <p><b>Graph component:</b> connected groups in the k-NN graph. This case has one main component.</p>
            </div>
          </div>

          <div className="control-group annotation-group">
            <h3>Point annotation</h3>
            <label className="field"><span>Label</span><select value={label} onChange={(event) => setLabel(event.target.value)}>{LABELS.map((item) => <option key={item}>{item}</option>)}</select></label>
            <label className="field"><span>Note</span><textarea value={note} onChange={(event) => setNote(event.target.value)} placeholder="Optional observation" rows="2" /></label>
            <button className="primary-button" onClick={saveAnnotation}>Save marker</button>
            <div className="button-row">
              <button onClick={exportAnnotations} disabled={!annotations.length}>Export</button>
              <button onClick={() => importRef.current?.click()}>Import</button>
              <input ref={importRef} type="file" accept="application/json" hidden onChange={importAnnotations} />
            </div>
            <div className="annotation-list-heading">
              <span>Saved annotations</span>
              <strong>{annotations.length}</strong>
            </div>
            {annotations.length ? (
              <div className="annotation-list">
                {annotations.map((annotation) => (
                  <div
                    className={`annotation-row ${selected === annotation.pointIndex ? 'active' : ''}`}
                    key={annotation.pointIndex}
                  >
                    <button
                      className="annotation-target"
                      onClick={() => setSelected(annotation.pointIndex)}
                      title="Select this point in the 3D viewer"
                    >
                      <i className={`annotation-dot ${annotation.label.toLowerCase()}`} />
                      <span className="annotation-copy">
                        <span><b>#{annotation.originalPointIndex}</b><em>{annotation.label}</em></span>
                        <small>{annotation.note || 'No note'}</small>
                      </span>
                    </button>
                    <button
                      className="annotation-delete"
                      aria-label={`Delete annotation for point ${annotation.originalPointIndex}`}
                      title="Delete annotation"
                      onClick={() => setAnnotations((items) => items.filter((item) => item.pointIndex !== annotation.pointIndex))}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            ) : <p className="muted annotation-empty">No saved annotations yet.</p>}
          </div>
        </aside>

        <section className="visual-panel panel">
          <div className="viewer-toolbar">
            <div><span className="toolbar-label">Electrical signal</span><strong>U<sub>heart</sub> · simulation time = {metadata.time[frame].toFixed(1)}</strong></div>
            <div className="viewer-actions">
              <label><span>Color contrast</span><select value={colorMode} onChange={(event) => setColorMode(event.target.value)}><option value="adaptive">Adaptive per frame</option><option value="fixed">Fixed full case</option></select></label>
              <button onClick={() => setResetVersion((value) => value + 1)}>Reset view</button>
            </div>
          </div>
          <div className="heart-canvas">
            <HeartViewer
              data={data}
              frame={frame}
              filters={filters}
              selected={selected}
              onSelect={setSelected}
              annotations={annotations}
              resetVersion={resetVersion}
              colorRange={colorRange}
            />
            <div className="viewer-hint">Drag to rotate · Scroll to zoom · Click a point to inspect</div>
            <div className="color-legend"><span>{colorRange[0].toFixed(2)}</span><i /><span>{colorRange[1].toFixed(2)}</span></div>
          </div>
          <div className="timeline">
            <button className="play-button" onClick={() => setPlaying((value) => !value)} aria-label={playing ? 'Pause animation' : 'Play animation'}>{playing ? 'Ⅱ' : '▶'}</button>
            <input type="range" min="0" max={metadata.timeCount - 1} value={frame} onChange={(event) => setFrame(Number(event.target.value))} aria-label="Time frame" />
            <output>{frame + 1} / {metadata.timeCount}</output>
            <label className="speed"><span>Speed</span><select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
          </div>
        </section>

        <aside className="detail-panel panel">
          <div className="panel-heading"><span>02</span><h2>Point inspector</h2></div>
          <p className="inspector-help">Each 3D point has its own 801-frame electrical signal. Click a point to highlight it and inspect that signal below.</p>
          {selected !== null ? (
            <>
              <div className="point-id">#{data.originalIndices[selected]}</div>
              <dl className="point-details">
                <div><dt>Sample index</dt><dd>{selected}</dd></div>
                <div><dt>Surface region</dt><dd>{data.regions[selected]} · {REGION_LABELS[data.regions[selected]]}</dd></div>
                <div><dt>Point type</dt><dd>{data.pointTypes[selected]} · {POINT_TYPE_LABELS[data.pointTypes[selected]]}</dd></div>
                <div><dt>Graph component</dt><dd>{data.components[selected]} · {COMPONENT_LABELS[data.components[selected]] || 'Connected group'}</dd></div>
                <div><dt>X</dt><dd>{data.points[selected * 3].toFixed(2)}</dd></div>
                <div><dt>Y</dt><dd>{data.points[selected * 3 + 1].toFixed(2)}</dd></div>
                <div><dt>Z</dt><dd>{data.points[selected * 3 + 2].toFixed(2)}</dd></div>
              </dl>
              <div className="potential-card"><span>Current signal value</span><strong>{currentValue.toFixed(4)}</strong><small>U<sub>heart</sub> at simulation time {metadata.time[frame].toFixed(1)}</small></div>
              {currentAnnotation && <div className={`annotation-chip ${currentAnnotation.label.toLowerCase()}`}><strong>{currentAnnotation.label}</strong><span>{currentAnnotation.note || 'No note'}</span><button onClick={() => setAnnotations((items) => items.filter((item) => item.pointIndex !== selected))}>Remove</button></div>}
            </>
          ) : <p className="empty-copy">Select a point in the 3D view.</p>}
        </aside>
      </section>

      <section className="signal-panel panel">
        <div className="signal-heading"><div><span className="eyebrow">Linked signal</span><h2>Electrical signal U<sub>heart</sub> over simulation time</h2></div><p>The red cursor is the time used for the 3D color map. Click or drag to move it.</p></div>
        <SignalPlot trace={selectedTrace} time={metadata.time} frame={frame} onFrameChange={setFrame} />
      </section>
    </main>
  )
}

export default App
