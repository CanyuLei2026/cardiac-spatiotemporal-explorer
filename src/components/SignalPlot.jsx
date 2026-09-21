import { useMemo, useRef } from 'react'

const WIDTH = 1000
const HEIGHT = 230
const PAD = { left: 58, right: 24, top: 18, bottom: 38 }

export default function SignalPlot({ trace, time, frame, onFrameChange }) {
  const svgRef = useRef()
  const geometry = useMemo(() => {
    if (!trace?.length) return null
    let min = Infinity
    let max = -Infinity
    for (const value of trace) {
      min = Math.min(min, value)
      max = Math.max(max, value)
    }
    const ySpan = Math.max(max - min, 1e-6)
    const plotWidth = WIDTH - PAD.left - PAD.right
    const plotHeight = HEIGHT - PAD.top - PAD.bottom
    const points = Array.from(trace, (value, index) => {
      const x = PAD.left + (index / (trace.length - 1)) * plotWidth
      const y = PAD.top + (1 - (value - min) / ySpan) * plotHeight
      return `${x.toFixed(2)},${y.toFixed(2)}`
    }).join(' ')
    return { min, max, points, plotWidth, plotHeight }
  }, [trace])

  if (!geometry) return <div className="signal-empty">Select a point to inspect its signal.</div>

  const cursorX = PAD.left + (frame / (trace.length - 1)) * geometry.plotWidth
  const handlePointer = (event) => {
    const rect = svgRef.current.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * WIDTH
    const ratio = Math.max(0, Math.min(1, (x - PAD.left) / geometry.plotWidth))
    onFrameChange(Math.round(ratio * (trace.length - 1)))
  }

  return (
    <svg
      ref={svgRef}
      className="signal-svg"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label="Electrical signal for the selected heart point"
      onPointerDown={handlePointer}
      onPointerMove={(event) => event.buttons === 1 && handlePointer(event)}
    >
      {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const y = PAD.top + ratio * geometry.plotHeight
        return <line key={ratio} x1={PAD.left} x2={WIDTH - PAD.right} y1={y} y2={y} className="chart-grid" />
      })}
      <polyline points={geometry.points} className="signal-line" />
      <line x1={cursorX} x2={cursorX} y1={PAD.top} y2={HEIGHT - PAD.bottom} className="time-cursor" />
      <circle
        cx={cursorX}
        cy={PAD.top + (1 - (trace[frame] - geometry.min) / Math.max(geometry.max - geometry.min, 1e-6)) * geometry.plotHeight}
        r="6"
        className="cursor-dot"
      />
      <text x="10" y={PAD.top + 6} className="axis-label">{geometry.max.toFixed(3)}</text>
      <text x="10" y={HEIGHT - PAD.bottom + 5} className="axis-label">{geometry.min.toFixed(3)}</text>
      <text x={PAD.left} y={HEIGHT - 10} className="axis-label">{time[0].toFixed(1)}</text>
      <text x={WIDTH - PAD.right} y={HEIGHT - 10} textAnchor="end" className="axis-label">{time[time.length - 1].toFixed(1)}</text>
      <text x={WIDTH / 2} y={HEIGHT - 10} textAnchor="middle" className="axis-title">Simulation time</text>
    </svg>
  )
}
