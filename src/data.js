const CASE_ROOT = '/data/case_0004'

async function fetchBuffer(path) {
  const response = await fetch(`${CASE_ROOT}/${path}`)
  if (!response.ok) throw new Error(`Unable to load ${path}`)
  return response.arrayBuffer()
}

export async function loadCardiacCase() {
  const metadataResponse = await fetch(`${CASE_ROOT}/metadata.json`)
  if (!metadataResponse.ok) throw new Error('Unable to load metadata.json')
  const metadata = await metadataResponse.json()

  const [points, signals, components, pointTypes, regions, originalIndices] = await Promise.all([
    fetchBuffer('points.f32'),
    fetchBuffer('signals.f32'),
    fetchBuffer('components.i16'),
    fetchBuffer('point_types.i16'),
    fetchBuffer('regions.i16'),
    fetchBuffer('original_indices.u32'),
  ])

  const data = {
    metadata,
    points: new Float32Array(points),
    signals: new Float32Array(signals),
    components: new Int16Array(components),
    pointTypes: new Int16Array(pointTypes),
    regions: new Int16Array(regions),
    originalIndices: new Uint32Array(originalIndices),
  }

  const expectedPoints = metadata.pointCount
  if (data.points.length !== expectedPoints * 3) throw new Error('Point data size is inconsistent')
  if (data.signals.length !== expectedPoints * metadata.timeCount) throw new Error('Signal data size is inconsistent')
  return data
}

export function signalAt(data, pointIndex, frameIndex) {
  return data.signals[pointIndex * data.metadata.timeCount + frameIndex]
}

export function traceFor(data, pointIndex) {
  const start = pointIndex * data.metadata.timeCount
  return data.signals.subarray(start, start + data.metadata.timeCount)
}
