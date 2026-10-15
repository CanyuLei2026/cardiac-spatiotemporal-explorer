const MODEL_URL = `${import.meta.env.BASE_URL}model/activation_model.json`

export async function loadActivationModel() {
  const response = await fetch(MODEL_URL)
  if (!response.ok) throw new Error('Activation model could not be loaded')
  return response.json()
}

export function findActivationFrame(trace) {
  if (!trace || trace.length < 2) return null
  let bestFrame = 1
  let bestRise = -Infinity
  for (let frame = 1; frame < trace.length; frame += 1) {
    const rise = trace[frame] - trace[frame - 1]
    if (rise > bestRise) {
      bestRise = rise
      bestFrame = frame
    }
  }
  return bestFrame
}

function dense(values, layer) {
  return layer.bias.map((bias, outputIndex) => {
    const weights = layer.weight[outputIndex]
    let sum = bias
    for (let inputIndex = 0; inputIndex < values.length; inputIndex += 1) {
      sum += values[inputIndex] * weights[inputIndex]
    }
    return layer.activation === 'relu' ? Math.max(0, sum) : sum
  })
}

function softmax(logits) {
  const maximum = Math.max(...logits)
  const exponentials = logits.map((value) => Math.exp(value - maximum))
  const total = exponentials.reduce((sum, value) => sum + value, 0)
  return exponentials.map((value) => value / total)
}

export function predictActivation(model, data, pointIndex, frame) {
  const { historyFrames, horizonFrames, featureMean, featureStd } = model.input
  if (frame < historyFrames - 1) throw new Error(`At least ${historyFrames} observed frames are required`)
  if (frame + horizonFrames >= data.metadata.timeCount) throw new Error(`At least ${horizonFrames} future frames are required for evaluation`)

  const traceStart = pointIndex * data.metadata.timeCount
  const historyStart = traceStart + frame - historyFrames + 1
  const features = []
  for (let offset = 0; offset < historyFrames; offset += 1) {
    features.push(data.signals[historyStart + offset])
  }
  features.push(
    data.points[pointIndex * 3],
    data.points[pointIndex * 3 + 1],
    data.points[pointIndex * 3 + 2],
    frame / (data.metadata.timeCount - 1),
  )
  const normalized = features.map((value, index) => (value - featureMean[index]) / featureStd[index])
  const logits = model.network.layers.reduce((values, layer) => dense(values, layer), normalized)
  const probabilities = softmax(logits)
  const predicted = probabilities[1] >= 0.5
  const trace = data.signals.subarray(traceStart, traceStart + data.metadata.timeCount)
  const activationFrame = findActivationFrame(trace)
  const actual = activationFrame > frame && activationFrame <= frame + horizonFrames
  return {
    predicted,
    confidence: probabilities[predicted ? 1 : 0],
    probabilityYes: probabilities[1],
    actual,
    correct: predicted === actual,
    activationFrame,
    historyStart: frame - historyFrames + 1,
    horizonEnd: frame + horizonFrames,
  }
}
