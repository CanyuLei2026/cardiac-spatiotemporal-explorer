const HISTORY_FRAMES = 40
const HORIZON_FRAMES = 20
const MIN_FRAME = HISTORY_FRAMES - 1

let stopped = false

function randomGenerator(seed = 7400) {
  let state = seed >>> 0
  return () => {
    state = (1664525 * state + 1013904223) >>> 0
    return state / 4294967296
  }
}

function activationFrame(signals, pointIndex, timeCount) {
  const start = pointIndex * timeCount
  let bestFrame = 1
  let bestRise = -Infinity
  for (let frame = 1; frame < timeCount; frame += 1) {
    const rise = signals[start + frame] - signals[start + frame - 1]
    if (rise > bestRise) {
      bestRise = rise
      bestFrame = frame
    }
  }
  return bestFrame
}

function selectPointIndices(pointCount, count, offset) {
  const indices = []
  const step = Math.max(1, Math.floor(pointCount / count))
  for (let index = offset; index < pointCount && indices.length < count; index += step) indices.push(index)
  return indices
}

function chooseNegativeFrame(activation, maximumFrame) {
  const before = activation - HORIZON_FRAMES - 35
  if (before >= MIN_FRAME) return before
  const after = activation + 35
  if (after <= maximumFrame) return after
  return maximumFrame
}

function makeExample(points, signals, pointIndex, frame, label, timeCount) {
  const values = new Float32Array(HISTORY_FRAMES + 4)
  const traceStart = pointIndex * timeCount
  for (let offset = 0; offset < HISTORY_FRAMES; offset += 1) {
    values[offset] = signals[traceStart + frame - HISTORY_FRAMES + 1 + offset]
  }
  values[HISTORY_FRAMES] = points[pointIndex * 3]
  values[HISTORY_FRAMES + 1] = points[pointIndex * 3 + 1]
  values[HISTORY_FRAMES + 2] = points[pointIndex * 3 + 2]
  values[HISTORY_FRAMES + 3] = frame / (timeCount - 1)
  return { values, label }
}

function buildExamples(points, signals, indices, timeCount) {
  const examples = []
  const maximumFrame = timeCount - HORIZON_FRAMES - 1
  for (const pointIndex of indices) {
    const activation = activationFrame(signals, pointIndex, timeCount)
    const positiveFrame = Math.max(MIN_FRAME, Math.min(maximumFrame, activation - 10))
    if (activation > positiveFrame && activation <= positiveFrame + HORIZON_FRAMES) {
      examples.push(makeExample(points, signals, pointIndex, positiveFrame, 1, timeCount))
    }
    const negativeFrame = chooseNegativeFrame(activation, maximumFrame)
    if (!(activation > negativeFrame && activation <= negativeFrame + HORIZON_FRAMES)) {
      examples.push(makeExample(points, signals, pointIndex, negativeFrame, 0, timeCount))
    }
  }
  return examples
}

function normalize(train, validation) {
  const width = train[0].values.length
  const mean = new Float32Array(width)
  const standardDeviation = new Float32Array(width)
  for (const example of train) {
    for (let feature = 0; feature < width; feature += 1) mean[feature] += example.values[feature]
  }
  for (let feature = 0; feature < width; feature += 1) mean[feature] /= train.length
  for (const example of train) {
    for (let feature = 0; feature < width; feature += 1) {
      const difference = example.values[feature] - mean[feature]
      standardDeviation[feature] += difference * difference
    }
  }
  for (let feature = 0; feature < width; feature += 1) {
    standardDeviation[feature] = Math.sqrt(standardDeviation[feature] / train.length) || 1
  }
  for (const example of [...train, ...validation]) {
    for (let feature = 0; feature < width; feature += 1) {
      example.values[feature] = (example.values[feature] - mean[feature]) / standardDeviation[feature]
    }
  }
}

function createParameters(inputSize, hiddenSize, random) {
  const w1 = new Float32Array(inputSize * hiddenSize)
  const b1 = new Float32Array(hiddenSize)
  const w2 = new Float32Array(hiddenSize * 2)
  const b2 = new Float32Array(2)
  const limit1 = Math.sqrt(6 / (inputSize + hiddenSize))
  const limit2 = Math.sqrt(6 / (hiddenSize + 2))
  for (let index = 0; index < w1.length; index += 1) w1[index] = (random() * 2 - 1) * limit1
  for (let index = 0; index < w2.length; index += 1) w2[index] = (random() * 2 - 1) * limit2
  return { w1, b1, w2, b2 }
}

function forward(parameters, input, hidden) {
  const { w1, b1, w2, b2 } = parameters
  for (let unit = 0; unit < hidden.length; unit += 1) {
    let sum = b1[unit]
    for (let feature = 0; feature < input.length; feature += 1) sum += input[feature] * w1[feature * hidden.length + unit]
    hidden[unit] = Math.max(0, sum)
  }
  let logit0 = b2[0]
  let logit1 = b2[1]
  for (let unit = 0; unit < hidden.length; unit += 1) {
    logit0 += hidden[unit] * w2[unit * 2]
    logit1 += hidden[unit] * w2[unit * 2 + 1]
  }
  const maximum = Math.max(logit0, logit1)
  const exp0 = Math.exp(logit0 - maximum)
  const exp1 = Math.exp(logit1 - maximum)
  const total = exp0 + exp1
  return [exp0 / total, exp1 / total]
}

function evaluate(parameters, examples, hiddenSize) {
  const hidden = new Float32Array(hiddenSize)
  let loss = 0
  let correct = 0
  for (const example of examples) {
    const probabilities = forward(parameters, example.values, hidden)
    loss -= Math.log(Math.max(probabilities[example.label], 1e-7))
    if ((probabilities[1] >= 0.5 ? 1 : 0) === example.label) correct += 1
  }
  return { loss: loss / examples.length, accuracy: correct / examples.length }
}

function adamUpdate(parameter, gradient, firstMoment, secondMoment, learningRate, step) {
  const beta1 = 0.9
  const beta2 = 0.999
  const firstCorrection = 1 - beta1 ** step
  const secondCorrection = 1 - beta2 ** step
  for (let index = 0; index < parameter.length; index += 1) {
    firstMoment[index] = beta1 * firstMoment[index] + (1 - beta1) * gradient[index]
    secondMoment[index] = beta2 * secondMoment[index] + (1 - beta2) * gradient[index] * gradient[index]
    const adjustedFirst = firstMoment[index] / firstCorrection
    const adjustedSecond = secondMoment[index] / secondCorrection
    parameter[index] -= learningRate * adjustedFirst / (Math.sqrt(adjustedSecond) + 1e-8)
  }
}

function shuffle(examples, random) {
  for (let index = examples.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1))
    ;[examples[index], examples[other]] = [examples[other], examples[index]]
  }
}

async function train({ pointsBuffer, signalsBuffer, pointCount, timeCount, learningRate, epochs, hiddenSize }) {
  stopped = false
  const points = new Float32Array(pointsBuffer)
  const signals = new Float32Array(signalsBuffer)
  const random = randomGenerator()
  const validationCount = Math.min(100, Math.max(30, Math.floor(pointCount * 0.12)))
  const trainingCount = Math.min(400, Math.max(100, Math.floor(pointCount * 0.4)))
  const validationIndices = selectPointIndices(pointCount, validationCount, 1)
  const validationSet = new Set(validationIndices)
  const trainingIndices = selectPointIndices(pointCount, trainingCount + validationCount, 0).filter((index) => !validationSet.has(index)).slice(0, trainingCount)
  const training = buildExamples(points, signals, trainingIndices, timeCount)
  const validation = buildExamples(points, signals, validationIndices, timeCount)
  normalize(training, validation)

  const parameters = createParameters(HISTORY_FRAMES + 4, hiddenSize, random)
  const parameterArrays = [parameters.w1, parameters.b1, parameters.w2, parameters.b2]
  const firstMoments = parameterArrays.map((values) => new Float32Array(values.length))
  const secondMoments = parameterArrays.map((values) => new Float32Array(values.length))
  const batchSize = 32
  let updateStep = 0
  let bestEpoch = 1
  let bestValidationLoss = Infinity

  postMessage({ type: 'ready', trainingSamples: training.length, validationSamples: validation.length })
  for (let epoch = 1; epoch <= epochs && !stopped; epoch += 1) {
    shuffle(training, random)
    for (let start = 0; start < training.length && !stopped; start += batchSize) {
      const end = Math.min(start + batchSize, training.length)
      const gradients = parameterArrays.map((values) => new Float32Array(values.length))
      const [gw1, gb1, gw2, gb2] = gradients
      const hidden = new Float32Array(hiddenSize)
      const hiddenGradient = new Float32Array(hiddenSize)
      for (let sample = start; sample < end; sample += 1) {
        const example = training[sample]
        const probabilities = forward(parameters, example.values, hidden)
        const d0 = probabilities[0] - (example.label === 0 ? 1 : 0)
        const d1 = probabilities[1] - (example.label === 1 ? 1 : 0)
        for (let unit = 0; unit < hiddenSize; unit += 1) {
          gw2[unit * 2] += hidden[unit] * d0
          gw2[unit * 2 + 1] += hidden[unit] * d1
          hiddenGradient[unit] = hidden[unit] > 0 ? parameters.w2[unit * 2] * d0 + parameters.w2[unit * 2 + 1] * d1 : 0
        }
        gb2[0] += d0
        gb2[1] += d1
        for (let feature = 0; feature < example.values.length; feature += 1) {
          for (let unit = 0; unit < hiddenSize; unit += 1) {
            gw1[feature * hiddenSize + unit] += example.values[feature] * hiddenGradient[unit]
          }
        }
        for (let unit = 0; unit < hiddenSize; unit += 1) gb1[unit] += hiddenGradient[unit]
      }
      const scale = 1 / (end - start)
      for (const gradient of gradients) for (let index = 0; index < gradient.length; index += 1) gradient[index] *= scale
      updateStep += 1
      for (let index = 0; index < parameterArrays.length; index += 1) {
        adamUpdate(parameterArrays[index], gradients[index], firstMoments[index], secondMoments[index], learningRate, updateStep)
      }
    }
    if (stopped) break
    const trainMetrics = evaluate(parameters, training, hiddenSize)
    const validationMetrics = evaluate(parameters, validation, hiddenSize)
    if (validationMetrics.loss < bestValidationLoss) {
      bestValidationLoss = validationMetrics.loss
      bestEpoch = epoch
    }
    postMessage({ type: 'epoch', epoch, train: trainMetrics, validation: validationMetrics, bestEpoch })
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
  postMessage({ type: stopped ? 'stopped' : 'complete', bestEpoch })
}

self.onmessage = (event) => {
  if (event.data.type === 'stop') {
    stopped = true
    return
  }
  if (event.data.type === 'train') train(event.data.payload).catch((error) => postMessage({ type: 'error', message: error.message }))
}
