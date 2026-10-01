import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

const PYTHON_CODE_NEURON = `import numpy as np

# ── Single linear neuron for regression ──
# Dataset: study hours → exam score
X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([15, 28, 42, 55, 64, 74, 81, 92], dtype=float)

# Normalise inputs (important for gradient descent)
X_norm = (X - X.mean()) / X.std()

# Parameters
w = 0.0   # weight
b = 0.0   # bias
lr = 0.1  # learning rate
n = len(X_norm)

losses = []
for epoch in range(200):
    # Forward pass: linear prediction
    y_pred = w * X_norm + b

    # MSE loss
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)

    # Gradients (∂L/∂w and ∂L/∂b)
    dw = (2 / n) * np.sum((y_pred - y) * X_norm)
    db = (2 / n) * np.sum(y_pred - y)

    # Gradient descent update
    w -= lr * dw
    b -= lr * db

print(f"Final w    : {w:.4f}")
print(f"Final b    : {b:.4f}")
print(f"Final MSE  : {losses[-1]:.4f}")
print(f"Predict 5h : {w * ((5 - X.mean())/X.std()) + b:.1f}")`

const PYTHON_CODE_TF = `import tensorflow as tf
import numpy as np

# Dataset: study hours → exam score
X = np.array([1,2,3,4,5,6,7,8], dtype=np.float32)
y = np.array([15,28,42,55,64,74,81,92], dtype=np.float32)

# Normalise inputs
X_norm = (X - X.mean()) / X.std()

# Single linear neuron — Dense(1) with no activation = linear regression
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=(1,))   # no activation
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss='mse'
)

history = model.fit(X_norm, y, epochs=200, verbose=0)

w, b = model.layers[0].get_weights()
print(f"Weight : {w[0,0]:.4f}")
print(f"Bias   : {b[0]:.4f}")
print(f"Final MSE: {history.history['loss'][-1]:.4f}")

# Inference
x_new = np.array([[(5 - X.mean()) / X.std()]])
pred  = model.predict(x_new, verbose=0)
print(f"Predict 5h: {pred[0,0]:.1f}")`

const QUIZ_QUESTIONS = [
  { question: 'A regression task predicts:', options: ['A category label', 'A continuous numerical value', 'A probability between 0 and 1', 'A binary 0/1'], correct: 1, explanation: 'Regression predicts a continuous quantity — "how much" or "how many" — like price, temperature, or exam score.' },
  { question: 'A single output neuron with no hidden layers and no activation function computes:', options: ['A non-linear mapping', 'ŷ = wx + b  (linear regression)', 'A probability via sigmoid', 'A class via softmax'], correct: 1, explanation: 'Without an activation function, the neuron output is simply ŷ = wx + b — identical to classical linear regression.' },
  { question: 'MSE loss is defined as:', options: ['(1/n) Σ|yᵢ − ŷᵢ|', '(1/n) Σ(yᵢ − ŷᵢ)²', 'Σ log(yᵢ/ŷᵢ)', 'max|yᵢ − ŷᵢ|'], correct: 1, explanation: 'MSE = (1/n)·Σ(yᵢ − ŷᵢ)². Squaring penalises large errors more heavily and gives a smooth, differentiable loss surface.' },
  { question: 'The gradient of MSE w.r.t. weight w is:', options: ['(2/n) Σ(ŷᵢ − yᵢ)', '(2/n) Σ(ŷᵢ − yᵢ)·xᵢ', '−(1/n) Σ xᵢ yᵢ', 'Σ(ŷᵢ² − yᵢ²)'], correct: 1, explanation: '∂MSE/∂w = (2/n)·Σ(ŷᵢ − yᵢ)·xᵢ. Residuals scaled by inputs — intuitively, features that matter more drive larger weight updates.' },
  { question: 'In batch gradient descent, weights are updated:', options: ['After each single sample', 'After seeing all n training samples', 'Randomly every iteration', 'Only when loss increases'], correct: 1, explanation: 'Batch GD computes the average gradient over the full training set, then takes one weight update step per epoch.' },
  { question: 'If the learning rate is too large:', options: ['Training is slow but accurate', 'Loss may oscillate or diverge', 'The model underfits', 'Gradients become zero'], correct: 1, explanation: 'A large lr causes the gradient update to overshoot the minimum. The loss bounces back and forth or grows — the model fails to converge.' },
  { question: 'The loss curve should look like:', options: ['Increasing then flat', 'Flat throughout training', 'Decreasing, levelling off at a minimum', 'Oscillating around zero'], correct: 2, explanation: 'A healthy loss curve falls steeply at first as the gradient is large far from the minimum, then flattens as it approaches convergence.' },
  { question: 'During inference (prediction), we:', options: ['Update weights once more', 'Run only the forward pass with learned w and b', 'Reset weights to zero', 'Re-train on the query'], correct: 1, explanation: 'Inference = forward pass only. Weights are frozen. Input x is fed through ŷ = wx + b to produce a prediction. No gradients computed.' },
  { question: 'Why normalise inputs before training?', options: ['To add noise', 'So gradient steps are balanced across features of different scales', 'To increase MSE', 'Normalisation is optional for linear models'], correct: 1, explanation: 'Without normalisation, large-scale features produce huge gradients while small-scale ones produce tiny ones. Training becomes slow and unstable.' },
  { question: 'A linear neuron for regression differs from a Perceptron because:', options: ['It has more inputs', 'It has no activation function — output is the raw linear combination', 'It uses cross-entropy loss', 'It has multiple outputs'], correct: 1, explanation: 'A Perceptron applies a step (or sigmoid) activation. A linear regression neuron has no activation — output = wx + b directly. This allows it to predict any real number.' },
]

// ── Neuron diagram ─────────────────────────────────────────────────────────────
function NeuronDiagram({ theme }) {
  const [w1, setW1] = useState(0.6)
  const [w2, setW2] = useState(0.4)
  const [w3, setW3] = useState(0.3)
  const [b, setB] = useState(0.5)
  const inputs = [1.2, 0.8, 1.5]
  const weights = [w1, w2, w3]
  const z = inputs.reduce((s, x, i) => s + x * weights[i], b)
  const W = 380, H = 220
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)
    const inputXs = [40, 40, 40]
    const inputYs = [55, 110, 165]
    const neuronX = 210, neuronY = 110, neuronR = 32
    const outX = 340, outY = 110

    const wColors = ['#6366F1', '#8B5CF6', '#06B6D4']

    // Draw input → neuron connections
    inputYs.forEach((iy, i) => {
      const w = weights[i]
      ctx.beginPath()
      ctx.moveTo(inputXs[i] + 14, iy)
      ctx.lineTo(neuronX - neuronR, neuronY)
      ctx.strokeStyle = wColors[i]
      ctx.lineWidth = 1.5 + Math.abs(w) * 2
      ctx.globalAlpha = 0.5 + Math.abs(w) * 0.5
      ctx.stroke()
      ctx.globalAlpha = 1

      // Weight label on line
      const mx = (inputXs[i] + 14 + neuronX - neuronR) / 2
      const my = (iy + neuronY) / 2 - 6
      ctx.fillStyle = wColors[i]
      ctx.font = 'bold 10px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(`w${i + 1}=${w.toFixed(1)}`, mx, my)
    })

    // Input circles
    inputYs.forEach((iy, i) => {
      ctx.beginPath(); ctx.arc(inputXs[i], iy, 14, 0, Math.PI * 2)
      ctx.fillStyle = theme === 'dark' ? '#1E293B' : '#F1F5F9'
      ctx.fill(); ctx.strokeStyle = wColors[i]; ctx.lineWidth = 2; ctx.stroke()
      ctx.fillStyle = wColors[i]; ctx.font = 'bold 11px monospace'; ctx.textAlign = 'center'
      ctx.fillText(`x${i + 1}`, inputXs[i], iy + 4)
    })

    // Neuron circle
    const grad = ctx.createRadialGradient(neuronX, neuronY, 4, neuronX, neuronY, neuronR)
    grad.addColorStop(0, theme === 'dark' ? '#4F46E5' : '#818CF8')
    grad.addColorStop(1, theme === 'dark' ? '#312E81' : '#6366F1')
    ctx.beginPath(); ctx.arc(neuronX, neuronY, neuronR, 0, Math.PI * 2)
    ctx.fillStyle = grad; ctx.fill()
    ctx.strokeStyle = theme === 'dark' ? '#A5B4FC' : 'white'; ctx.lineWidth = 2; ctx.stroke()

    // Bias label inside neuron
    ctx.fillStyle = 'white'; ctx.font = 'bold 10px monospace'; ctx.textAlign = 'center'
    ctx.fillText('Σ', neuronX, neuronY - 5)
    ctx.font = '9px monospace'
    ctx.fillText(`+b=${b.toFixed(1)}`, neuronX, neuronY + 8)

    // Output arrow
    ctx.beginPath(); ctx.moveTo(neuronX + neuronR, neuronY); ctx.lineTo(outX - 14, outY)
    ctx.strokeStyle = '#10B981'; ctx.lineWidth = 2.5; ctx.stroke()
    // arrowhead
    ctx.beginPath(); ctx.moveTo(outX - 14, outY - 5); ctx.lineTo(outX - 14, outY + 5); ctx.lineTo(outX, outY); ctx.closePath()
    ctx.fillStyle = '#10B981'; ctx.fill()
    // z label on arrow
    ctx.fillStyle = '#10B981'; ctx.font = 'bold 10px monospace'; ctx.textAlign = 'center'
    ctx.fillText(`ŷ = z`, (neuronX + neuronR + outX - 14) / 2, neuronY - 8)

    // Output node
    ctx.beginPath(); ctx.arc(outX, outY, 20, 0, Math.PI * 2)
    ctx.fillStyle = theme === 'dark' ? '#065F46' : '#D1FAE5'
    ctx.fill(); ctx.strokeStyle = '#10B981'; ctx.lineWidth = 2.5; ctx.stroke()
    ctx.fillStyle = '#10B981'; ctx.font = 'bold 11px monospace'; ctx.textAlign = 'center'
    ctx.fillText(z.toFixed(2), outX, outY + 4)

    // No activation label
    ctx.fillStyle = theme === 'dark' ? '#4B5563' : '#9CA3AF'
    ctx.font = '9px sans-serif'; ctx.textAlign = 'center'
    ctx.fillText('no activation', neuronX, neuronY + neuronR + 14)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [w1, w2, w3, b, theme])

  const sliders = [
    { label: 'w₁', val: w1, set: setW1, color: '#6366F1' },
    { label: 'w₂', val: w2, set: setW2, color: '#8B5CF6' },
    { label: 'w₃', val: w3, set: setW3, color: '#06B6D4' },
    { label: 'b', val: b, set: setB, color: '#10B981' },
  ]

  return (
    <div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border mb-4 ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
      <div className="grid grid-cols-2 gap-3 mb-3">
        {sliders.map(s => (
          <div key={s.label}>
            <label className={`text-xs font-medium block mb-1 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
              {s.label} = <span className="font-bold font-mono" style={{ color: s.color }}>{s.val.toFixed(1)}</span>
            </label>
            <input type="range" min="-2" max="2" step="0.1" value={s.val}
              onChange={e => s.set(parseFloat(e.target.value))}
              className="w-full" style={{ accentColor: s.color }} />
          </div>
        ))}
      </div>
      <div className={`rounded-xl p-3 border font-mono text-xs ${theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-300' : 'bg-gray-100 border-gray-200 text-gray-700'}`}>
        ŷ = {inputs.map((x, i) => `${x}×${weights[i].toFixed(1)}`).join(' + ')} + {b.toFixed(1)} = <span className="text-emerald-400 font-bold">{z.toFixed(4)}</span>
      </div>
    </div>
  )
}

// ── MSE Loss surface ────────────────────────────────────────────────────────────
function LossSurface({ theme }) {
  const canvasRef = useRef(null)
  const [w, setW] = useState(-2.0)
  const [b, setBVal] = useState(-2.0)
  const W = 380, H = 240

  // Toy dataset: y = 2x + 1
  const data = [
    [0.5, 2.0], [1.0, 3.1], [1.5, 3.9], [2.0, 5.2],
    [2.5, 6.0], [3.0, 7.1], [3.5, 7.8], [4.0, 9.1],
  ]

  const mse = (wv, bv) => {
    const n = data.length
    return data.reduce((s, [x, y]) => s + (wv * x + bv - y) ** 2, 0) / n
  }

  // Analytical minimum: w*=2, b*=1 → MSE≈0
  const loss = mse(w, b)
  const optLoss = mse(2, 1)

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)
    const pad = 35
    const wRange = [-4, 6], bRange = [-4, 5]
    const toX = wv => pad + ((wv - wRange[0]) / (wRange[1] - wRange[0])) * (W - 2 * pad)
    const toY = bv => (H - pad) - ((bv - bRange[0]) / (bRange[1] - bRange[0])) * (H - 2 * pad)

    // Loss contours (filled iso-bands)
    const step = 4
    for (let px = pad; px < W - pad; px += step) {
      for (let py = pad; py < H - pad; py += step) {
        const wv = wRange[0] + ((px - pad) / (W - 2 * pad)) * (wRange[1] - wRange[0])
        const bv = bRange[0] + ((H - pad - py) / (H - 2 * pad)) * (bRange[1] - bRange[0])
        const l = Math.min(mse(wv, bv), 60)
        const t = l / 60
        const r = Math.round(99 * t + 99 * (1 - t))
        const g = Math.round(102 * (1 - t))
        const bl = Math.round(241 * (1 - t))
        ctx.fillStyle = `rgba(${r},${g},${bl},${0.35 + 0.15 * (1 - t)})`
        ctx.fillRect(px, py, step, step)
      }
    }

    // Axes
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); ctx.stroke()

    // Optimal point
    ctx.beginPath(); ctx.arc(toX(2), toY(1), 7, 0, Math.PI * 2)
    ctx.fillStyle = '#10B981'; ctx.fill(); ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.stroke()
    ctx.fillStyle = '#10B981'; ctx.font = '9px sans-serif'; ctx.textAlign = 'center'
    ctx.fillText('min', toX(2), toY(1) - 11)

    // Current point
    ctx.beginPath(); ctx.arc(toX(w), toY(b), 8, 0, Math.PI * 2)
    ctx.fillStyle = '#F59E0B'; ctx.fill(); ctx.strokeStyle = 'white'; ctx.lineWidth = 2.5; ctx.stroke()

    // Axis labels
    ctx.fillStyle = theme === 'dark' ? '#6B7280' : '#9CA3AF'
    ctx.font = '10px monospace'; ctx.textAlign = 'center'
    for (const wv of [-4, -2, 0, 2, 4, 6]) ctx.fillText(wv, toX(wv), H - pad + 13)
    ctx.textAlign = 'right'
    for (const bv of [-4, -2, 0, 2, 4]) ctx.fillText(bv, pad - 5, toY(bv) + 3)
    ctx.textAlign = 'center'
    ctx.fillStyle = theme === 'dark' ? '#9CA3AF' : '#6B7280'
    ctx.fillText('w (weight)', W / 2, H - 4)
    ctx.save(); ctx.translate(10, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('b (bias)', 0, 0); ctx.restore()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [w, b, theme])

  return (
    <div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            w = <span className="font-bold text-yellow-400">{w.toFixed(1)}</span>
          </label>
          <input type="range" min="-4" max="6" step="0.1" value={w} onChange={e => setW(+e.target.value)} className="w-full accent-yellow-500" />
        </div>
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            b = <span className="font-bold text-yellow-400">{b.toFixed(1)}</span>
          </label>
          <input type="range" min="-4" max="5" step="0.1" value={b} onChange={e => setBVal(+e.target.value)} className="w-full accent-yellow-500" />
        </div>
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
      <div className="flex gap-3 mt-3 text-xs flex-wrap">
        <div className={`flex-1 px-3 py-2 rounded-lg ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-100'}`}>
          <span className={theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}>Current MSE: </span>
          <span className="text-yellow-400 font-bold font-mono">{loss.toFixed(4)}</span>
        </div>
        <div className={`flex-1 px-3 py-2 rounded-lg ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-100'}`}>
          <span className={theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}>Min MSE: </span>
          <span className="text-emerald-400 font-bold font-mono">{optLoss.toFixed(4)}</span>
          <span className={`text-xs ml-1 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>(w=2, b=1)</span>
        </div>
      </div>
    </div>
  )
}

// ── Gradient Descent trainer ───────────────────────────────────────────────────
function GDTrainer({ theme }) {
  const [lr, setLr] = useState(0.15)
  const [running, setRunning] = useState(false)
  const [epoch, setEpoch] = useState(0)
  const [w, setW] = useState(-2.5)
  const [b, setBVal] = useState(-1.5)
  const [lossHistory, setLossHistory] = useState([])
  const rafRef = useRef(null)
  const stateRef = useRef({ w: -2.5, b: -1.5, lr: 0.15, losses: [], ep: 0 })

  // Dataset: y = 2x + 1 with noise
  const data = [
    [0.5, 2.0], [1.0, 3.1], [1.5, 3.9], [2.0, 5.2],
    [2.5, 6.0], [3.0, 7.1], [3.5, 7.8], [4.0, 9.1],
  ]
  const n = data.length

  function runStep() {
    const s = stateRef.current
    if (s.ep >= 120) { setRunning(false); return }
    // Forward
    const preds = data.map(([x]) => s.w * x + s.b)
    const residuals = preds.map((p, i) => p - data[i][1])
    const loss = residuals.reduce((acc, r) => acc + r * r, 0) / n
    // Gradients
    const dw = (2 / n) * residuals.reduce((acc, r, i) => acc + r * data[i][0], 0)
    const db = (2 / n) * residuals.reduce((acc, r) => acc + r, 0)
    // Update
    s.w -= s.lr * dw
    s.b -= s.lr * db
    s.losses = [...s.losses, loss]
    s.ep += 1
    setW(s.w); setBVal(s.b); setEpoch(s.ep); setLossHistory([...s.losses])
    rafRef.current = requestAnimationFrame(runStep)
  }

  const start = () => {
    stateRef.current = { w: -2.5, b: -1.5, lr, losses: [], ep: 0 }
    setW(-2.5); setBVal(-1.5); setLossHistory([]); setEpoch(0); setRunning(true)
    rafRef.current = requestAnimationFrame(runStep)
  }
  const reset = () => {
    cancelAnimationFrame(rafRef.current)
    stateRef.current = { w: -2.5, b: -1.5, lr, losses: [], ep: 0 }
    setW(-2.5); setBVal(-1.5); setLossHistory([]); setEpoch(0); setRunning(false)
  }
  useEffect(() => () => cancelAnimationFrame(rafRef.current), [])

  const canvasRef = useRef(null)
  const W = 380, H = 200

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)
    const pad = { l: 45, r: 15, t: 15, b: 30 }
    const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b

    // Axes
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke()

    if (lossHistory.length === 0) {
      ctx.fillStyle = theme === 'dark' ? '#374151' : '#D1D5DB'
      ctx.font = '12px sans-serif'; ctx.textAlign = 'center'
      ctx.fillText('Press Run to start training', W / 2, H / 2)
      return
    }

    const maxL = Math.max(...lossHistory, 0.01)
    const toX = i => pad.l + (i / 119) * plotW
    const toY = l => (H - pad.b) - (Math.min(l, maxL) / maxL) * plotH

    // Grid
    ctx.setLineDash([3, 3]); ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'
    for (let v = 1; v <= 4; v++) {
      const gy = (H - pad.b) - (v / 4) * plotH
      ctx.beginPath(); ctx.moveTo(pad.l, gy); ctx.lineTo(W - pad.r, gy); ctx.stroke()
    }
    ctx.setLineDash([])

    // Loss curve
    const grad = ctx.createLinearGradient(pad.l, 0, W - pad.r, 0)
    grad.addColorStop(0, '#EF4444'); grad.addColorStop(0.6, '#F59E0B'); grad.addColorStop(1, '#10B981')
    ctx.beginPath()
    lossHistory.forEach((l, i) => {
      i === 0 ? ctx.moveTo(toX(i), toY(l)) : ctx.lineTo(toX(i), toY(l))
    })
    ctx.strokeStyle = grad; ctx.lineWidth = 2.5; ctx.stroke()

    // Current dot
    const last = lossHistory.length - 1
    ctx.beginPath(); ctx.arc(toX(last), toY(lossHistory[last]), 5, 0, Math.PI * 2)
    ctx.fillStyle = '#10B981'; ctx.fill()

    // Axis labels
    ctx.fillStyle = theme === 'dark' ? '#6B7280' : '#9CA3AF'
    ctx.font = '9px monospace'; ctx.textAlign = 'right'
    for (let v = 0; v <= 4; v++) {
      const l = (maxL * v / 4).toFixed(1)
      ctx.fillText(l, pad.l - 4, toY(maxL * v / 4) + 3)
    }
    ctx.textAlign = 'center'
    for (const ep of [0, 30, 60, 90, 119]) ctx.fillText(ep, toX(ep), H - pad.b + 12)
    ctx.fillText('Epoch', W / 2, H - 2)
    ctx.save(); ctx.translate(10, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('MSE Loss', 0, 0); ctx.restore()
  }, [lossHistory, theme])

  // Scatter + line canvas
  const scatterRef = useRef(null)
  useEffect(() => {
    const canvas = scatterRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)
    const pad = { l: 35, r: 15, t: 15, b: 30 }
    const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b
    const toX = v => pad.l + ((v - 0) / 5) * plotW
    const toY = v => (H - pad.b) - ((v - 0) / 12) * plotH

    // Grid
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.05)'
    ctx.lineWidth = 1; ctx.setLineDash([3, 3])
    for (let v = 0; v <= 5; v++) { ctx.beginPath(); ctx.moveTo(toX(v), pad.t); ctx.lineTo(toX(v), H - pad.b); ctx.stroke() }
    for (let v = 0; v <= 12; v += 3) { ctx.beginPath(); ctx.moveTo(pad.l, toY(v)); ctx.lineTo(W - pad.r, toY(v)); ctx.stroke() }
    ctx.setLineDash([])

    // Axes
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)'; ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke()

    // Ideal line y = 2x + 1
    ctx.beginPath(); ctx.moveTo(toX(0), toY(1)); ctx.lineTo(toX(5), toY(11))
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.08)'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([])

    // Current model line
    ctx.beginPath(); ctx.moveTo(toX(0), toY(b)); ctx.lineTo(toX(5), toY(w * 5 + b))
    ctx.strokeStyle = '#6366F1'; ctx.lineWidth = 2.5; ctx.stroke()

    // Data points
    data.forEach(([x, y]) => {
      ctx.beginPath(); ctx.arc(toX(x), toY(y), 5, 0, Math.PI * 2)
      ctx.fillStyle = '#10B981'; ctx.fill(); ctx.strokeStyle = 'white'; ctx.lineWidth = 1.5; ctx.stroke()
    })

    // Labels
    ctx.fillStyle = theme === 'dark' ? '#6B7280' : '#9CA3AF'
    ctx.font = '9px monospace'; ctx.textAlign = 'center'
    for (const v of [0, 1, 2, 3, 4, 5]) ctx.fillText(v, toX(v), H - pad.b + 12)
    ctx.textAlign = 'right'
    for (const v of [0, 3, 6, 9, 12]) ctx.fillText(v, pad.l - 4, toY(v) + 3)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [w, b, theme])

  const currentLoss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].toFixed(4) : '—'

  return (
    <div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            Learning rate α = <span className="text-yellow-400 font-bold">{lr.toFixed(2)}</span>
          </label>
          <input type="range" min="0.01" max="0.5" step="0.01" value={lr}
            onChange={e => { setLr(+e.target.value); stateRef.current.lr = +e.target.value }}
            className="w-full accent-yellow-500" disabled={running} />
          <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>
            {lr > 0.35 ? '⚠️ Too large — may diverge' : lr < 0.05 ? 'Very slow convergence' : 'Good rate'}
          </p>
        </div>
        <div className="flex flex-col justify-end gap-2">
          <button onClick={running ? reset : start}
            className={`w-full py-2 rounded-xl text-sm font-semibold transition-all border ${running
              ? 'bg-red-500/20 border-red-500/40 text-red-400 hover:bg-red-500/30'
              : 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300 hover:bg-indigo-500/30'}`}>
            {running ? '⏹ Stop' : epoch > 0 ? '🔄 Reset & Run' : '▶ Run Training'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 mb-4 text-xs">
        {[
          { label: 'Epoch', val: epoch, color: 'text-indigo-400' },
          { label: 'MSE', val: currentLoss, color: 'text-red-400' },
          { label: 'w', val: w.toFixed(3), color: 'text-purple-400' },
          { label: 'b', val: b.toFixed(3), color: 'text-cyan-400' },
        ].map(m => (
          <div key={m.label} className={`rounded-xl p-2 border text-center ${theme === 'dark' ? 'bg-slate-800 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className={`font-mono font-bold text-sm ${m.color}`}>{m.val}</p>
            <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{m.label}</p>
          </div>
        ))}
      </div>

      <p className={`text-xs font-semibold uppercase tracking-wider mb-2 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>Loss Curve</p>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border mb-4 ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />

      <p className={`text-xs font-semibold uppercase tracking-wider mb-2 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>Fitted Line (purple) vs Data</p>
      <canvas ref={scatterRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
    </div>
  )
}

// ── Inference demo ─────────────────────────────────────────────────────────────
function InferenceDemo({ theme }) {
  // Weights learned from above dataset (y ≈ 2x + 1)
  const w = 1.97, b = 1.08
  const [x, setX] = useState(3.5)
  const pred = w * x + b

  return (
    <div>
      <p className={`text-xs mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
        Training is done. Weights are frozen: <span className="font-mono text-purple-400">w = {w}</span>, <span className="font-mono text-cyan-400">b = {b}</span>. Slide x to run inference.
      </p>
      <div className="mb-5">
        <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
          Input x = <span className="text-indigo-400 font-bold">{x.toFixed(1)}</span>
        </label>
        <input type="range" min="0" max="10" step="0.1" value={x} onChange={e => setX(+e.target.value)} className="w-full accent-indigo-500" />
      </div>
      <div className={`rounded-xl border p-4 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
        <div className="flex items-center gap-3 flex-wrap">
          {[
            { label: 'Input x', val: x.toFixed(1), color: 'text-indigo-400', bg: theme === 'dark' ? 'bg-indigo-500/10' : 'bg-indigo-50' },
            { label: '→', val: null },
            { label: 'w × x', val: `${w} × ${x.toFixed(1)} = ${(w * x).toFixed(3)}`, color: 'text-purple-400', bg: theme === 'dark' ? 'bg-purple-500/10' : 'bg-purple-50' },
            { label: '+', val: null },
            { label: 'b', val: b.toString(), color: 'text-cyan-400', bg: theme === 'dark' ? 'bg-cyan-500/10' : 'bg-cyan-50' },
            { label: '=', val: null },
            { label: 'ŷ', val: pred.toFixed(3), color: 'text-emerald-400', bg: theme === 'dark' ? 'bg-emerald-500/10' : 'bg-emerald-50' },
          ].map((item, i) => item.val === null ? (
            <span key={i} className={`text-lg font-bold ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{item.label}</span>
          ) : (
            <div key={i} className={`rounded-xl px-3 py-2 border text-center ${item.bg} ${theme === 'dark' ? 'border-white/10' : 'border-gray-200'}`}>
              <p className={`text-xs mb-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{item.label}</p>
              <p className={`font-mono font-bold text-sm ${item.color}`}>{item.val}</p>
            </div>
          ))}
        </div>
      </div>
      <Callout type="success" title="No gradients — just arithmetic" className="mt-4">
        During inference, only the forward pass runs: multiply input by weight, add bias, done. No loss computation, no backward pass, no weight updates. The model is read-only.
      </Callout>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function LinearNeuronRegression() {
  const { theme } = useApp()
  const S = `rounded-2xl border p-6 mb-6 ${theme === 'dark' ? 'bg-slate-900/80 border-white/10' : 'bg-white border-gray-200'}`
  const LBL = `text-xs font-semibold uppercase tracking-wider mb-3 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`
  const H2 = `text-xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`
  const BODY = `text-sm leading-relaxed ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-3xl">

      {/* Hero */}
      <div className="mb-8">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/20 border border-indigo-500/30 mb-4">
          <span className="text-xs text-indigo-400 font-medium">Deep Neural Networks • Session 3</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Linear Neural Networks <span className="gradient-text">for Regression</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          The simplest neural network: a <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>single neuron with no activation function</strong>. The foundation of every deep network — and identical to linear regression.
        </p>
        <Callout type="analogy" title="Why start here?">
          Before stacking layers and adding activations, understand the atomic unit. A single linear neuron teaches you the full forward pass → loss → gradient → update cycle. Every deep network is just this loop, repeated millions of times across many neurons.
        </Callout>
      </div>

      <TheoryBlock title="Core Concepts" cards={[
        { icon: '📏', title: 'Linear Neuron', body: 'One weight per input, one bias. No activation function — output is the raw weighted sum. Expressive enough for any linear relationship.', mono: 'ŷ = w₁x₁ + w₂x₂ + ... + b' },
        { icon: '📉', title: 'MSE Loss', body: 'Mean squared error penalises the square of each residual. Smooth and convex for linear models — guarantees a unique global minimum.', mono: 'L = (1/n) Σ (yᵢ − ŷᵢ)²' },
        { icon: '⬇️', title: 'Gradient Descent', body: 'Iteratively nudge weights in the direction that reduces loss. For linear models, the loss surface is a perfect bowl — gradient always points to the minimum.', mono: 'w ← w − α · ∂L/∂w' },
        { icon: '🎚️', title: 'Learning Rate', body: 'Controls step size. Too large → overshoots minimum. Too small → very slow convergence. Typical range: 0.001–0.5 for normalised inputs.', mono: 'α ∈ [0.001, 0.5]' },
        { icon: '📈', title: 'Loss Curve', body: 'A healthy loss curve falls steeply then flattens. Oscillation signals too-large lr. No decrease at all signals too-small lr or a bug in gradients.', mono: 'loss[t] < loss[t−1] (ideally)' },
        { icon: '🔍', title: 'Inference', body: 'After training, freeze weights and run the forward pass only. No backward pass, no loss, no updates. Prediction = one multiplication and one addition.', mono: 'ŷ = wx + b  (frozen w, b)' },
      ]} />

      {/* ── Section 3.1 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 3.1</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Single Neuron for Regression</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>What a regression task is, how a linear neuron models it, and the MSE loss that drives learning.</p>
      </div>

      {/* 3.1.1 Regression task */}
      <div className={S}>
        <p className={LBL}>3.1.1 — Regression: "How Much / How Many?"</p>
        <h2 className={H2}>Predicting continuous quantities</h2>
        <p className={`${BODY} mb-4`}>
          Regression tasks ask for a real-valued output — not a class label. The model must learn a mapping from input features to a continuous target variable. The answer is a number on a spectrum, not a category.
        </p>
        <TheoryBlock title="Regression vs Classification" cards={[
          { icon: '📏', title: 'Regression', body: 'Output is a real number. "How much?" or "How many?" Examples: house price, exam score, tomorrow\'s temperature, stock return, patient\'s age.', mono: 'ŷ ∈ ℝ  (any real number)' },
          { icon: '🏷️', title: 'Classification', body: 'Output is a discrete class. "Which one?" Examples: spam/not-spam, cat/dog/bird, digit 0–9. Output is a label or probability.', mono: 'ŷ ∈ {0, 1, ..., K}' },
          { icon: '📐', title: 'Evaluation', body: 'Regression quality is measured by how close predictions are to true values — MSE, MAE, RMSE, R². Classification uses accuracy, F1, AUC-ROC.', mono: 'MSE = (1/n)Σ(y−ŷ)²' },
        ]} />
        <Callout type="analogy" title="Analogy: thermometer vs label-maker">
          Classification is a label-maker — it stamps one of K fixed labels. Regression is a thermometer — it outputs a reading anywhere on a continuous scale. Same data, very different outputs.
        </Callout>
      </div>

      {/* 3.1.2 Linear neuron */}
      <div className={S}>
        <p className={LBL}>3.1.2 — Linear Neuron: No Hidden Layers, No Activation</p>
        <h2 className={H2}>The simplest possible neural network</h2>
        <p className={`${BODY} mb-4`}>
          A linear neuron takes d input features, multiplies each by a learned weight, adds a bias, and outputs the result directly. There is no activation function — output = input to the neuron. This is architecturally identical to classical linear regression.
        </p>
        <TheoryBlock title="Single neuron anatomy" cards={[
          { icon: '➡️', title: 'Inputs x', body: 'The feature vector: x = [x₁, x₂, ..., xd]. Each input is a measured property of the data point — hours studied, square footage, temperature reading, etc.', mono: 'x ∈ ℝᵈ' },
          { icon: '⚖️', title: 'Weights w', body: 'One scalar weight per input. Positive weight = feature increases ŷ. Negative = decreases. Magnitude = importance. Learned during training.', mono: 'w = [w₁, w₂, ..., wd] ∈ ℝᵈ' },
          { icon: '➕', title: 'Bias b', body: 'A single learnable scalar added after the weighted sum. Shifts the output up or down independently of the inputs. Without bias, the hyperplane is forced through the origin.', mono: 'b ∈ ℝ  (intercept term)' },
          { icon: '🔢', title: 'Pre-activation z', body: 'The weighted sum of inputs plus bias. For regression, this IS the output — no activation is applied on top.', mono: 'z = wᵀx + b' },
          { icon: '📤', title: 'Output ŷ', body: 'For a linear neuron: ŷ = z. No activation. Output can be any real number — positive, negative, or zero. Perfect for regression targets.', mono: 'ŷ = z = wᵀx + b' },
          { icon: '🔗', title: 'Neural network view', body: 'In network diagrams: one input layer (d nodes), direct connections to one output node. No hidden layers. The simplest non-trivial feedforward architecture.', mono: 'input → [neuron] → output' },
        ]} />
        <p className={`${BODY} mb-4`}>Adjust weights and bias to see the neuron compute its output in real time:</p>
        <NeuronDiagram theme={theme} />
        <DeepDive title="Why is this a neural network?">
          <p className={`text-sm ${BODY} mb-2`}>Technically, a single linear neuron is equivalent to ordinary least-squares linear regression — nothing "neural" about it. But it introduces the vocabulary and computational structure (forward pass, parameters, gradients) that scales to deep networks with millions of neurons.</p>
          <Callout type="info">The key insight: a deep network is just many of these computations composed. Add an activation function, stack multiple layers, and you get universal function approximation. The linear neuron is step 0.</Callout>
        </DeepDive>
      </div>

      {/* 3.1.3 MSE */}
      <div className={S}>
        <p className={LBL}>3.1.3 — Squared Loss (MSE)</p>
        <h2 className={H2}>Measuring how wrong the neuron is</h2>
        <p className={`${BODY} mb-4`}>
          The loss function quantifies the gap between predictions ŷ and true targets y. For regression, Mean Squared Error (MSE) is standard: it averages the squared residuals. Squaring ensures the loss is always non-negative and penalises large errors more than small ones.
        </p>
        <TheoryBlock title="MSE properties" cards={[
          { icon: '📐', title: 'Definition', body: 'Average of squared differences between each prediction and its true value. Division by n gives a per-sample average, making it independent of dataset size.', mono: 'L = (1/n) Σᵢ (yᵢ − ŷᵢ)²' },
          { icon: '🏔️', title: 'Convex surface', body: 'For a linear neuron, MSE as a function of (w, b) is a perfect bowl (quadratic). Only one global minimum — gradient descent is guaranteed to find it.', mono: 'L(w,b) is convex ✓' },
          { icon: '∂', title: 'Gradients', body: 'MSE has clean, closed-form gradients. ∂L/∂w = (2/n)·Σ(ŷᵢ−yᵢ)·xᵢ. ∂L/∂b = (2/n)·Σ(ŷᵢ−yᵢ). Simple residuals scaled by the inputs.', mono: '∂L/∂w = (2/n) Xᵀ(ŷ − y)' },
        ]} />
        <Callout type="formula" mono="L = (1/n) Σᵢ (yᵢ − ŷᵢ)²  =  (1/n) Σᵢ (yᵢ − wᵀxᵢ − b)²">
          Expanding: L depends on w and b. For fixed data, this is a quadratic function — the loss surface is an elliptic paraboloid in (w, b) space. The minimum is at the exact (w*, b*) that minimises prediction error.
        </Callout>
        <p className={`${BODY} mb-4`}>Drag w and b on the loss surface below. The yellow dot is your current (w, b); the green dot is the true minimum. Try to find it:</p>
        <LossSurface theme={theme} />
        <DeepDive title="MSE vs MAE — when to use which">
          <TheoryBlock title="" cards={[
            { icon: '📊', title: 'MSE', body: 'Penalises large errors quadratically. Sensitive to outliers — one large residual can dominate. Differentiable everywhere → smooth gradient descent.', mono: '(1/n)Σ(y−ŷ)²' },
            { icon: '📏', title: 'MAE', body: 'Penalises all errors equally (absolute value). Robust to outliers. Not differentiable at 0 — requires subgradient methods.', mono: '(1/n)Σ|y−ŷ|' },
            { icon: '⚖️', title: 'Huber Loss', body: 'MSE for small residuals, MAE for large ones. Best of both worlds: smooth near zero, robust to outliers. Controlled by δ.', mono: 'MSE if |r|≤δ, else MAE' },
          ]} />
        </DeepDive>
      </div>

      {/* ── Section 3.2 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-purple-500/5 border-purple-500/20' : 'bg-purple-50 border-purple-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-purple-400' : 'text-purple-600'}`}>Section 3.2</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Training and Inference</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>The full training loop — batch gradient descent, learning rate, loss curve — and what changes at inference time.</p>
      </div>

      {/* 3.2.1 Batch GD */}
      <div className={S}>
        <p className={LBL}>3.2.1 — Batch Gradient Descent for Single Neuron Regression</p>
        <h2 className={H2}>The training loop, step by step</h2>
        <p className={`${BODY} mb-4`}>
          Batch gradient descent computes the average gradient across all n training samples, then takes a single parameter update. Repeat for many epochs until the loss converges.
        </p>
        <TheoryBlock title="One training epoch" cards={[
          { icon: '1️⃣', title: 'Forward pass', body: 'Compute predictions ŷᵢ = w·xᵢ + b for every training sample. This is the "forward" direction: inputs → output.', mono: 'ŷ = Xw + b  (vectorised)' },
          { icon: '2️⃣', title: 'Compute loss', body: 'Calculate MSE between predictions and true targets. This scalar tells us how wrong the model is on the full training set.', mono: 'L = (1/n)‖ŷ − y‖²' },
          { icon: '3️⃣', title: 'Compute gradients', body: 'Differentiate L w.r.t. each parameter. For w: (2/n)·Xᵀ(ŷ−y). For b: (2/n)·Σ(ŷᵢ−yᵢ). These are the "backward" direction.', mono: '∂L/∂w = (2/n)Xᵀ(ŷ−y)' },
          { icon: '4️⃣', title: 'Update parameters', body: 'Subtract a fraction α of the gradient from current parameters. Step in the direction of steepest descent. w decreases if gradient is positive.', mono: 'w ← w − α·∂L/∂w' },
        ]} />
        <Callout type="info" title="Why 'batch'?">
          "Batch" means the gradient is averaged over the full training set before each update. This is the most stable variant. Stochastic GD (one sample per update) is noisier but faster per step. Mini-batch GD (small subsets) is the standard in deep learning.
        </Callout>
        <DeepDive title="Vectorised implementation">
          <p className={`text-sm ${BODY} mb-2`}>The entire batch update can be written in 4 numpy lines — no Python loops over samples:</p>
          <Callout type="formula" mono="y_pred = X @ w + b    # forward (n,)">Matrix multiply X (n×d) by w (d,) → predictions vector of length n.</Callout>
          <Callout type="formula" mono="dw = (2/n) * X.T @ (y_pred - y)">Gradient: d-dimensional vector. Each element is the correlation of feature j with residuals.</Callout>
          <Callout type="formula" mono="w -= lr * dw  |  b -= lr * db">Update step: subtract scaled gradient from current parameters.</Callout>
        </DeepDive>
      </div>

      {/* 3.2.2 Learning rate + loss curve */}
      <div className={S}>
        <p className={LBL}>3.2.2 — Learning Rate, Convergence and the Loss Curve</p>
        <h2 className={H2}>Watch the neuron learn in real time</h2>
        <p className={`${BODY} mb-4`}>
          The learning rate α is the most critical hyperparameter. It controls how large each gradient step is. The loss curve — a plot of L vs epoch — is your primary diagnostic for whether training is going well.
        </p>
        <Callout type="warning" title="Three types of loss curves">
          📉 <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>Healthy:</strong> decreasing, smoothly flattening. 
          〰️ <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>Oscillating:</strong> lr too large — overshooting. 
          ➡️ <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>Flat:</strong> lr too small or already converged.
        </Callout>
        <p className={`${BODY} mb-4`}>Set a learning rate and press Run. Watch weights update epoch-by-epoch and the fitted line converge toward the data. Try α = 0.5 to see divergence:</p>
        <GDTrainer theme={theme} />
        <DeepDive title="Convergence guarantees for linear regression">
          <p className={`text-sm ${BODY} mb-2`}>For a linear neuron with MSE loss, the loss surface is a convex quadratic. Gradient descent is guaranteed to converge to the global minimum if the learning rate satisfies:</p>
          <Callout type="formula" mono="α < 2 / λ_max(XᵀX / n)">
            Where λ_max is the largest eigenvalue of the input covariance matrix. In practice, feature normalisation (StandardScaler) ensures λ_max ≈ 1, making α &lt; 2 safe. After normalisation, α = 0.1–0.5 almost always works.
          </Callout>
        </DeepDive>
      </div>

      {/* 3.2.3 Inference */}
      <div className={S}>
        <p className={LBL}>3.2.3 — Prediction / Inference After Training</p>
        <h2 className={H2}>Using the trained neuron</h2>
        <p className={`${BODY} mb-4`}>
          Once training is complete, the weight w and bias b are fixed. Inference is just the forward pass — a single arithmetic operation. No gradients, no loss, no updates. The model is read-only.
        </p>
        <TheoryBlock title="Training vs Inference" cards={[
          { icon: '🏋️', title: 'Training mode', body: 'Forward pass → compute loss → backward pass (gradients) → update w and b. Repeat for all epochs. Computationally expensive.', mono: 'forward + backward + update' },
          { icon: '🔍', title: 'Inference mode', body: 'Forward pass only. Feed new input x through the frozen neuron: ŷ = wx + b. No loss, no gradients, no weight changes. Very fast.', mono: 'ŷ = wx + b  (read-only)' },
          { icon: '⚡', title: 'Speed difference', body: 'Inference is ~2× faster than one training forward pass (no gradient computation). For deep networks, the backward pass can be 3–5× more expensive than the forward pass.', mono: 'inference ≪ training cost' },
        ]} />
        <p className={`${BODY} mb-4`}>The neuron has been trained on y = 2x + 1 data. Drag x to run inference with the learned weights:</p>
        <InferenceDemo theme={theme} />
        <DeepDive title="In PyTorch: torch.no_grad()">
          <p className={`text-sm ${BODY} mb-2`}>PyTorch tracks operations to build a computation graph for backprop. During inference, this is wasteful. Wrapping predictions in <span className="font-mono">torch.no_grad()</span> disables gradient tracking, reducing memory and speeding up inference:</p>
          <Callout type="formula" mono="with torch.no_grad():  ŷ = model(x_new)">
            Also call <span className="font-mono">model.eval()</span> before inference — this switches off training-specific behaviours like dropout and batch normalisation running statistics.
          </Callout>
        </DeepDive>
      </div>

      {/* Code */}
      <div className={S}>
        <p className={LBL}>Python — NumPy from scratch</p>
        <h2 className={H2}>Single neuron regression, manual gradients</h2>
        <p className={`${BODY} mb-3`}>Full training loop in pure NumPy — forward pass, MSE loss, analytical gradients, weight updates.</p>
        <CodeBlock code={PYTHON_CODE_NEURON} />
      </div>

      <div className={S}>
        <p className={LBL}>Python — TensorFlow / Keras</p>
        <h2 className={H2}>Same model using tf.keras.Sequential + Dense</h2>
        <p className={`${BODY} mb-3`}>Identical task in TensorFlow: <span className={`font-mono text-xs ${theme === 'dark' ? 'text-indigo-300' : 'text-indigo-600'}`}>Dense(1)</span> with no activation, MSE loss, SGD optimiser — Keras handles the training loop automatically.</p>
        <CodeBlock code={PYTHON_CODE_TF} />
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme === 'dark' ? 'bg-indigo-500/10 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>10 questions covering both sections • +100 XP on completion</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="linear-neuron-regression" />
      </div>
    </motion.div>
  )
}
