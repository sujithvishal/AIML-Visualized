import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

const PYTHON_CODE_BINARY = `import tensorflow as tf
import numpy as np

# Dataset: hours studied → pass (1) / fail (0)
X = np.array([1,2,3,4,5,6,7,8], dtype=np.float32)
y = np.array([0,0,0,0,1,1,1,1], dtype=np.float32)

# Normalise
X_norm = (X - X.mean()) / X.std()

# Single neuron with sigmoid activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(1,))
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
    loss='binary_crossentropy'
)

history = model.fit(X_norm, y, epochs=200, verbose=0)
for ep in [49, 99, 149, 199]:
    print(f"Epoch {ep+1:3d}  Loss: {history.history['loss'][ep]:.4f}")

# Inference
x_means = X.mean(); x_std = X.std()
for hours in [3.0, 5.5, 8.0]:
    x_norm = np.array([[(hours - x_means) / x_std]])
    p = model.predict(x_norm, verbose=0)[0, 0]
    print(f"  {hours}h → P(pass)={p:.2%}  → {'Pass' if p >= 0.5 else 'Fail'}")`

const PYTHON_CODE_SGD = `import tensorflow as tf
import numpy as np

# Same dataset (already normalised)
X = np.array([-1.53,-1.09,-0.65,-0.22, 0.22, 0.65, 1.09, 1.53], dtype=np.float32)
y = np.array([0., 0., 0., 0., 1., 1., 1., 1.], dtype=np.float32)

# ── True SGD: batch_size=1 ──
model_sgd = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(1,))
])
model_sgd.compile(optimizer=tf.keras.optimizers.SGD(0.5),
                  loss='binary_crossentropy')
hist_sgd = model_sgd.fit(X, y, epochs=100, batch_size=1, shuffle=True, verbose=0)

# ── Mini-batch SGD: batch_size=4 ──
model_mini = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(1,))
])
model_mini.compile(optimizer=tf.keras.optimizers.SGD(0.5),
                   loss='binary_crossentropy')
hist_mini = model_mini.fit(X, y, epochs=100, batch_size=4, shuffle=True, verbose=0)

print(f"SGD  final loss: {hist_sgd.history['loss'][-1]:.4f}")
print(f"Mini final loss: {hist_mini.history['loss'][-1]:.4f}")`

const PYTHON_CODE_MULTICLASS = `import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Iris: 3 classes, 4 features
X_raw, y_raw = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# Multiple output neurons: Dense(3, activation='softmax')
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', input_shape=(4,))
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
    loss='sparse_categorical_crossentropy',   # integer labels, no one-hot needed
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=100, batch_size=16,
          validation_data=(X_test, y_test), verbose=0)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.2%}")

probs = model.predict(X_test[:1], verbose=0)
print("Softmax probs sample:", probs[0].round(3))`

const QUIZ_QUESTIONS = [
  { question: 'A classification task differs from regression in that it predicts:', options: ['A continuous number', 'A discrete class label', 'A probability only', 'A ranking'], correct: 1, explanation: 'Classification predicts which category (class) an input belongs to. Regression predicts a continuous value.' },
  { question: 'The sigmoid function σ(z) outputs:', options: ['Any real number', 'A value in (0, 1)', 'Exactly 0 or 1', 'A value in (−1, 1)'], correct: 1, explanation: 'σ(z) = 1/(1+e^(−z)) always produces a value strictly between 0 and 1 — suitable for binary probability output.' },
  { question: 'Binary cross-entropy loss is large when:', options: ['The model is correct and confident', 'The model is wrong and confident', 'The model outputs exactly 0.5', 'Loss is always constant'], correct: 1, explanation: '−[y·log(p)+(1−y)·log(1−p)] → ∞ as p→0 for y=1 or p→1 for y=0. Confident wrong predictions are punished heavily.' },
  { question: 'Stochastic gradient descent (SGD) updates weights:', options: ['Once after all epochs', 'After each single training sample', 'After every 100 samples', 'Only when loss decreases'], correct: 1, explanation: 'True SGD processes one sample at a time and updates weights immediately — noisy but often faster to make initial progress.' },
  { question: 'For binary classification inference, we predict class 1 when:', options: ['z > 0', 'σ(z) ≥ 0.5', 'Loss < 0.5', 'w > 0'], correct: 1, explanation: 'σ(z) ≥ 0.5 ⟺ z ≥ 0. Both conditions are equivalent. The 0.5 threshold can be adjusted based on precision/recall needs.' },
  { question: 'For K-class softmax output, the probabilities always:', options: ['Sum to K', 'Sum to 1', 'Are all equal', 'Are all > 0.5'], correct: 1, explanation: 'Softmax normalises K raw logits into a proper probability distribution: each pₖ ∈ (0,1) and Σpₖ = 1.' },
  { question: 'nn.CrossEntropyLoss in PyTorch expects:', options: ['Softmax probabilities as input', 'Raw logits (no activation)', 'Log-probabilities', 'Binary 0/1 targets'], correct: 1, explanation: 'CrossEntropyLoss applies log-softmax internally. Pass raw logits — applying softmax before would double-apply it.' },
  { question: 'Mini-batch SGD uses:', options: ['All n samples per update', 'One sample per update', 'A small subset of m samples per update', 'Random weights each step'], correct: 2, explanation: 'Mini-batch SGD averages the gradient over m samples (typically 32–512) before each update — a balance between SGD noise and batch GD stability.' },
  { question: 'The gradient of BCE loss w.r.t. logit z is:', options: ['2(ŷ − y)', 'ŷ − y  (prediction error)', '−y/ŷ', 'log(ŷ)'], correct: 1, explanation: 'For BCE + sigmoid: ∂L/∂z = σ(z) − y = ŷ − y. The sigmoid derivative cancels cleanly — prediction minus label.' },
  { question: 'Compared to batch GD, mini-batch SGD typically:', options: ['Converges slower always', 'Converges faster in practice and generalises better', 'Uses more memory', 'Cannot handle large datasets'], correct: 1, explanation: 'Mini-batch GD updates more frequently (m updates per epoch vs 1), escapes local minima better, and GPU hardware is optimised for matrix ops on batches.' },
]

// ── Sigmoid playground ─────────────────────────────────────────────────────────
function SigmoidPlayground({ theme }) {
  const [w, setW] = useState(1.5)
  const [b, setB] = useState(0.0)
  const canvasRef = useRef(null)
  const W = 380, H = 230

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)
    const pad = { l: 38, r: 10, t: 14, b: 28 }
    const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b
    const sig = z => 1 / (1 + Math.exp(-z))
    const toX = z => pad.l + ((z + 6) / 12) * pW
    const toY = p => (H - pad.b) - p * pH

    // grid
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'
    ctx.lineWidth = 1; ctx.setLineDash([3, 3])
    for (const p of [0.25, 0.5, 0.75, 1]) {
      ctx.beginPath(); ctx.moveTo(pad.l, toY(p)); ctx.lineTo(W - pad.r, toY(p)); ctx.stroke()
    }
    ctx.setLineDash([])

    // decision boundary at z=0 (P=0.5)
    ctx.beginPath(); ctx.moveTo(toX(0), pad.t); ctx.lineTo(toX(0), H - pad.b)
    ctx.strokeStyle = 'rgba(239,68,68,0.4)'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([])

    // axes
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke()

    // sigmoid curve
    const grad = ctx.createLinearGradient(pad.l, 0, W - pad.r, 0)
    grad.addColorStop(0, '#6366F1'); grad.addColorStop(0.5, '#8B5CF6'); grad.addColorStop(1, '#06B6D4')
    ctx.beginPath()
    for (let px = 0; px <= pW; px++) {
      const z = -6 + (px / pW) * 12
      const p = sig(w * z + b)
      px === 0 ? ctx.moveTo(pad.l + px, toY(p)) : ctx.lineTo(pad.l + px, toY(p))
    }
    ctx.strokeStyle = grad; ctx.lineWidth = 3; ctx.stroke()

    // axis ticks
    ctx.fillStyle = theme === 'dark' ? '#6B7280' : '#9CA3AF'
    ctx.font = '10px monospace'; ctx.textAlign = 'center'
    for (const z of [-6, -3, 0, 3, 6]) ctx.fillText(z, toX(z), H - pad.b + 13)
    ctx.textAlign = 'right'
    for (const p of [0, 0.25, 0.5, 0.75, 1]) ctx.fillText(p.toFixed(2), pad.l - 4, toY(p) + 3)
    ctx.textAlign = 'center'
    ctx.fillStyle = theme === 'dark' ? '#4B5563' : '#9CA3AF'
    ctx.fillText('z (pre-activation)', W / 2, H - 2)
  }, [w, b, theme])

  return (
    <div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            Weight w = <span className="font-bold text-indigo-400">{w.toFixed(1)}</span>
          </label>
          <input type="range" min="-4" max="4" step="0.1" value={w} onChange={e => setW(+e.target.value)} className="w-full accent-indigo-500" />
          <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>{Math.abs(w) > 2 ? 'Steep — sharp decision' : Math.abs(w) < 0.5 ? 'Flat — uncertain everywhere' : 'Balanced S-curve'}</p>
        </div>
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            Bias b = <span className="font-bold text-purple-400">{b.toFixed(1)}</span>
          </label>
          <input type="range" min="-4" max="4" step="0.1" value={b} onChange={e => setB(+e.target.value)} className="w-full accent-purple-500" />
          <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>{b > 0 ? 'Boundary shifted left' : b < 0 ? 'Boundary shifted right' : 'Boundary at z=0'}</p>
        </div>
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
      <p className={`text-xs mt-2 text-center font-mono ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        σ(z) = 1/(1+e⁻ᶻ) &nbsp;·&nbsp; z = {w.toFixed(1)}·x + {b.toFixed(1)} &nbsp;·&nbsp; Red = decision boundary
      </p>
    </div>
  )
}

// ── BCE loss visualiser ────────────────────────────────────────────────────────
function BCEViz({ theme }) {
  const [pred, setPred] = useState(0.7)
  const W = 380, H = 210
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)
    const pad = { l: 38, r: 10, t: 14, b: 28 }
    const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b
    const toX = p => pad.l + p * pW
    const toY = l => (H - pad.b) - (Math.min(l, 4) / 4) * pH

    // grid
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'
    ctx.lineWidth = 1; ctx.setLineDash([3, 3])
    for (let v = 1; v <= 4; v++) {
      ctx.beginPath(); ctx.moveTo(pad.l, toY(v)); ctx.lineTo(W - pad.r, toY(v)); ctx.stroke()
    }
    ctx.setLineDash([])

    // axes
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke()

    const drawCurve = (yTrue, color) => {
      ctx.beginPath()
      for (let px = 0; px <= pW; px++) {
        const p = Math.max(0.001, Math.min(0.999, px / pW))
        const loss = yTrue === 1 ? -Math.log(p) : -Math.log(1 - p)
        px === 0 ? ctx.moveTo(toX(px / pW), toY(loss)) : ctx.lineTo(toX(px / pW), toY(loss))
      }
      ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.stroke()
    }
    drawCurve(1, '#6366F1')
    drawCurve(0, '#EF4444')

    // vertical marker
    ctx.beginPath(); ctx.moveTo(toX(pred), pad.t); ctx.lineTo(toX(pred), H - pad.b)
    ctx.strokeStyle = '#F59E0B'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([])

    const l1 = Math.min(-Math.log(pred), 4), l0 = Math.min(-Math.log(1 - pred), 4)
    ctx.beginPath(); ctx.arc(toX(pred), toY(l1), 5, 0, Math.PI * 2); ctx.fillStyle = '#6366F1'; ctx.fill()
    ctx.beginPath(); ctx.arc(toX(pred), toY(l0), 5, 0, Math.PI * 2); ctx.fillStyle = '#EF4444'; ctx.fill()

    // labels
    ctx.fillStyle = theme === 'dark' ? '#6B7280' : '#9CA3AF'
    ctx.font = '10px monospace'; ctx.textAlign = 'center'
    for (const p of [0.25, 0.5, 0.75, 1]) ctx.fillText(p.toFixed(2), toX(p), H - pad.b + 13)
    ctx.textAlign = 'right'
    for (let v = 0; v <= 4; v++) ctx.fillText(v, pad.l - 4, toY(v) + 3)
  }, [pred, theme])

  const l1 = (-Math.log(Math.max(0.001, pred))).toFixed(3)
  const l0 = (-Math.log(Math.max(0.001, 1 - pred))).toFixed(3)

  return (
    <div>
      <div className="mb-4">
        <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
          Predicted p = <span className="font-bold text-yellow-400">{pred.toFixed(2)}</span>
        </label>
        <input type="range" min="0.01" max="0.99" step="0.01" value={pred} onChange={e => setPred(+e.target.value)} className="w-full accent-yellow-500" />
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
      <div className="flex gap-3 mt-3 text-xs">
        <div className={`flex-1 p-3 rounded-xl border ${theme === 'dark' ? 'bg-slate-800 border-white/10' : 'bg-indigo-50 border-indigo-200'}`}>
          <p className="text-indigo-400 font-semibold">y=1 loss = <span className={`font-mono ${theme === 'dark' ? 'text-white' : 'text-indigo-700'}`}>{l1}</span></p>
          <p className={`mt-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>−log(p) → ∞ when p→0</p>
        </div>
        <div className={`flex-1 p-3 rounded-xl border ${theme === 'dark' ? 'bg-slate-800 border-white/10' : 'bg-red-50 border-red-200'}`}>
          <p className="text-red-400 font-semibold">y=0 loss = <span className={`font-mono ${theme === 'dark' ? 'text-white' : 'text-red-700'}`}>{l0}</span></p>
          <p className={`mt-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>−log(1−p) → ∞ when p→1</p>
        </div>
      </div>
    </div>
  )
}

// ── SGD trainer ────────────────────────────────────────────────────────────────
function SGDTrainer({ theme }) {
  const [batchSize, setBatchSize] = useState(1)
  const [lr, setLr] = useState(0.5)
  const [epoch, setEpoch] = useState(0)
  const [lossHist, setLossHist] = useState([])
  const [w, setW] = useState(-1.5)
  const [b, setBVal] = useState(0.0)
  const [running, setRunning] = useState(false)
  const rafRef = useRef(null)
  const stateRef = useRef({ w: -1.5, b: 0, lr: 0.5, batch: 1, losses: [], ep: 0 })

  // Dataset: normalised X, binary y
  const data = [
    [-1.53, 0], [-1.09, 0], [-0.65, 0], [-0.22, 0],
    [0.22, 1], [0.65, 1], [1.09, 1], [1.53, 1],
  ]
  const sig = z => 1 / (1 + Math.exp(-z))
  const bce = (p, y) => -(y * Math.log(Math.max(p, 1e-7)) + (1 - y) * Math.log(Math.max(1 - p, 1e-7)))

  function runStep() {
    const s = stateRef.current
    if (s.ep >= 150) { setRunning(false); return }

    // shuffle
    const shuffled = [...data].sort(() => Math.random() - 0.5)
    const batches = []
    for (let i = 0; i < shuffled.length; i += s.batch) batches.push(shuffled.slice(i, i + s.batch))

    for (const batch of batches) {
      const preds = batch.map(([x]) => sig(s.w * x + s.b))
      const dw = batch.reduce((acc, [x], i) => acc + (preds[i] - batch[i][1]) * x, 0) / batch.length
      const db = batch.reduce((acc, [, y], i) => acc + (preds[i] - y), 0) / batch.length
      s.w -= s.lr * dw
      s.b -= s.lr * db
    }

    // full loss
    const fullLoss = data.reduce((acc, [x, y]) => acc + bce(sig(s.w * x + s.b), y), 0) / data.length
    s.losses = [...s.losses, fullLoss]
    s.ep += 1
    setW(s.w); setBVal(s.b); setEpoch(s.ep); setLossHist([...s.losses])
    rafRef.current = requestAnimationFrame(runStep)
  }

  const start = () => {
    stateRef.current = { w: -1.5, b: 0, lr, batch: batchSize, losses: [], ep: 0 }
    setW(-1.5); setBVal(0); setLossHist([]); setEpoch(0); setRunning(true)
    rafRef.current = requestAnimationFrame(runStep)
  }
  const reset = () => {
    cancelAnimationFrame(rafRef.current)
    stateRef.current = { w: -1.5, b: 0, lr, batch: batchSize, losses: [], ep: 0 }
    setW(-1.5); setBVal(0); setLossHist([]); setEpoch(0); setRunning(false)
  }
  useEffect(() => () => cancelAnimationFrame(rafRef.current), [])

  const canvasRef = useRef(null)
  const W = 380, H = 190

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)
    const pad = { l: 42, r: 12, t: 14, b: 28 }
    const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b

    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke()

    if (lossHist.length === 0) {
      ctx.fillStyle = theme === 'dark' ? '#374151' : '#D1D5DB'
      ctx.font = '12px sans-serif'; ctx.textAlign = 'center'
      ctx.fillText('Press Run to start training', W / 2, H / 2); return
    }

    const maxL = Math.max(...lossHist, 0.01)
    const toX = i => pad.l + (i / 149) * pW
    const toY = l => (H - pad.b) - (Math.min(l, maxL) / maxL) * pH

    ctx.setLineDash([3, 3])
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'
    for (let v = 1; v <= 4; v++) {
      ctx.beginPath(); ctx.moveTo(pad.l, toY(maxL * v / 4)); ctx.lineTo(W - pad.r, toY(maxL * v / 4)); ctx.stroke()
    }
    ctx.setLineDash([])

    const grad = ctx.createLinearGradient(pad.l, 0, W - pad.r, 0)
    grad.addColorStop(0, '#EF4444'); grad.addColorStop(0.5, '#F59E0B'); grad.addColorStop(1, '#10B981')
    ctx.beginPath()
    lossHist.forEach((l, i) => i === 0 ? ctx.moveTo(toX(i), toY(l)) : ctx.lineTo(toX(i), toY(l)))
    ctx.strokeStyle = grad; ctx.lineWidth = 2.5; ctx.stroke()

    const last = lossHist.length - 1
    ctx.beginPath(); ctx.arc(toX(last), toY(lossHist[last]), 5, 0, Math.PI * 2)
    ctx.fillStyle = '#10B981'; ctx.fill()

    ctx.fillStyle = theme === 'dark' ? '#6B7280' : '#9CA3AF'
    ctx.font = '9px monospace'; ctx.textAlign = 'right'
    for (let v = 0; v <= 4; v++) ctx.fillText((maxL * v / 4).toFixed(2), pad.l - 4, toY(maxL * v / 4) + 3)
    ctx.textAlign = 'center'
    for (const ep of [0, 50, 100, 149]) ctx.fillText(ep, toX(ep), H - pad.b + 12)
    ctx.fillText('Epoch', W / 2, H - 2)
    ctx.save(); ctx.translate(11, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('BCE Loss', 0, 0); ctx.restore()
  }, [lossHist, theme])

  const bsLabels = { 1: 'True SGD (1 sample)', 4: 'Mini-batch (4)', 8: 'Batch GD (all 8)' }
  const currentLoss = lossHist.length > 0 ? lossHist[lossHist.length - 1].toFixed(4) : '—'

  return (
    <div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            Batch size = <span className="font-bold text-cyan-400">{batchSize}</span>
            <span className={`ml-2 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{bsLabels[batchSize]}</span>
          </label>
          <div className="flex gap-2">
            {[1, 4, 8].map(bs => (
              <button key={bs} disabled={running} onClick={() => { setBatchSize(bs); stateRef.current.batch = bs }}
                className={`flex-1 py-1.5 rounded-lg text-xs font-bold border transition-all disabled:opacity-50 ${batchSize === bs
                  ? theme === 'dark' ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300' : 'bg-cyan-100 border-cyan-300 text-cyan-700'
                  : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'}`}>{bs}</button>
            ))}
          </div>
        </div>
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            Learning rate α = <span className="font-bold text-yellow-400">{lr.toFixed(2)}</span>
          </label>
          <input type="range" min="0.05" max="1.5" step="0.05" value={lr} disabled={running}
            onChange={e => { setLr(+e.target.value); stateRef.current.lr = +e.target.value }}
            className="w-full accent-yellow-500" />
        </div>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={running ? reset : start}
          className={`flex-1 py-2 rounded-xl text-sm font-semibold border transition-all ${running
            ? 'bg-red-500/20 border-red-500/40 text-red-400 hover:bg-red-500/30'
            : 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300 hover:bg-indigo-500/30'}`}>
          {running ? '⏹ Stop' : epoch > 0 ? '🔄 Reset & Run' : '▶ Run Training'}
        </button>
      </div>
      <div className="grid grid-cols-4 gap-2 mb-4 text-xs">
        {[
          { label: 'Epoch', val: epoch, color: 'text-indigo-400' },
          { label: 'BCE Loss', val: currentLoss, color: 'text-red-400' },
          { label: 'w', val: w.toFixed(3), color: 'text-purple-400' },
          { label: 'b', val: b.toFixed(3), color: 'text-cyan-400' },
        ].map(m => (
          <div key={m.label} className={`rounded-xl p-2 border text-center ${theme === 'dark' ? 'bg-slate-800 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className={`font-mono font-bold text-sm ${m.color}`}>{m.val}</p>
            <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{m.label}</p>
          </div>
        ))}
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
    </div>
  )
}

// ── Softmax visualiser ─────────────────────────────────────────────────────────
function SoftmaxViz({ theme }) {
  const classes = ['Class A', 'Class B', 'Class C']
  const [logits, setLogits] = useState([2.0, 1.0, 0.1])
  const colors = ['#6366F1', '#10B981', '#F59E0B']

  const softmax = zs => {
    const m = Math.max(...zs)
    const exps = zs.map(z => Math.exp(z - m))
    const s = exps.reduce((a, b) => a + b, 0)
    return exps.map(e => e / s)
  }
  const probs = softmax(logits)

  return (
    <div>
      <p className={`text-xs mb-3 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
        Drag raw logits (z) — watch softmax probabilities update. Sum always = 1.
      </p>
      <div className="space-y-4 mb-4">
        {classes.map((cls, i) => (
          <div key={cls}>
            <div className="flex justify-between mb-1">
              <span className={`text-sm font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{cls}</span>
              <div className="flex gap-3 text-xs font-mono">
                <span className={theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}>z = <span className="font-bold" style={{ color: colors[i] }}>{logits[i].toFixed(1)}</span></span>
                <span className={`font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{(probs[i] * 100).toFixed(1)}%</span>
              </div>
            </div>
            <input type="range" min="-3" max="5" step="0.1" value={logits[i]}
              onChange={e => setLogits(prev => { const n = [...prev]; n[i] = +e.target.value; return n })}
              className="w-full mb-1.5" style={{ accentColor: colors[i] }} />
            <div className={`h-3 rounded-full overflow-hidden ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-200'}`}>
              <motion.div className="h-full rounded-full" style={{ backgroundColor: colors[i] }}
                animate={{ width: `${probs[i] * 100}%` }} transition={{ duration: 0.15 }} />
            </div>
          </div>
        ))}
      </div>
      <div className={`rounded-xl p-3 font-mono text-xs border ${theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-300' : 'bg-gray-100 border-gray-200 text-gray-700'}`}>
        softmax([{logits.map(z => z.toFixed(1)).join(', ')}]) = [{probs.map(p => p.toFixed(3)).join(', ')}]
        <br />sum = {probs.reduce((a, b) => a + b, 0).toFixed(6)} ✓
      </div>
    </div>
  )
}

// ── Multi-class neuron diagram ─────────────────────────────────────────────────
function MulticlassDiagram({ theme }) {
  const W = 380, H = 230
  const canvasRef = useRef(null)
  const [highlight, setHighlight] = useState(null)

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)

    const inXs = [30, 30, 30, 30]
    const inYs = [42, 90, 138, 186]
    const outYs = [60, 115, 170]
    const midX = 200
    const outX = 340
    const r = 16

    const inColors = ['#6366F1', '#8B5CF6', '#06B6D4', '#10B981']
    const outColors = ['#6366F1', '#10B981', '#F59E0B']
    const classLabels = ['Cat', 'Dog', 'Bird']

    // connections
    inYs.forEach((iy, ii) => {
      outYs.forEach((oy, oi) => {
        const active = highlight === oi
        ctx.beginPath(); ctx.moveTo(inXs[ii] + r, iy); ctx.lineTo(midX - r, oy)
        ctx.strokeStyle = active ? outColors[oi] : (theme === 'dark' ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)')
        ctx.lineWidth = active ? 2 : 1; ctx.stroke()
      })
    })

    // output → softmax connections
    outYs.forEach((oy, i) => {
      ctx.beginPath(); ctx.moveTo(midX + r, oy); ctx.lineTo(outX - r, oy)
      ctx.strokeStyle = outColors[i]; ctx.lineWidth = 2; ctx.stroke()
    })

    // input nodes
    inYs.forEach((iy, i) => {
      ctx.beginPath(); ctx.arc(inXs[i], iy, r, 0, Math.PI * 2)
      ctx.fillStyle = theme === 'dark' ? '#1E293B' : '#F1F5F9'
      ctx.fill(); ctx.strokeStyle = inColors[i]; ctx.lineWidth = 2; ctx.stroke()
      ctx.fillStyle = inColors[i]; ctx.font = 'bold 10px monospace'; ctx.textAlign = 'center'
      ctx.fillText(`x${i + 1}`, inXs[i], iy + 4)
    })

    // neuron nodes (one per class)
    outYs.forEach((oy, i) => {
      ctx.beginPath(); ctx.arc(midX, oy, r, 0, Math.PI * 2)
      ctx.fillStyle = highlight === i ? outColors[i] : (theme === 'dark' ? '#312E81' : '#EEF2FF')
      ctx.fill(); ctx.strokeStyle = outColors[i]; ctx.lineWidth = 2.5; ctx.stroke()
      ctx.fillStyle = highlight === i ? 'white' : outColors[i]
      ctx.font = 'bold 10px monospace'; ctx.textAlign = 'center'
      ctx.fillText(`z${i + 1}`, midX, oy + 4)
    })

    // softmax box
    const smX = outX - 26, smY = 50, smW = 52, smH = 130
    ctx.fillStyle = theme === 'dark' ? '#0F172A' : '#F8FAFC'
    ctx.strokeStyle = theme === 'dark' ? '#334155' : '#E2E8F0'
    ctx.lineWidth = 1.5
    const rr = 8
    ctx.beginPath()
    ctx.moveTo(smX + rr, smY); ctx.lineTo(smX + smW - rr, smY)
    ctx.arcTo(smX + smW, smY, smX + smW, smY + rr, rr)
    ctx.lineTo(smX + smW, smY + smH - rr)
    ctx.arcTo(smX + smW, smY + smH, smX + smW - rr, smY + smH, rr)
    ctx.lineTo(smX + rr, smY + smH)
    ctx.arcTo(smX, smY + smH, smX, smY + smH - rr, rr)
    ctx.lineTo(smX, smY + rr)
    ctx.arcTo(smX, smY, smX + rr, smY, rr)
    ctx.closePath(); ctx.fill(); ctx.stroke()

    ctx.fillStyle = theme === 'dark' ? '#6366F1' : '#4F46E5'
    ctx.font = 'bold 9px monospace'; ctx.textAlign = 'center'
    ctx.fillText('soft', outX, 88); ctx.fillText('max', outX, 100)

    // class output labels
    outYs.forEach((oy, i) => {
      ctx.fillStyle = outColors[i]; ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'left'
      ctx.fillText(`p(${classLabels[i]})`, outX + 30, oy + 4)
    })

    // label: 4 inputs
    ctx.fillStyle = theme === 'dark' ? '#4B5563' : '#9CA3AF'
    ctx.font = '9px sans-serif'; ctx.textAlign = 'center'
    ctx.fillText('4 inputs', inXs[0], 18)
    ctx.fillText('3 neurons', midX, 18)
  }, [highlight, theme])

  return (
    <div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border mb-3 ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
      <div className="flex gap-2">
        {['Class A', 'Class B', 'Class C'].map((cls, i) => (
          <button key={cls}
            onMouseEnter={() => setHighlight(i)} onMouseLeave={() => setHighlight(null)}
            className={`flex-1 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
              theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400 hover:border-indigo-500/40 hover:text-indigo-300' : 'bg-gray-100 border-gray-200 text-gray-500 hover:border-indigo-300 hover:text-indigo-600'
            }`}>Hover {cls}</button>
        ))}
      </div>
      <p className={`text-xs mt-2 text-center ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>Hover class buttons to highlight its weight connections</p>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function LinearNeuronClassification() {
  const { theme } = useApp()
  const S = `rounded-2xl border p-6 mb-6 ${theme === 'dark' ? 'bg-slate-900/80 border-white/10' : 'bg-white border-gray-200'}`
  const LBL = `text-xs font-semibold uppercase tracking-wider mb-3 ${theme === 'dark' ? 'text-purple-400' : 'text-purple-600'}`
  const H2 = `text-xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`
  const BODY = `text-sm leading-relaxed ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-3xl">

      {/* Hero */}
      <div className="mb-8">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-purple-500/20 border border-purple-500/30 mb-4">
          <span className="text-xs text-purple-400 font-medium">Deep Neural Networks • Session 4</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Linear Neural Networks <span className="gradient-text">for Classification</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Add an <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>activation function</strong> to a linear neuron and it becomes a classifier. Add more output neurons and it handles any number of classes.
        </p>
        <Callout type="analogy" title="From regression to classification">
          Session 3 taught the neuron to predict a number. Now we ask it to predict a category. The only change: wrap the output in sigmoid (binary) or softmax (multi-class). Same neuron, same gradient descent — different activation and loss.
        </Callout>
      </div>

      <TheoryBlock title="Core Concepts" cards={[
        { icon: 'σ', title: 'Sigmoid activation', body: 'Squashes the linear output z into (0,1). Interpreted as P(y=1|x). Applied after z = wx + b for binary classification.', mono: 'σ(z) = 1 / (1 + e⁻ᶻ)' },
        { icon: '📉', title: 'Binary Cross-Entropy', body: 'Loss for binary classification. Heavily penalises confident wrong predictions. Derived from maximum likelihood with Bernoulli distribution.', mono: 'L = −[y·log p + (1−y)·log(1−p)]' },
        { icon: '🎲', title: 'Softmax activation', body: 'Generalises sigmoid to K classes. Converts K raw logits to a proper probability distribution — positive values summing to 1.', mono: 'p_k = e^{z_k} / Σ e^{z_j}' },
        { icon: '📊', title: 'Cross-Entropy (multi)', body: 'Only the log-prob of the true class contributes. Equivalent to negative log-likelihood over a categorical distribution.', mono: 'L = −Σ_k y_k · log(p_k)' },
        { icon: '🔀', title: 'SGD vs Mini-batch', body: 'SGD updates after every sample (noisy, fast progress). Mini-batch averages over m samples (balance of noise and stability). Batch GD uses all n samples.', mono: 'mini-batch: m = 32–512' },
        { icon: '∂', title: 'Clean gradient', body: 'BCE + sigmoid gives gradient = ŷ − y. Cross-entropy + softmax gives the same. This elegant result makes training numerically stable.', mono: '∂L/∂z = ŷ − y' },
      ]} />

      {/* ── Section 4.1 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-purple-500/5 border-purple-500/20' : 'bg-purple-50 border-purple-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-purple-400' : 'text-purple-600'}`}>Section 4.1</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Binary Classification (Single Neuron)</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Sigmoid, BCE loss, SGD training, and inference for two-class problems.</p>
      </div>

      {/* 4.1.1 Classification task */}
      <div className={S}>
        <p className={LBL}>4.1.1 — Classification Task: "Which Category?"</p>
        <h2 className={H2}>Predicting a discrete class label</h2>
        <p className={`${BODY} mb-4`}>
          Classification maps inputs to one of K discrete classes. Binary classification (K=2) is the simplest case — is this email spam? Does this patient have the disease? Will this customer churn?
        </p>
        <TheoryBlock title="Binary classification setup" cards={[
          { icon: '🏷️', title: 'Binary labels', body: 'Targets are 0 or 1. The neuron must learn to output high probability for class 1 and low for class 0. We treat the output as P(y=1|x).', mono: 'y ∈ {0, 1}' },
          { icon: '📊', title: 'What the neuron outputs', body: 'We want a probability — a number in (0, 1). The raw linear output z = wx + b can be any real number, so we apply sigmoid to squash it.', mono: 'p = σ(wx + b) ∈ (0, 1)' },
          { icon: '✂️', title: 'Decision threshold', body: 'Predict class 1 if p ≥ threshold (default 0.5). Lowering the threshold increases recall (catches more positives) at the cost of precision.', mono: 'ŷ = 1 if p ≥ 0.5 else 0' },
        ]} />
        <Callout type="analogy" title="Analogy: confidence meter">
          Think of the neuron as a confidence meter. It reads the features and says "I'm 87% sure this is class 1." The threshold then converts that confidence into a yes/no decision. Changing the threshold changes where you draw that line.
        </Callout>
      </div>

      {/* 4.1.2 Sigmoid */}
      <div className={S}>
        <p className={LBL}>4.1.2 — Sigmoid Activation Function</p>
        <h2 className={H2}>Turning any real number into a probability</h2>
        <p className={`${BODY} mb-4`}>
          The sigmoid (logistic) function maps any real-valued z to (0, 1). It's the natural choice for binary classification because its output can be directly interpreted as a probability.
        </p>
        <TheoryBlock title="Sigmoid properties" cards={[
          { icon: '📐', title: 'Formula', body: 'σ(z) = 1/(1+e^(−z)). Symmetric around z=0 where σ(0)=0.5. Smooth and differentiable everywhere — essential for gradient descent.', mono: 'σ(z) = 1 / (1 + e⁻ᶻ)' },
          { icon: '∂', title: 'Derivative', body: 'σ\'(z) = σ(z)·(1−σ(z)). The gradient is maximal at z=0 (= 0.25) and vanishes at extremes. This causes the vanishing gradient problem in deep networks.', mono: "σ'(z) = σ(z)(1 − σ(z))" },
          { icon: '🔗', title: 'Link to logistic regression', body: 'A single neuron with sigmoid activation IS logistic regression. The neural network framework just provides a cleaner way to compute gradients and extend to deeper architectures.', mono: 'single neuron + sigmoid = logistic regression' },
        ]} />
        <p className={`${BODY} mb-4`}>Adjust weight and bias to see how they reshape the sigmoid curve and shift the decision boundary:</p>
        <SigmoidPlayground theme={theme} />
        <DeepDive title="Why sigmoid causes vanishing gradients in deep nets">
          <p className={`text-sm ${BODY} mb-2`}>σ'(z) = σ(z)·(1−σ(z)) has a maximum of 0.25. In a deep network, the gradient is multiplied by this value at every layer during backpropagation. After 10 layers: 0.25^10 ≈ 0.000001. The gradient effectively vanishes before reaching early layers.</p>
          <Callout type="warning">This is why modern deep networks use ReLU instead of sigmoid for hidden layers. Sigmoid is still appropriate for the final output layer in binary classification.</Callout>
        </DeepDive>
      </div>

      {/* 4.1.3 BCE loss */}
      <div className={S}>
        <p className={LBL}>4.1.3 — Binary Cross-Entropy Loss</p>
        <h2 className={H2}>The right loss for binary classification</h2>
        <p className={`${BODY} mb-4`}>
          Binary cross-entropy (BCE) is derived from maximum likelihood estimation — it's the log-likelihood of a Bernoulli model. It penalises confident wrong predictions exponentially, making learning signal strong even when the model is badly wrong.
        </p>
        <TheoryBlock title="BCE breakdown" cards={[
          { icon: '📐', title: 'Formula', body: 'For one sample: L = −[y·log(p) + (1−y)·log(1−p)]. When y=1 only the first term survives; when y=0 only the second. Averaged over all n samples.', mono: 'L = −(1/n)Σ[yᵢ log pᵢ + (1−yᵢ)log(1−pᵢ)]' },
          { icon: '🎯', title: 'Clean gradient', body: 'The gradient of BCE w.r.t. the logit z is simply ŷ−y. This elegant result comes from the sigmoid derivative cancelling with the BCE derivative.', mono: '∂L/∂z = σ(z) − y = ŷ − y' },
          { icon: '⛔', title: 'Why not MSE?', body: 'MSE with sigmoid produces a non-convex loss surface with many local minima. BCE with sigmoid is convex — gradient descent is guaranteed to converge to the global optimum.', mono: 'BCE + sigmoid = convex ✓' },
        ]} />
        <Callout type="formula" mono="L = −[y·log(p) + (1−y)·log(1−p)]">
          Intuition: if y=1 and p≈0 (confident wrong), −log(0.001) ≈ 6.9 — huge loss. If y=1 and p≈1 (confident right), −log(0.999) ≈ 0.001 — tiny loss. Exactly the incentive we want.
        </Callout>
        <p className={`${BODY} mb-4`}>Drag predicted probability p to see how the loss changes for each true label:</p>
        <BCEViz theme={theme} />
      </div>

      {/* 4.1.4 SGD training */}
      <div className={S}>
        <p className={LBL}>4.1.4 — Training with Stochastic Gradient Descent</p>
        <h2 className={H2}>SGD, Mini-batch, and Batch GD compared</h2>
        <p className={`${BODY} mb-4`}>
          The weight update rule is identical to regression — subtract a scaled gradient. The difference is which gradient: BCE gradient instead of MSE, and the update frequency depends on which SGD variant you use.
        </p>
        <TheoryBlock title="Gradient descent variants" cards={[
          { icon: '🎲', title: 'SGD (batch=1)', body: 'Update after every single sample. High variance — loss curve is noisy. Very fast first progress; can escape shallow local minima. Poor use of parallelism.', mono: 'w ← w − α·(ŷᵢ−yᵢ)·xᵢ' },
          { icon: '📦', title: 'Mini-batch (batch=m)', body: 'Average gradient over m samples before updating. Standard in deep learning — m=32 to 512. Balances noise and speed. Works well with GPU parallelism.', mono: 'w ← w − α·(1/m)·Xᵀ(ŷ−y)' },
          { icon: '🗄️', title: 'Batch GD (batch=n)', body: 'Average over the full dataset before each update. Most stable, least noisy. One update per epoch — very slow for large datasets. Never used in practice at scale.', mono: 'w ← w − α·(1/n)·Xᵀ(ŷ−y)' },
        ]} />
        <Callout type="warning" title="SGD noise can help">
          The noise in SGD acts as implicit regularisation — it can help the model escape sharp minima and find flatter ones that generalise better. This is partly why SGD-trained models sometimes outperform perfectly converged batch GD models.
        </Callout>
        <p className={`${BODY} mb-4`}>Select batch size (1 = SGD, 4 = mini-batch, 8 = batch GD) and compare convergence speed and smoothness:</p>
        <SGDTrainer theme={theme} />
        <DeepDive title="SGD with momentum">
          <p className={`text-sm ${BODY} mb-2`}>Plain SGD can oscillate in narrow loss valleys. Momentum adds a velocity term that accumulates gradient history, damping oscillations and speeding up convergence:</p>
          <Callout type="formula" mono="v ← β·v + (1−β)·∇L    w ← w − α·v">
            β ≈ 0.9 is standard. Momentum smooths the gradient path, effectively allowing larger learning rates. Most practitioners use SGD with momentum=0.9 or Adam (adaptive learning rates).
          </Callout>
        </DeepDive>
      </div>

      {/* 4.1.5 Inference binary */}
      <div className={S}>
        <p className={LBL}>4.1.5 — Inference for Binary Classification</p>
        <h2 className={H2}>Forward pass → probability → decision</h2>
        <p className={`${BODY} mb-4`}>
          After training, inference is three steps: (1) linear combination z = wx + b, (2) sigmoid p = σ(z), (3) threshold ŷ = 1 if p ≥ 0.5. No backward pass, no weight updates — read-only.
        </p>
        <TheoryBlock title="Inference pipeline" cards={[
          { icon: '1️⃣', title: 'Linear combination', body: 'z = wᵀx + b. Same as the regression neuron — just a dot product and add bias. Can be any real number.', mono: 'z = wᵀx + b  ∈ ℝ' },
          { icon: '2️⃣', title: 'Sigmoid', body: 'p = σ(z). Squashes z to (0,1). Interpret as P(y=1|x) — the probability the input belongs to class 1.', mono: 'p = σ(z)  ∈ (0,1)' },
          { icon: '3️⃣', title: 'Threshold decision', body: 'ŷ = 1 if p ≥ τ, else 0. Default τ=0.5. Adjusting τ trades precision for recall — lower τ catches more positives.', mono: 'ŷ = 𝟙[p ≥ τ]' },
        ]} />
        <Callout type="success" title="Adjusting the threshold">
          The 0.5 threshold is arbitrary. In medical diagnosis, you might set τ = 0.2 to maximise recall (catch all positives) at the cost of more false alarms. In spam detection, τ = 0.8 minimises false positives (wrongly flagged emails).
        </Callout>
      </div>

      {/* ── Section 4.2 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 4.2</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Multi-Class Classification (Multiple Output Neurons)</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Extending to K classes with multiple output neurons, softmax, and cross-entropy loss.</p>
      </div>

      {/* 4.2.1 Multiple output neurons */}
      <div className={S}>
        <p className={LBL}>4.2.1 — Multiple Output Neurons for Multi-Class Problems</p>
        <h2 className={H2}>One neuron per class</h2>
        <p className={`${BODY} mb-4`}>
          For K classes, we use K output neurons. Each neuron has its own weight vector and bias — it learns to "detect" its assigned class. The neurons share the same input features but have independent parameters.
        </p>
        <TheoryBlock title="Architecture" cards={[
          { icon: '🔢', title: 'K output neurons', body: 'One linear neuron per class. Each computes a class score (logit): zₖ = wₖᵀx + bₖ. The class with the highest logit wins.', mono: 'z = Wx + b  (W is K×d matrix)' },
          { icon: '📦', title: 'Parameter count', body: 'Each of K neurons has d weights (one per input feature) plus one bias. Total: K×(d+1) parameters. For Iris (d=4, K=3): 15 parameters.', mono: 'params = K × (d + 1)' },
          { icon: '🔁', title: 'Shared inputs', body: 'All K neurons receive the same input x. They differ only in their learned weights — each captures a different linear combination of features that discriminates its class.', mono: 'all neurons share x' },
        ]} />
        <p className={`${BODY} mb-4`}>Hover the class buttons to see which weight connections belong to each output neuron:</p>
        <MulticlassDiagram theme={theme} />
      </div>

      {/* 4.2.2 Softmax */}
      <div className={S}>
        <p className={LBL}>4.2.2 — Softmax Activation Function</p>
        <h2 className={H2}>Turning K logits into a probability distribution</h2>
        <p className={`${BODY} mb-4`}>
          K raw logits can be any real numbers and don't form a valid probability distribution. Softmax exponentiates each logit and normalises by the total — giving K positive probabilities that sum to exactly 1.
        </p>
        <TheoryBlock title="Softmax properties" cards={[
          { icon: '📐', title: 'Formula', body: 'Exponentiate each logit (ensuring positivity), then divide by the sum to normalise. The exponential makes larger logits dominate — it\'s a "soft" version of argmax.', mono: 'p_k = e^{z_k} / Σⱼ e^{z_j}' },
          { icon: '⚡', title: 'Numerical stability', body: 'e^z overflows for large z. Subtract the maximum logit first: e^(z−max)/Σe^(z−max). Mathematically identical but never overflows.', mono: 'subtract max logit first' },
          { icon: '🎯', title: 'Gradient (cross-entropy pair)', body: 'Just like sigmoid+BCE, the gradient of CE+softmax w.r.t. logit zₖ is simply pₖ−yₖ. Clean prediction-error signal for every class.', mono: '∂L/∂zₖ = pₖ − yₖ' },
        ]} />
        <Callout type="formula" mono="p_k = e^{z_k} / (e^{z_1} + e^{z_2} + ... + e^{z_K})">
          Note that classes compete: increasing z₁ not only increases p₁ but also decreases p₂ and p₃ (denominator grows). This is the "competitive" normalisation that makes softmax natural for mutual-exclusive classes.
        </Callout>
        <p className={`${BODY} mb-4`}>Drag logits below — notice how increasing one class score automatically reduces the others:</p>
        <SoftmaxViz theme={theme} />
      </div>

      {/* 4.2.3 Cross-entropy multi-class */}
      <div className={S}>
        <p className={LBL}>4.2.3 — Cross-Entropy Loss for Multi-Class Classification</p>
        <h2 className={H2}>Measuring error across all K classes</h2>
        <p className={`${BODY} mb-4`}>
          Multi-class cross-entropy extends BCE to K classes using one-hot encoded labels. Because y is one-hot, only the log-probability of the true class contributes — the loss is simply −log(p_correct).
        </p>
        <TheoryBlock title="Multi-class cross-entropy" cards={[
          { icon: '🏷️', title: 'One-hot labels', body: 'True label is encoded as a K-vector with a 1 at the correct class position and 0s elsewhere. Enables vectorised loss computation over all K classes.', mono: 'y = [0, 1, 0]  for class 2 of 3' },
          { icon: '📉', title: 'Loss formula', body: 'Only the log-probability of the true class matters: L = −log(p_k_true). Wrong class probabilities are penalised indirectly through the softmax denominator.', mono: 'L = −Σₖ yₖ·log(pₖ) = −log(p_true)' },
          { icon: '🔢', title: 'Batch loss', body: 'Average over n samples. PyTorch\'s nn.CrossEntropyLoss accepts raw logits (applies log-softmax internally) with integer class labels — no one-hot needed.', mono: 'L = −(1/n)Σᵢ log(p_i[y_i])' },
        ]} />
        <Callout type="formula" mono="L = −log(p_correct_class)">
          If the model assigns 90% to the correct class → loss = −log(0.9) = 0.105. If it assigns 10% → loss = −log(0.1) = 2.303. The model is pressured to push as much probability mass as possible onto the true class.
        </Callout>
        <DeepDive title="Why not use K separate binary losses?">
          <p className={`text-sm ${BODY} mb-2`}>You could train K separate binary classifiers (one-vs-rest). But cross-entropy + softmax is better because: (1) probabilities are jointly normalised — classes are aware of each other during training, (2) one joint optimisation problem is cleaner than K independent ones, (3) softmax gradients automatically redistribute probability mass between competing classes.</p>
        </DeepDive>
      </div>

      {/* 4.2.4 Mini-batch SGD multi-class */}
      <div className={S}>
        <p className={LBL}>4.2.4 — Training with Mini-Batch SGD</p>
        <h2 className={H2}>The standard training algorithm for neural networks</h2>
        <p className={`${BODY} mb-4`}>
          Mini-batch SGD is the workhorse of deep learning. Shuffle the dataset, split into batches of size m, process each batch with a forward+backward pass, update weights. One full pass through the data = one epoch.
        </p>
        <TheoryBlock title="Mini-batch training loop" cards={[
          { icon: '🔀', title: 'Shuffle', body: 'Shuffle the dataset at the start of each epoch. Prevents the model from learning the ordering of samples and improves generalisation.', mono: 'random.shuffle(dataset)' },
          { icon: '📦', title: 'Batch loop', body: 'Process m samples at a time. For each batch: forward pass (get logits + loss), backward pass (compute gradients), update weights.', mono: 'for xb, yb in batches: ...' },
          { icon: '⚡', title: 'GPU efficiency', body: 'GPUs are matrix-multiplication engines. Processing a batch of m samples at once is a matrix multiply (m×d)·(d×K) — far more efficient than m sequential vector multiplies.', mono: 'batch ops = one matrix multiply' },
        ]} />
        <Callout type="info" title="Choosing batch size">
          Larger batches give more accurate gradient estimates but require more memory. Smaller batches add regularising noise. The sweet spot is usually 32–256. Very large batches (&gt;4096) can hurt generalisation without careful learning rate scaling.
        </Callout>
        <DeepDive title="Learning rate scaling with batch size">
          <p className={`text-sm ${BODY} mb-2`}>When you increase batch size by factor k, the gradient estimate variance decreases by k. This means you can increase the learning rate by √k (or sometimes k) without losing stability — the "linear scaling rule" used in large-scale training.</p>
          <Callout type="formula" mono="α_large = α_base × √(m_large / m_base)">
            Example: if you double batch size from 64 to 128, try multiplying lr by √2 ≈ 1.41. Always warm up lr gradually at the start of training when using large batches.
          </Callout>
        </DeepDive>
      </div>

      {/* Code */}
      <div className={S}>
        <p className={LBL}>Python — Binary Classification (TensorFlow / Keras)</p>
        <h2 className={H2}>Single sigmoid neuron + BCE + SGD</h2>
        <p className={`${BODY} mb-3`}>Full binary classification workflow: <span className={`font-mono text-xs ${theme === 'dark' ? 'text-indigo-300' : 'text-indigo-600'}`}>Dense(1, activation='sigmoid')</span>, binary_crossentropy loss, SGD training, and threshold inference.</p>
        <CodeBlock code={PYTHON_CODE_BINARY} />
      </div>

      <div className={S}>
        <p className={LBL}>Python — SGD vs Mini-batch comparison (TensorFlow / Keras)</p>
        <h2 className={H2}>True SGD (batch_size=1) vs mini-batch (batch_size=4)</h2>
        <p className={`${BODY} mb-3`}>Side-by-side using Keras <span className={`font-mono text-xs ${theme === 'dark' ? 'text-indigo-300' : 'text-indigo-600'}`}>batch_size</span> argument — Keras handles the loop automatically.</p>
        <CodeBlock code={PYTHON_CODE_SGD} />
      </div>

      <div className={S}>
        <p className={LBL}>Python — Multi-Class (TensorFlow / Keras)</p>
        <h2 className={H2}>Dense(3, activation='softmax') + sparse_categorical_crossentropy on Iris</h2>
        <p className={`${BODY} mb-3`}>Multi-class network: 4 inputs → 3 softmax outputs, <span className={`font-mono text-xs ${theme === 'dark' ? 'text-indigo-300' : 'text-indigo-600'}`}>sparse_categorical_crossentropy</span> accepts integer labels directly — no one-hot encoding needed.</p>
        <CodeBlock code={PYTHON_CODE_MULTICLASS} />
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme === 'dark' ? 'bg-purple-500/10 border-purple-500/20' : 'bg-purple-50 border-purple-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>10 questions covering Sections 4.1 and 4.2 • +100 XP on completion</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="linear-neuron-classification" />
      </div>
    </motion.div>
  )
}
