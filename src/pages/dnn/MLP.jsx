import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

// ─── Static data (module-level to avoid useEffect dep warnings) ───────────────

const MLP_THEORY = [
  {
    icon: '🔗',
    title: 'Why Multiple Layers?',
    body: 'Each layer learns a different level of abstraction. Layer 1 detects simple patterns; layer 2 combines them into complex ones. This is how XOR — unsolvable by a Perceptron — becomes trivial.',
  },
  {
    icon: '⚡',
    title: 'Activation Functions',
    body: 'Non-linear functions applied after each weighted sum. Without them, stacking layers is mathematically identical to one layer — the whole stack collapses into a single linear transform.',
    mono: 'ReLU(x) = max(0, x)',
  },
  {
    icon: '🔄',
    title: 'Forward Propagation',
    body: 'Data flows layer-by-layer: input → hidden layers → output. At every neuron: compute z = Wx+b, then apply activation f(z). The final layer outputs the prediction.',
    mono: 'h = f(Wx + b)',
  },
  {
    icon: '⬅️',
    title: 'Backpropagation',
    body: 'The chain rule in reverse — starting from the loss, compute ∂Loss/∂w for every weight. Gradients flow backwards from output to input, enabling every weight to be updated.',
    mono: '∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w',
  },
  {
    icon: '🎯',
    title: 'Loss Functions',
    body: 'The loss measures prediction error. MSE is used for regression; Binary Cross-Entropy for classification. The optimizer minimizes this loss over many training epochs.',
    mono: 'BCE = −[y·log(ŷ) + (1−y)·log(1−ŷ)]',
  },
  {
    icon: '🚀',
    title: 'Universal Approximation',
    body: 'An MLP with at least one hidden layer and a non-linear activation can approximate any continuous function — given enough neurons. This is the theoretical foundation of deep learning.',
  },
]

const PYTHON_CODE = `import tensorflow as tf
import numpy as np

# XOR problem — not solvable by a single Perceptron!
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)

# MLP with 2 hidden layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=1000, verbose=0)

preds = (model.predict(X) > 0.5).astype(int).flatten()
print("XOR Predictions:", preds)
# Output: XOR Predictions: [0 1 1 0] ✓

# Check model structure
model.summary()
# Layer 1:  Dense(4, relu)  → 4×2+4 = 12 parameters
# Layer 2:  Dense(4, relu)  → 4×4+4 = 20 parameters
# Layer 3:  Dense(1, sigmoid) → 1×4+1 =  5 parameters
# Total: 37 parameters`

const QUIZ_QUESTIONS = [
  {
    question: 'What advantage does an MLP have over a single Perceptron?',
    options: [
      'It trains faster',
      'It can learn non-linearly separable patterns',
      'It uses less memory',
      'It requires no activation function',
    ],
    correct: 1,
    explanation: 'An MLP with hidden layers can learn non-linear decision boundaries, solving problems like XOR that a single Perceptron cannot.',
  },
  {
    question: 'What does ReLU(x) output when x = -3?',
    options: ['-3', '0', '3', '0.05'],
    correct: 1,
    explanation: 'ReLU(x) = max(0, x). For x = -3, the output is 0. ReLU "turns off" negative values, introducing sparsity.',
  },
  {
    question: 'Which activation function suffers from the vanishing gradient problem?',
    options: ['ReLU', 'Leaky ReLU', 'Sigmoid', 'None of the above'],
    correct: 2,
    explanation: 'Sigmoid saturates near 0 or 1 — gradients become near-zero, causing earlier layers to learn extremely slowly in deep networks.',
  },
  {
    question: 'In forward propagation, a neuron computes:',
    options: [
      'Only the activation function',
      'Weighted sum (z) → activation function → output',
      'The gradient of the loss',
      'The output directly from inputs',
    ],
    correct: 1,
    explanation: 'Each neuron: (1) computes z = Σ(wᵢxᵢ)+b, (2) applies activation f(z), (3) passes result to next layer.',
  },
  {
    question: 'Tanh outputs values in the range:',
    options: ['[0, 1]', '[-1, 1]', '[-∞, +∞]', '[0, +∞]'],
    correct: 1,
    explanation: 'Tanh is a scaled sigmoid that outputs values between -1 and 1, making it zero-centered — better for gradient flow than sigmoid.',
  },
]

const ACTIVATION_FNS = {
  ReLU:         { fn: x => Math.max(0, x),                  color: '#6366F1', formula: 'f(x) = max(0, x)',               pros: 'Fast, no vanishing gradient',    cons: '"Dying ReLU" — dead neurons if always 0' },
  Sigmoid:      { fn: x => 1 / (1 + Math.exp(-x)),          color: '#8B5CF6', formula: 'f(x) = 1 / (1 + e⁻ˣ)',           pros: 'Output in (0,1) — great for probability',  cons: 'Vanishing gradient for |x| > 3' },
  Tanh:         { fn: x => Math.tanh(x),                    color: '#06B6D4', formula: 'f(x) = tanh(x)',                 pros: 'Zero-centered, stronger gradients',cons: 'Still saturates at extremes' },
  'Leaky ReLU': { fn: x => x >= 0 ? x : 0.1 * x,           color: '#10B981', formula: 'f(x) = x if x>0, else 0.1x',    pros: 'Fixes dying ReLU problem',       cons: 'Extra hyperparameter (slope α)' },
}

const MLP_FP_LAYERS = [
  { neurons: 2, label: 'Input',    x: 0.1  },
  { neurons: 4, label: 'Hidden 1', x: 0.37 },
  { neurons: 3, label: 'Hidden 2', x: 0.63 },
  { neurons: 1, label: 'Output',   x: 0.9  },
]

const STEP_INFO = [
  { title: 'Input Layer',    desc: 'Raw features enter the network (e.g., [0, 1] for XOR). No computation — just data.' },
  { title: 'Hidden Layer 1', desc: 'Computes z = Wx+b, then ReLU(z). Learns simple feature combinations.' },
  { title: 'Hidden Layer 2', desc: 'Learns complex, non-linear compositions of Hidden 1 features.' },
  { title: 'Output Layer',   desc: 'Final neuron: sigmoid gives P(y=1). Threshold at 0.5 → binary prediction.' },
]

const BACKPROP_STEPS = [
  { step: '1', color: '#EF4444', title: 'Compute Loss',      desc: 'Compare ŷ to y using BCE or MSE. This scalar measures how wrong the network is.' },
  { step: '2', color: '#F97316', title: '∂L/∂ŷ',            desc: 'Differentiate loss w.r.t. output. Starting gradient for the backward pass.' },
  { step: '3', color: '#EAB308', title: '∂ŷ/∂z (output)',   desc: 'Chain through sigmoid in output layer. Gradient flows to last hidden layer.' },
  { step: '4', color: '#22C55E', title: 'Through hidden layers', desc: 'Chain rule continues backwards. ∂z/∂w = x, so ∂L/∂w = δ · x.' },
  { step: '5', color: '#6366F1', title: 'Update Weights',    desc: 'w ← w − α · ∂L/∂w for every weight. Adam adapts α per parameter.' },
]

const WHEN_TO_USE = [
  { model: 'Logistic Regression', linearity: '✅ Linear only', layers: '0 hidden', best: 'Simple binary classification' },
  { model: 'Single Perceptron',   linearity: '✅ Linear only', layers: '0 hidden', best: 'AND/OR gates, linearly separable' },
  { model: 'Shallow MLP (1 layer)', linearity: '⚠️ Mild non-linear', layers: '1 hidden', best: 'XOR, simple non-linear problems' },
  { model: 'Deep MLP (3+ layers)', linearity: '🚀 Highly non-linear', layers: '3+ hidden', best: 'Images, text, tabular data' },
  { model: 'CNN / RNN',           linearity: '🚀 + spatial/sequential', layers: 'Many',  best: 'Images, sequences, NLP' },
]

// ─── Activation Function Graph ─────────────────────────────────────────────

function ActivationGraph({ theme }) {
  const [active, setActive] = useState('ReLU')
  const canvasRef = useRef(null)
  const W = 380, H = 200

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)
    const { fn, color } = ACTIVATION_FNS[active]
    const toX = v => ((v + 4) / 8) * W
    const toY = v => H / 2 - (v / 2) * (H * 0.38)

    // Grid
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.05)'
    ctx.lineWidth = 1
    for (let x = 0; x <= W; x += W / 8) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke() }
    for (let y = 0; y <= H; y += H / 4) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke() }

    // Axes
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke()

    // Curve
    ctx.beginPath()
    for (let px = 0; px <= W; px++) {
      const x = (px / W) * 8 - 4
      const y = fn(x)
      if (px === 0) ctx.moveTo(px, toY(y)); else ctx.lineTo(px, toY(y))
    }
    ctx.strokeStyle = color
    ctx.lineWidth = 3
    ctx.stroke()

    // Axis labels
    ctx.fillStyle = theme === 'dark' ? '#6B7280' : '#9CA3AF'
    ctx.font = '10px monospace'
    ctx.textAlign = 'center'
    for (let v = -3; v <= 3; v++) {
      if (v !== 0) ctx.fillText(v, toX(v), H / 2 + 12)
    }
  }, [active, theme])

  const info = ACTIVATION_FNS[active]

  return (
    <div>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(ACTIVATION_FNS).map(([name, { color }]) => (
          <button key={name} onClick={() => setActive(name)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all border ${
              active === name
                ? 'text-white border-transparent'
                : theme === 'dark' ? 'border-white/10 text-gray-400 hover:text-gray-200' : 'border-gray-200 text-gray-500 hover:text-gray-700'
            }`}
            style={active === name ? { background: color } : {}}
          >
            {name}
          </button>
        ))}
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border mb-3 ${
          theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'
        }`}
        style={{ maxWidth: W, height: H }}
      />
      <p className="font-mono text-sm text-center font-medium mb-3" style={{ color: info.color }}>
        {info.formula}
      </p>
      <div className={`grid grid-cols-2 gap-3 text-xs rounded-xl p-3 border ${
        theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'
      }`}>
        <div>
          <span className={`font-semibold block mb-1 ${theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600'}`}>✅ Pros</span>
          <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>{info.pros}</span>
        </div>
        <div>
          <span className={`font-semibold block mb-1 ${theme === 'dark' ? 'text-amber-400' : 'text-amber-600'}`}>⚠️ Cons</span>
          <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>{info.cons}</span>
        </div>
      </div>
    </div>
  )
}

// ─── MLP Forward Propagation Animation ────────────────────────────────────

function MLPForwardProp({ theme }) {
  const canvasRef = useRef(null)
  const [step, setStep] = useState(-1)
  const [running, setRunning] = useState(false)
  const W = 380, H = 260
  const layers = MLP_FP_LAYERS

  const getPos = (li, ni, total) => ({
    x: layers[li].x * W,
    y: (ni + 1) / (total + 1) * H,
  })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    // Connections
    layers.forEach((layer, li) => {
      if (li >= layers.length - 1) return
      const next = layers[li + 1]
      for (let ni = 0; ni < layer.neurons; ni++) {
        for (let nj = 0; nj < next.neurons; nj++) {
          const a = getPos(li, ni, layer.neurons)
          const b = getPos(li + 1, nj, next.neurons)
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y)
          ctx.strokeStyle = step >= li
            ? 'rgba(99,102,241,0.45)'
            : theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.07)'
          ctx.lineWidth = step >= li ? 1.5 : 0.8
          ctx.stroke()
        }
      }
    })

    // Neurons
    layers.forEach((layer, li) => {
      for (let ni = 0; ni < layer.neurons; ni++) {
        const pos = getPos(li, ni, layer.neurons)
        const active = step >= li

        if (active) {
          ctx.beginPath(); ctx.arc(pos.x, pos.y, 18, 0, Math.PI * 2)
          ctx.fillStyle = 'rgba(99,102,241,0.15)'; ctx.fill()
        }

        ctx.beginPath(); ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2)
        ctx.fillStyle = active
          ? li === 0 ? '#06B6D4' : li === layers.length - 1 ? '#10B981' : '#6366F1'
          : theme === 'dark' ? '#1E293B' : '#F1F5F9'
        ctx.fill()
        ctx.strokeStyle = active ? 'white' : theme === 'dark' ? '#334155' : '#CBD5E1'
        ctx.lineWidth = 2; ctx.stroke()
      }

      // Layer label
      ctx.fillStyle = theme === 'dark' ? '#6B7280' : '#9CA3AF'
      ctx.font = '9px sans-serif'; ctx.textAlign = 'center'
      ctx.fillText(layer.label, layer.x * W, H - 4)

      // Step annotation
      if (step === li) {
        const anns = ['features in', 'z=Wx+b → ReLU', 'z=Wx+b → ReLU', 'z=Wx+b → σ']
        ctx.fillStyle = '#6366F1'; ctx.font = 'bold 9px monospace'
        ctx.fillText(anns[li] || '', layer.x * W, 13)
      }
    })
    ctx.textAlign = 'left'
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, theme])

  const runAnimation = async () => {
    if (running) return
    setRunning(true); setStep(-1)
    const delay = ms => new Promise(r => setTimeout(r, ms))
    for (let i = 0; i < layers.length; i++) { setStep(i); await delay(750) }
    setRunning(false)
  }

  return (
    <div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border mb-4 ${
          theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'
        }`}
        style={{ maxWidth: W, height: H }}
      />
      <div className="flex items-center gap-3 mb-4">
        <button onClick={runAnimation} disabled={running}
          className="px-5 py-2.5 rounded-xl bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 text-white text-sm font-medium transition-colors">
          {running ? '⚡ Propagating...' : '▶ Run Forward Pass'}
        </button>
        <button onClick={() => setStep(-1)}
          className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
            theme === 'dark' ? 'bg-slate-800 text-gray-400 hover:text-white' : 'bg-gray-100 text-gray-500 hover:text-gray-700'
          }`}>Reset</button>
      </div>
      <AnimatePresence>
        {step >= 0 && (
          <motion.div key={step}
            initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className={`p-4 rounded-xl border ${
              theme === 'dark' ? 'bg-indigo-500/10 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'
            }`}
          >
            <p className={`font-semibold text-sm mb-1 ${theme === 'dark' ? 'text-indigo-300' : 'text-indigo-700'}`}>
              Layer {step + 1}: {STEP_INFO[step]?.title}
            </p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
              {STEP_INFO[step]?.desc}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ─── Backpropagation Step Visualizer ─────────────────────────────────────

function BackpropVisualizer({ theme }) {
  const [activeStep, setActiveStep] = useState(null)

  return (
    <div>
      <div className="space-y-2 mb-4">
        {BACKPROP_STEPS.map((s, i) => (
          <motion.button key={i} whileHover={{ x: 4 }}
            onClick={() => setActiveStep(activeStep === i ? null : i)}
            className={`w-full text-left flex items-center gap-3 p-3 rounded-xl border transition-all ${
              activeStep === i
                ? theme === 'dark' ? 'border-white/20 bg-white/5' : 'border-gray-300 bg-gray-50'
                : theme === 'dark' ? 'border-white/8 bg-transparent hover:bg-white/5' : 'border-gray-200 bg-transparent hover:bg-gray-50'
            }`}
          >
            <span className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white shrink-0"
              style={{ background: s.color }}>
              {s.step}
            </span>
            <span className={`font-mono text-sm font-medium ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              {s.title}
            </span>
          </motion.button>
        ))}
      </div>
      <AnimatePresence>
        {activeStep !== null && (
          <motion.div
            key={activeStep}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className={`overflow-hidden rounded-xl border p-4 ${
              theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'
            }`}
          >
            <p className="font-semibold text-sm mb-1" style={{ color: BACKPROP_STEPS[activeStep].color }}>
              Step {BACKPROP_STEPS[activeStep].step}: {BACKPROP_STEPS[activeStep].title}
            </p>
            <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
              {BACKPROP_STEPS[activeStep].desc}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ─── Main Export ────────────────────────────────────────────────────────────

export default function MLP() {
  const { theme } = useApp()
  const section = `rounded-2xl border p-6 mb-6 ${theme === 'dark' ? 'bg-slate-900/80 border-white/10' : 'bg-white border-gray-200'}`
  const label   = `text-xs font-semibold uppercase tracking-wider mb-3 ${theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}`
  const h2      = `text-lg font-semibold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`
  const body    = `text-sm leading-relaxed mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`
  const th      = `text-xs font-semibold uppercase tracking-wider px-4 py-3 text-left ${theme === 'dark' ? 'text-gray-400 bg-slate-800' : 'text-gray-500 bg-gray-50'}`
  const td      = `px-4 py-3 text-sm border-t ${theme === 'dark' ? 'text-gray-300 border-white/5' : 'text-gray-700 border-gray-100'}`

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-3xl">

      {/* ── Hero ── */}
      <div className="mb-8">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-500/20 border border-cyan-500/30 mb-4">
          <span className="text-xs text-cyan-400 font-medium">Deep Neural Networks · Architecture</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Multi-Layer <span className="gradient-text">Perceptron</span>
        </h1>
        <p className={`text-lg mb-6 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Stack layers of neurons to learn non-linear patterns — the foundational architecture behind every modern deep learning model.
        </p>
        <Callout type="analogy" title="The Key Insight">
          A single Perceptron draws one straight line. An MLP folds and bends that line through multiple layers — until it can carve any shape through the data.
        </Callout>
      </div>

      {/* ── Theory ── */}
      <TheoryBlock title="Core Concepts" cards={MLP_THEORY} />

      {/* ── Architecture Overview ── */}
      <div className={section}>
        <p className={label}>Architecture</p>
        <h3 className={h2}>Input → Hidden Layers → Output</h3>
        <p className={body}>
          An MLP is fully connected — every neuron in layer L connects to every neuron in layer L+1.
          Adding more hidden layers (depth) and more neurons per layer (width) increases the model&apos;s capacity.
        </p>
        <Callout type="formula" title="Neuron Computation" mono="z = w₁x₁ + w₂x₂ + … + wₙxₙ + b     →     output = f(z)">
          Every neuron applies a weighted sum of its inputs, adds a bias, then transforms the result through a non-linear activation function f.
        </Callout>
        <div className={`overflow-x-auto rounded-xl border mb-4 ${theme === 'dark' ? 'border-white/10' : 'border-gray-200'}`}>
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className={th}>Architecture</th>
                <th className={th}>Non-linearity</th>
                <th className={th}>Hidden Layers</th>
                <th className={th}>Best For</th>
              </tr>
            </thead>
            <tbody>
              {WHEN_TO_USE.map((row, i) => (
                <tr key={i} className={i % 2 === 0 && theme === 'dark' ? 'bg-slate-800/30' : ''}>
                  <td className={`${td} font-medium font-mono text-xs`}>{row.model}</td>
                  <td className={td}>{row.linearity}</td>
                  <td className={td}>{row.layers}</td>
                  <td className={td}>{row.best}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <DeepDive title="How many layers and neurons should you use?">
          <p className={`text-sm mb-3 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            There is no universal rule — but these guidelines help:
          </p>
          <ul className={`text-sm space-y-2 list-disc pl-5 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            <li><strong className={theme === 'dark' ? 'text-gray-200' : 'text-gray-800'}>Start small.</strong> 1–2 hidden layers solve the majority of tabular data problems.</li>
            <li><strong className={theme === 'dark' ? 'text-gray-200' : 'text-gray-800'}>Wider vs. deeper.</strong> Wider (more neurons) captures more features per layer. Deeper (more layers) learns more abstract hierarchies.</li>
            <li><strong className={theme === 'dark' ? 'text-gray-200' : 'text-gray-800'}>Powers of 2.</strong> Using 32, 64, 128, 256 neurons exploits GPU memory alignment.</li>
            <li><strong className={theme === 'dark' ? 'text-gray-200' : 'text-gray-800'}>Overfitting check.</strong> If training accuracy {'>>'} validation accuracy, reduce network size or add Dropout.</li>
          </ul>
        </DeepDive>
      </div>

      {/* ── Activation Functions ── */}
      <div className={section}>
        <p className={label}>Activation Functions</p>
        <h3 className={h2}>The non-linearity that makes deep learning possible</h3>
        <p className={body}>
          Without activation functions, stacking N linear layers is mathematically equivalent to one linear layer — no matter how deep the network.
          Click each tab to compare shapes, formulas, pros and cons.
        </p>
        <Callout type="warning" title="Vanishing Gradient — Why It Matters">
          When gradients are multiplied through many sigmoid layers they shrink exponentially. Deep networks trained with sigmoid rarely converge — use ReLU or its variants instead.
        </Callout>
        <ActivationGraph theme={theme} />
        <DeepDive title="What is the Dying ReLU problem?">
          <p className={`text-sm mb-3 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            If a ReLU neuron always receives negative input (z {'<'} 0), its gradient is permanently zero — it can never recover during training. This is the <em>dying ReLU</em> problem.
          </p>
          <Callout type="success" title="Fix: Use Leaky ReLU or He Initialisation">
            Leaky ReLU passes a small fraction (0.1x) for negative inputs, keeping gradients alive. He weight initialisation keeps activations in a healthy range from the start.
          </Callout>
        </DeepDive>
      </div>

      {/* ── Forward Propagation ── */}
      <div className={section}>
        <p className={label}>Forward Propagation — Step by Step</p>
        <h3 className={h2}>See how a signal travels through each layer</h3>
        <p className={body}>
          Click <strong>Run Forward Pass</strong> to watch data propagate from the input layer, through two hidden layers, to the output neuron.
          Each layer&apos;s annotation shows what computation is performed.
        </p>
        <MLPForwardProp theme={theme} />
        <Callout type="info" title="XOR — Why MLPs Solve What Perceptrons Can't">
          XOR(0,1)=1, XOR(1,0)=1, XOR(0,0)=0, XOR(1,1)=0. No single straight line separates these classes. A hidden layer bends the space — making the classes linearly separable in the transformed representation.
        </Callout>
      </div>

      {/* ── Backpropagation ── */}
      <div className={section}>
        <p className={label}>Backpropagation</p>
        <h3 className={h2}>How the network learns — gradients in reverse</h3>
        <p className={body}>
          After a forward pass, the loss tells us how wrong the prediction was. Backpropagation uses the chain rule to compute how much each weight contributed to that error,
          then gradient descent nudges every weight to reduce the loss.
        </p>
        <Callout type="formula" title="Chain Rule" mono="∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w">
          The chain rule decomposes the gradient of the loss with respect to any weight into a product of local gradients — computed layer by layer, backwards.
        </Callout>
        <BackpropVisualizer theme={theme} />
        <DeepDive title="Adam Optimizer — Why not plain SGD?">
          <p className={`text-sm mb-3 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            SGD updates every weight with the same learning rate α. Adam (Adaptive Moment Estimation) maintains a running average of both gradients and their squared magnitudes, adapting α per parameter:
          </p>
          <div className={`font-mono text-xs p-3 rounded-xl mb-3 ${theme === 'dark' ? 'bg-slate-800 text-cyan-300' : 'bg-gray-100 text-cyan-700'}`}>
            {'m ← β₁·m + (1−β₁)·g      // first moment (mean)\n'}
            {'v ← β₂·v + (1−β₂)·g²     // second moment (variance)\n'}
            {'w ← w − α · m̂ / (√v̂ + ε)  // adaptive step'}
          </div>
          <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            Default: α=0.001, β₁=0.9, β₂=0.999. Adam is the standard choice for most MLP and deep learning tasks.
          </p>
        </DeepDive>
      </div>

      {/* ── Python Example ── */}
      <div className={section}>
        <p className={label}>Python Example — Solving XOR with TensorFlow</p>
        <h3 className={h2}>A complete MLP in ~10 lines of code</h3>
        <p className={body}>This model stacks two ReLU hidden layers and a sigmoid output to solve the XOR problem — a benchmark that requires non-linear learning.</p>
        <CodeBlock code={PYTHON_CODE} />
        <Callout type="success" title="Why 1000 epochs?">
          XOR is a tiny dataset (4 points) — the network needs many passes to converge. For real datasets, use early stopping with a validation set to avoid overfitting.
        </Callout>
      </div>

      {/* ── Real-world Use Cases ── */}
      <div className={section}>
        <p className={label}>Real-World Use Cases</p>
        <h3 className={h2}>Where MLPs are used today</h3>
        <div className="grid grid-cols-1 gap-3">
          {[
            { icon: '🏦', title: 'Credit Scoring',       desc: 'Banks use MLPs on tabular data (age, income, debt ratio) to predict loan default probability.' },
            { icon: '🏥', title: 'Medical Diagnosis',    desc: 'Lab results + patient history fed into an MLP to predict disease risk or triage priority.' },
            { icon: '🛒', title: 'Recommendation',       desc: 'User embeddings passed through hidden layers to predict item ratings or click-through.' },
            { icon: '🔤', title: 'Feature Extraction',   desc: 'The "head" of a larger model — a CNN or transformer produces embeddings, then an MLP classifies.' },
          ].map((uc, i) => (
            <motion.div key={i} whileHover={{ x: 4 }}
              className={`flex gap-4 p-4 rounded-xl border ${
                theme === 'dark' ? 'bg-slate-800/40 border-white/8' : 'bg-gray-50 border-gray-200'
              }`}
            >
              <span className="text-2xl shrink-0">{uc.icon}</span>
              <div>
                <p className={`font-semibold text-sm mb-0.5 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{uc.title}</p>
                <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{uc.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* ── Quiz ── */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme === 'dark' ? 'bg-cyan-500/10 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>5 questions · +100 XP · Neural Network Starter badge</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="mlp" />
      </div>

    </motion.div>
  )
}
