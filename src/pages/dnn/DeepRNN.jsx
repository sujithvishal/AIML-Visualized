import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

// ── Code blocks ────────────────────────────────────────────────────────────────

const CODE_VANISHING = `import numpy as np

# ── Vanishing gradient demo: tanh backprop over T steps ──

def dtanh(x):
    return 1 - np.tanh(x) ** 2   # derivative of tanh, max value = 1

np.random.seed(42)
T   = 20          # sequence length
H   = 8           # hidden dim
Wh  = np.random.randn(H, H) * 0.5   # recurrent weight matrix

# Simulate gradient magnitude after T BPTT steps
# At each step the gradient is scaled by Wh^T * diag(dtanh(h))
# We track the spectral norm product

h     = np.random.randn(H) * 0.1
grad  = np.ones(H)
norms = [np.linalg.norm(grad)]

for t in range(T):
    h    = np.tanh(Wh @ h)
    # Jacobian: dh_t/dh_{t-1} = diag(dtanh(h)) @ Wh
    jac  = np.diag(dtanh(h)) @ Wh
    grad = jac.T @ grad          # backprop one step
    norms.append(np.linalg.norm(grad))

for t, n in enumerate(norms):
    bar = '█' * max(1, int(n * 5))
    print(f"t={t:02d}  |grad|={n:.6f}  {bar}")

# If spectral_radius(Wh) < 1  → vanishing
# If spectral_radius(Wh) > 1  → exploding
sr = np.max(np.abs(np.linalg.eigvals(Wh)))
print(f"\\nSpectral radius of Wh: {sr:.3f}")`

const CODE_LSTM = `import tensorflow as tf
import numpy as np

# ── Manual LSTM cell (single step) — shows all four gates ──

def lstm_step(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, bf, bi, bg, bo):
    """
    x_t    : (D,)  current input
    h_prev : (H,)  previous hidden state
    c_prev : (H,)  previous cell state
    Returns: (h_t, c_t)
    """
    z = np.concatenate([h_prev, x_t])   # (H+D,)

    f_t = 1 / (1 + np.exp(-(Wf @ z + bf)))   # forget gate  ∈ (0,1)
    i_t = 1 / (1 + np.exp(-(Wi @ z + bi)))   # input  gate  ∈ (0,1)
    g_t = np.tanh(Wg @ z + bg)               # cell candidate ∈ (-1,1)
    o_t = 1 / (1 + np.exp(-(Wo @ z + bo)))   # output gate  ∈ (0,1)

    c_t = f_t * c_prev + i_t * g_t           # update cell state
    h_t = o_t * np.tanh(c_t)                 # output hidden state

    return h_t, c_t

# ── Keras LSTM ──
SEQ_LEN, D, H = 10, 8, 32

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(H, return_sequences=True, input_shape=(SEQ_LEN, D)),
    tf.keras.layers.LSTM(H),                    # stack a second LSTM
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.summary()

# A Keras LSTM layer encapsulates 4 × (H×(H+D) + H) parameters
params = 4 * (H * (H + D) + H)
print(f"\\nExpected LSTM params: {params:,}")
print(f"Actual    LSTM params: {model.layers[0].count_params():,}")`

const CODE_GATES = `import numpy as np

# ── Gate equations — step-by-step illustration ──

np.random.seed(0)
D, H = 3, 4   # input_dim, hidden_dim

# Concatenated weight matrix style (common in implementations)
# W ∈ R^{4H × (H+D)},  b ∈ R^{4H}
W = np.random.randn(4 * H, H + D) * 0.1
b = np.zeros(4 * H)

def sigmoid(x): return 1 / (1 + np.exp(-x))

def lstm_step_compact(x_t, h_prev, c_prev, W, b):
    z  = np.concatenate([h_prev, x_t])     # (H+D,)
    gates = W @ z + b                      # (4H,)

    f = sigmoid(gates[0*H : 1*H])          # forget gate
    i = sigmoid(gates[1*H : 2*H])          # input  gate
    g = np.tanh( gates[2*H : 3*H])         # candidate cell
    o = sigmoid(gates[3*H : 4*H])          # output gate

    c_new = f * c_prev + i * g
    h_new = o * np.tanh(c_new)

    print(f"  forget  gate f : {np.round(f, 3)}")
    print(f"  input   gate i : {np.round(i, 3)}")
    print(f"  candidate    g : {np.round(g, 3)}")
    print(f"  output  gate o : {np.round(o, 3)}")
    print(f"  cell state c   : {np.round(c_new, 3)}")
    print(f"  hidden   h     : {np.round(h_new, 3)}")
    return h_new, c_new

x = np.random.randn(D)
h = np.zeros(H)
c = np.zeros(H)

print("=== Step 1 ===")
h, c = lstm_step_compact(x, h, c, W, b)
print("\\n=== Step 2 ===")
h, c = lstm_step_compact(np.random.randn(D), h, c, W, b)`

const CODE_GRU = `import tensorflow as tf
import numpy as np

# ── Manual GRU cell (single step) ──

def gru_step(x_t, h_prev, Wr, Wz, Wh_cand, br, bz, bh):
    """
    x_t    : (D,)
    h_prev : (H,)
    Returns: h_t (H,)
    """
    z_in = np.concatenate([h_prev, x_t])

    def sigmoid(x): return 1 / (1 + np.exp(-x))

    r_t = sigmoid(Wr     @ z_in + br)           # reset  gate ∈ (0,1)
    z_t = sigmoid(Wz     @ z_in + bz)           # update gate ∈ (0,1)

    # Candidate: only let reset gate pass h_prev info
    h_cand = np.tanh(Wh_cand @ np.concatenate([r_t * h_prev, x_t]) + bh)

    # Interpolate: z_t=1 → keep old state; z_t=0 → replace with candidate
    h_t = (1 - z_t) * h_cand + z_t * h_prev

    return h_t

# ── Keras GRU ──
SEQ_LEN, D, H = 10, 8, 32

model = tf.keras.Sequential([
    tf.keras.layers.GRU(H, return_sequences=True, input_shape=(SEQ_LEN, D)),
    tf.keras.layers.GRU(H),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.summary()

# GRU has 3 × (H×(H+D) + H) parameters — fewer than LSTM
params_gru  = 3 * (H * (H + D) + H)
params_lstm = 4 * (H * (H + D) + H)
print(f"GRU  params (expected): {params_gru:,}")
print(f"LSTM params (expected): {params_lstm:,}")`

const CODE_STACKED = `import tensorflow as tf

# ── Stacked (deep) RNN — 3 LSTM layers ──
SEQ_LEN, D = 20, 16
UNITS = [64, 64, 32]      # hidden units per layer

inputs = tf.keras.Input(shape=(SEQ_LEN, D))
x = inputs
for i, units in enumerate(UNITS):
    return_seq = (i < len(UNITS) - 1)   # all but last return full sequence
    x = tf.keras.layers.LSTM(units, return_sequences=return_seq,
                              dropout=0.2, recurrent_dropout=0.1)(x)

outputs = tf.keras.layers.Dense(1)(x)
model   = tf.keras.Model(inputs, outputs)
model.summary()

# Layer 1 receives raw features (D-dim)
# Layer 2 receives the full sequence of hidden states from layer 1 (UNITS[0]-dim)
# Layer 3 receives the full sequence of hidden states from layer 2 (UNITS[1]-dim)
# Final Dense receives only the last hidden state of layer 3`

const CODE_BIDIR = `import tensorflow as tf

# ── Bidirectional RNN ──
SEQ_LEN, D, H = 15, 10, 32

model = tf.keras.Sequential([
    # Bidirectional wraps any RNN layer; merge_mode controls how fwd + bwd are combined
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(H, return_sequences=True),
        merge_mode='concat',          # output dim = 2H
        input_shape=(SEQ_LEN, D)
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(H),      # only last time step
        merge_mode='concat',
    ),
    tf.keras.layers.Dense(3, activation='softmax'),
])
model.summary()

# Parameters: 2 × LSTM(H) = 2 × 4*(H*(H+D)+H)
# But second Bidirectional LSTM receives 2H input from first layer
# So its param count is 2 × 4*(H*(H+2H)+H) = 2 × 4*(3H²+H)
print("\\nOutput dim after Bidirectional(return_seq=True):", 2 * H)
print("Expected: each position has forward h + backward h concatenated")`

// ── Quiz ──────────────────────────────────────────────────────────────────────

const QUIZ_QUESTIONS = [
  {
    question: 'What is the PRIMARY purpose of the forget gate in an LSTM?',
    options: [
      'Scale the output hidden state',
      'Decide how much of the previous cell state to retain',
      'Produce the candidate cell value',
      'Control the input to the next layer',
    ],
    correct: 1,
    explanation: 'The forget gate f_t ∈ (0,1) multiplies the previous cell state c_{t−1}. Values near 0 erase old memory; values near 1 preserve it.',
  },
  {
    question: 'In an LSTM the cell state c_t flows through time with only:',
    options: [
      'Tanh non-linearities at every step',
      'Sigmoid activations at every step',
      'Element-wise multiplications and additions — no squashing',
      'Matrix multiplications like a standard RNN',
    ],
    correct: 2,
    explanation: 'The cell state "highway" only passes through element-wise operations (f * c + i * g), allowing gradients to flow with minimal attenuation.',
  },
  {
    question: 'A GRU has fewer parameters than an LSTM of the same hidden size because:',
    options: [
      'It uses smaller weight matrices',
      'It has only 2 gates instead of 3, and no separate cell state',
      'It does not use biases',
      'It shares weights between the reset and update gates',
    ],
    correct: 1,
    explanation: 'GRU merges the cell and hidden state into one, uses only a reset gate and an update gate (vs. forget, input, output in LSTM), giving 3× vs 4× parameter groups.',
  },
  {
    question: 'In a stacked RNN, what does each upper layer receive as its input sequence?',
    options: [
      'The raw input embeddings repeated at each layer',
      'The final hidden state of the layer below',
      'The full sequence of hidden states output by the layer below',
      'A concatenation of all lower-layer outputs',
    ],
    correct: 2,
    explanation: 'With return_sequences=True, each lower layer emits h_t for every time step. The next layer treats these as its input sequence, learning higher-level temporal patterns.',
  },
  {
    question: 'A Bidirectional LSTM with hidden size H and merge_mode="concat" produces an output vector of dimension:',
    options: ['H', '2H', 'H²', '4H'],
    correct: 1,
    explanation: 'The forward LSTM contributes H dimensions and the backward LSTM contributes H dimensions. Concatenation gives 2H.',
  },
]

// ── LSTM gate visualiser ──────────────────────────────────────────────────────

function LSTMGateVis({ theme }) {
  const dark = theme === 'dark'
  const gates = [
    { label: 'Forget gate f_t', color: dark ? '#ef4444' : '#dc2626', bg: dark ? 'bg-red-900/30' : 'bg-red-50', border: dark ? 'border-red-700' : 'border-red-300', text: dark ? 'text-red-300' : 'text-red-700', desc: 'σ(W_f · [h,x] + b_f) — how much to erase from cell' },
    { label: 'Input gate i_t', color: dark ? '#22c55e' : '#16a34a', bg: dark ? 'bg-green-900/30' : 'bg-green-50', border: dark ? 'border-green-700' : 'border-green-300', text: dark ? 'text-green-300' : 'text-green-700', desc: 'σ(W_i · [h,x] + b_i) — how much new info to write' },
    { label: 'Candidate g_t', color: dark ? '#f59e0b' : '#d97706', bg: dark ? 'bg-amber-900/30' : 'bg-amber-50', border: dark ? 'border-amber-700' : 'border-amber-300', text: dark ? 'text-amber-300' : 'text-amber-700', desc: 'tanh(W_g · [h,x] + b_g) — new candidate values ∈ (−1,1)' },
    { label: 'Output gate o_t', color: dark ? '#38bdf8' : '#0284c7', bg: dark ? 'bg-sky-900/30' : 'bg-sky-50', border: dark ? 'border-sky-700' : 'border-sky-300', text: dark ? 'text-sky-300' : 'text-sky-700', desc: 'σ(W_o · [h,x] + b_o) — which part of cell to expose' },
  ]

  return (
    <div className="grid grid-cols-2 gap-3 my-4">
      {gates.map((g, i) => (
        <motion.div
          key={i}
          className={`rounded-xl p-3 border ${g.bg} ${g.border}`}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: i * 0.08 }}
        >
          <p className={`text-xs font-bold mb-1 ${g.text}`}>{g.label}</p>
          <p className={`text-xs leading-relaxed font-mono ${dark ? 'text-slate-300' : 'text-slate-600'}`}>{g.desc}</p>
        </motion.div>
      ))}
      {/* Cell update */}
      <div className={`col-span-2 rounded-xl p-3 border mt-1 ${dark ? 'bg-violet-900/30 border-violet-700' : 'bg-violet-50 border-violet-300'}`}>
        <p className={`text-xs font-bold mb-1 ${dark ? 'text-violet-300' : 'text-violet-700'}`}>Cell update &amp; hidden state</p>
        <p className={`text-xs font-mono ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          c_t = f_t ⊙ c_{'t−1'} + i_t ⊙ g_t &nbsp;&nbsp;|&nbsp;&nbsp; h_t = o_t ⊙ tanh(c_t)
        </p>
      </div>
    </div>
  )
}

// ── GRU vs LSTM comparison table ─────────────────────────────────────────────

function ComparisonTable({ theme }) {
  const dark = theme === 'dark'
  const rows = [
    ['Gates', '3 (forget, input, output)', '2 (reset, update)'],
    ['Separate cell state', 'Yes — c_t highway', 'No — merged into h_t'],
    ['Parameters (H, D)', '4 × (H(H+D) + H)', '3 × (H(H+D) + H)'],
    ['Training speed', 'Slower (more params)', 'Faster'],
    ['Long-range memory', 'Strong', 'Comparable, slightly weaker'],
    ['When to prefer', 'Complex tasks, more data', 'Simpler tasks, speed matters'],
  ]
  return (
    <div className="overflow-x-auto my-4">
      <table className={`w-full text-sm border-collapse rounded-xl overflow-hidden ${dark ? 'text-slate-200' : 'text-slate-700'}`}>
        <thead>
          <tr className={dark ? 'bg-slate-800' : 'bg-slate-100'}>
            <th className={`px-4 py-2 text-left font-bold text-xs uppercase tracking-wide ${dark ? 'text-slate-400' : 'text-slate-500'}`}>Property</th>
            <th className={`px-4 py-2 text-left font-bold text-xs uppercase tracking-wide ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>LSTM</th>
            <th className={`px-4 py-2 text-left font-bold text-xs uppercase tracking-wide ${dark ? 'text-violet-400' : 'text-violet-600'}`}>GRU</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([prop, lstm, gru], i) => (
            <tr key={i} className={i % 2 === 0 ? (dark ? 'bg-slate-900/50' : 'bg-white') : (dark ? 'bg-slate-800/40' : 'bg-slate-50')}>
              <td className={`px-4 py-2 font-semibold text-xs ${dark ? 'text-slate-400' : 'text-slate-500'}`}>{prop}</td>
              <td className="px-4 py-2 text-xs">{lstm}</td>
              <td className="px-4 py-2 text-xs">{gru}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Bidirectional diagram ─────────────────────────────────────────────────────

function BidirDiagram({ theme }) {
  const dark = theme === 'dark'
  const tokens = ['x₁', 'x₂', 'x₃', 'x₄']

  return (
    <div className="overflow-x-auto my-4">
      <svg viewBox="0 0 480 160" className="w-full max-w-lg mx-auto block" style={{ minWidth: 320 }}>
        {/* Forward row label */}
        <text x="8" y="48" fontSize="10" fontWeight="700" fill={dark ? '#818cf8' : '#4f46e5'}>→ fwd</text>
        {/* Backward row label */}
        <text x="8" y="118" fontSize="10" fontWeight="700" fill={dark ? '#f472b6' : '#db2777'}>← bwd</text>

        {tokens.map((tok, i) => {
          const cx = 80 + i * 100
          // forward cell
          return (
            <g key={i}>
              {/* forward LSTM box */}
              <rect x={cx - 28} y="28" width="56" height="30" rx="6"
                fill={dark ? '#312e81' : '#eef2ff'} stroke={dark ? '#6366f1' : '#6366f1'} strokeWidth="1.5" />
              <text x={cx} y="48" textAnchor="middle" fontSize="10" fontWeight="700"
                fill={dark ? '#c7d2fe' : '#4338ca'}>h→{i + 1}</text>

              {/* backward LSTM box */}
              <rect x={cx - 28} y="96" width="56" height="30" rx="6"
                fill={dark ? '#500724' : '#fdf2f8'} stroke={dark ? '#f472b6' : '#db2777'} strokeWidth="1.5" />
              <text x={cx} y="116" textAnchor="middle" fontSize="10" fontWeight="700"
                fill={dark ? '#fbcfe8' : '#9d174d'}>h←{tokens.length - i}</text>

              {/* input token */}
              <text x={cx} y="160" textAnchor="middle" fontSize="11" fontWeight="700"
                fill={dark ? '#94a3b8' : '#64748b'}>{tok}</text>

              {/* arrow input → fwd */}
              <line x1={cx} y1="148" x2={cx} y2="59" stroke={dark ? '#475569' : '#94a3b8'} strokeWidth="1" markerEnd="url(#arrd)" />

              {/* arrow input → bwd */}
              <line x1={cx} y1="148" x2={cx} y2="127" stroke={dark ? '#475569' : '#94a3b8'} strokeWidth="1" markerEnd="url(#arrd)" />

              {/* forward arrow between cells */}
              {i < tokens.length - 1 && (
                <line x1={cx + 28} y1="43" x2={cx + 72} y2="43"
                  stroke={dark ? '#6366f1' : '#4f46e5'} strokeWidth="1.5" markerEnd="url(#arrfwd)" />
              )}
              {/* backward arrow between cells */}
              {i > 0 && (
                <line x1={cx - 28} y1="111" x2={cx - 72} y2="111"
                  stroke={dark ? '#f472b6' : '#db2777'} strokeWidth="1.5" markerEnd="url(#arrbwd)" />
              )}
            </g>
          )
        })}
        <defs>
          <marker id="arrd" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 z" fill={dark ? '#475569' : '#94a3b8'} />
          </marker>
          <marker id="arrfwd" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 z" fill={dark ? '#6366f1' : '#4f46e5'} />
          </marker>
          <marker id="arrbwd" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
            <path d="M0,6 L6,6 L6,0 z" fill={dark ? '#f472b6' : '#db2777'} />
          </marker>
        </defs>
      </svg>
      <p className={`text-center text-xs ${dark ? 'text-slate-400' : 'text-slate-500'}`}>
        Forward pass (blue, left→right) and backward pass (pink, right→left) run independently; outputs are concatenated at each position.
      </p>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function DeepRNN() {
  const { theme, markTopicComplete } = useApp()
  const dark = theme === 'dark'

  return (
    <div className="space-y-10">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3 mb-3">
          <span className="text-4xl">🧠</span>
          <div>
            <p className={`text-sm font-semibold uppercase tracking-widest ${dark ? 'text-violet-400' : 'text-violet-600'}`}>
              Session 10 · Deep Learning
            </p>
            <h1 className={`text-3xl font-black ${dark ? 'text-white' : 'text-slate-900'}`}>
              Deep Recurrent Neural Networks
            </h1>
          </div>
        </div>
        <p className={`text-lg leading-relaxed ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Vanilla RNNs struggle with long sequences due to vanishing gradients. <strong>LSTMs</strong> and
          <strong> GRUs</strong> solve this with gating mechanisms that explicitly control what to remember and
          forget. <strong>Stacking</strong> layers and running them <strong>bidirectionally</strong> pushes
          representational power further.
        </p>
      </motion.div>

      {/* ── Section 10.1 ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 10.1</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>LSTM</h2>
        </div>

        {/* 10.1.1 Vanishing gradient */}
        <h3 className={`text-lg font-bold mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          10.1.1 — Vanishing Gradient Problem in Vanilla RNNs
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          During BPTT the gradient at time step t flows back through every earlier step by repeated multiplication
          with the recurrent weight matrix W_h and the derivative of tanh. Since <code>|dtanh(x)| ≤ 1</code>,
          gradients typically shrink exponentially, making it impossible for the network to learn dependencies
          that span many time steps.
        </p>

        <Callout type="warning" title="Why tanh makes it worse">
          dtanh(x) = 1 − tanh²(x) peaks at 1 (when x = 0) and quickly approaches 0 for large |x|.
          Over T steps the gradient magnitude ≈ (spectral_radius(W_h) × avg_dtanh)^T.
          With a spectral radius less than 1, this collapses toward zero.
        </Callout>

        <CodeBlock code={CODE_VANISHING} language="python" title="vanishing_gradient.py" />

        <TheoryBlock items={[
          { title: 'Vanishing gradient', content: 'Gradient signal for tokens far in the past becomes negligible. The model cannot update weights responsible for long-range dependencies.' },
          { title: 'Exploding gradient', content: 'If spectral_radius(W_h) > 1, gradients grow exponentially. Fixed by gradient clipping but not by architecture.' },
          { title: 'Practical consequence', content: 'Vanilla RNNs effectively have a "memory horizon" of ~10–20 steps. Anything beyond is forgotten during training.' },
          { title: 'Solution', content: 'LSTM and GRU introduce additive update paths for the state — gradients can flow through them without repeated multiplication.' },
        ]} />

        {/* 10.1.2 LSTM cell state and gates */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          10.1.2 — LSTM: Cell State and Gates
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The key innovation of the LSTM (Hochreiter &amp; Schmidhuber, 1997) is a separate
          <strong> cell state</strong> c_t — a "memory conveyor belt" that passes through time with only
          element-wise multiplications and additions, no squashing. This creates a near-constant error
          carousel that lets gradients flow unimpeded over hundreds of steps.
        </p>

        <Callout type="info" title="Two states, not one">
          An LSTM has <strong>two</strong> state vectors per layer: the <strong>cell state c_t</strong>
          (long-term memory, not directly output) and the <strong>hidden state h_t</strong> (short-term,
          exposed as output). A vanilla RNN has only h_t.
        </Callout>

        <CodeBlock code={CODE_LSTM} language="python" title="lstm_cell.py" />

        {/* 10.1.3 Gate equations */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          10.1.3 — Input Gate, Forget Gate, Output Gate — Equations
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          All four gates read the same concatenated vector <code>[h_{'t−1'}; x_t]</code> but use
          independent weight matrices, so each specialises during training.
        </p>

        <LSTMGateVis theme={theme} />

        <CodeBlock code={CODE_GATES} language="python" title="lstm_gates.py" />

        <DeepDive title="Why sigmoid for gates, tanh for candidate?">
          <TheoryBlock items={[
            { title: 'Sigmoid → gating', content: 'Outputs in (0,1) act as soft on/off switches. A value of 0 blocks information completely; 1 passes it unchanged.' },
            { title: 'Tanh → value', content: 'Outputs in (−1,1) represent a signed value — positive means "write this positive info", negative means "write this negative info".' },
            { title: 'Cell highway', content: 'c_t = f⊙c + i⊙g is purely additive (after gating). The gradient ∂L/∂c_{t−1} = ∂L/∂c_t ⊙ f_t, which is close to 1 when the forget gate is open.' },
          ]} />
        </DeepDive>

        {/* 10.1.4 Gated memory intuition */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          10.1.4 — Gated Memory Cell Intuition
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Think of the LSTM cell as a <strong>register with read/write/erase controls</strong>:
        </p>

        <TheoryBlock cols={2} items={[
          { title: '🗑️ Forget gate — erase', content: 'Sets c_{t−1} entries to zero that are no longer relevant (e.g. gender of a subject we\'ve moved past in a sentence).' },
          { title: '✏️ Input gate — write', content: 'Selectively adds new information from the current input. Works with the candidate g_t to determine what and how much to write.' },
          { title: '📖 Cell state — store', content: 'The actual long-term memory. Flows through time almost unchanged when the forget gate is near 1 and input gate near 0.' },
          { title: '📤 Output gate — read', content: 'Controls which parts of c_t are visible as h_t to the next layer. Allows the cell to hold information without immediately exposing it.' },
        ]} />

        <Callout type="analogy" title="Analogy: working memory vs long-term memory">
          h_t is like working memory — actively used right now. c_t is long-term memory — stored but not
          always active. The gates decide when to move information between them.
        </Callout>
      </motion.div>

      {/* ── Section 10.2 ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-violet-500/5 border-violet-500/20' : 'bg-violet-50 border-violet-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-violet-400' : 'text-violet-600'}`}>Section 10.2</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>GRU, Stacked and Bidirectional RNNs</h2>
        </div>

        {/* 10.2.1 GRU reset & update gates */}
        <h3 className={`text-lg font-bold mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          10.2.1 — GRU: Reset Gate and Update Gate
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The Gated Recurrent Unit (Cho et al., 2014) achieves similar long-range memory to the LSTM with
          only <strong>two gates</strong> and no separate cell state — merging everything into a single
          hidden state vector.
        </p>

        <TheoryBlock items={[
          { title: 'Reset gate r_t', content: 'r_t = σ(W_r · [h_{t−1}; x_t] + b_r). Controls how much of the previous hidden state is used when forming the candidate. r≈0 lets the GRU forget the past completely.' },
          { title: 'Update gate z_t', content: 'z_t = σ(W_z · [h_{t−1}; x_t] + b_z). Interpolates between old state and candidate: h_t = (1−z)⊙h̃_t + z⊙h_{t−1}. z≈1 → copy old; z≈0 → use new.' },
        ]} />

        {/* 10.2.2 Candidate hidden state */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          10.2.2 — Candidate Hidden State in GRU
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The candidate h̃_t is the <em>proposed</em> new hidden state. It is computed with the reset gate
          masking h_{'t−1'}, so when r≈0 the candidate ignores history and acts like a fresh feedforward
          computation on x_t alone.
        </p>

        <Callout type="formula" title="GRU equations">
          r_t = σ(W_r · [h_{'t−1'}; x_t]) &nbsp; — reset gate<br />
          z_t = σ(W_z · [h_{'t−1'}; x_t]) &nbsp; — update gate<br />
          h̃_t = tanh(W_h · [r_t ⊙ h_{'t−1'}; x_t]) &nbsp; — candidate<br />
          h_t = (1 − z_t) ⊙ h̃_t + z_t ⊙ h_{'t−1'} &nbsp; — new state
        </Callout>

        <CodeBlock code={CODE_GRU} language="python" title="gru_cell.py" />

        {/* 10.2.3 LSTM vs GRU */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          10.2.3 — LSTM vs GRU: Comparison
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Both architectures address the vanishing gradient problem. Choice between them is largely empirical
          — GRU trains faster; LSTM has slightly more expressive power on tasks with rich long-range structure.
        </p>

        <ComparisonTable theme={theme} />

        <DeepDive title="Which should I pick in practice?">
          <TheoryBlock items={[
            { title: 'Start with GRU', content: 'Fewer parameters means faster iteration. GRU is often as good as LSTM for text classification, time-series forecasting, and short-sequence tasks.' },
            { title: 'Try LSTM for long documents', content: 'The explicit cell state gives LSTM a slight edge on very long-range dependencies (e.g. document-level NLP, long time series).' },
            { title: 'Both are obsolete for NLP?', content: 'For NLP, Transformers (self-attention) have largely superseded LSTMs/GRUs. But gated RNNs remain competitive for streaming, online, and small-data settings.' },
          ]} />
        </DeepDive>

        {/* 10.2.4 Stacked RNNs */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          10.2.4 — Stacked (Deep) RNN Architectures
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Just as stacking convolutional layers builds hierarchical visual features, stacking recurrent layers
          builds hierarchical temporal features. Lower layers capture local patterns; upper layers integrate
          over longer contexts.
        </p>

        <TheoryBlock items={[
          { title: 'How to stack', content: 'Set return_sequences=True on all but the last RNN layer. The output sequence of layer l becomes the input sequence of layer l+1.' },
          { title: 'Depth vs width', content: '2–4 stacked layers is typical. Very deep stacks (>4) rarely improve over shallower but wider layers and are harder to train.' },
          { title: 'Dropout between layers', content: 'Standard dropout on inputs between layers (the dropout= argument) and recurrent dropout on the hidden-to-hidden connections (recurrent_dropout=).' },
          { title: 'Residual connections', content: 'Add skip connections from layer l to layer l+2 for very deep stacks, just as in ResNet. Keras supports this via functional API.' },
        ]} />

        <div className="mt-4">
          <CodeBlock code={CODE_STACKED} language="python" title="stacked_rnn.py" />
        </div>

        {/* 10.2.5 Bidirectional RNNs */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          10.2.5 — Bidirectional RNNs: Forward and Backward Passes
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          A unidirectional RNN only sees past context at each position. A <strong>Bidirectional RNN</strong>
          runs two independent RNNs over the sequence — one left-to-right, one right-to-left — then merges
          their outputs. Every position now has access to the full surrounding context.
        </p>

        <BidirDiagram theme={theme} />

        <TheoryBlock cols={2} items={[
          {
            title: '→ Forward pass',
            content: 'Runs left to right. At position t, h→_t summarises tokens x₁…x_t — all past context.',
          },
          {
            title: '← Backward pass',
            content: 'Runs right to left. At position t, h←_t summarises tokens x_t…x_T — all future context.',
          },
          {
            title: 'Merge modes',
            content: '"concat" (default, doubles dim), "sum", "mul", "ave". Concat is most expressive; sum/ave reduce dim.',
          },
          {
            title: 'When to use',
            content: 'Text classification, NER, POS tagging — any task where the full sequence is available at inference. Not suitable for real-time / autoregressive generation.',
          },
        ]} />

        <div className="mt-4">
          <CodeBlock code={CODE_BIDIR} language="python" title="bidirectional_rnn.py" />
        </div>

        <Callout type="info" title="Bidirectional in sequence-to-sequence">
          In an encoder–decoder model the <strong>encoder</strong> is often bidirectional (full sequence
          available), while the <strong>decoder</strong> must remain unidirectional (generates left-to-right,
          future tokens not yet available).
        </Callout>
      </motion.div>

      {/* ── Quiz ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-cyan-500/5 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-cyan-400' : 'text-cyan-600'}`}>Knowledge Check</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Quiz — Sessions 10.1 &amp; 10.2</h2>
        </div>
        <Quiz
          questions={QUIZ_QUESTIONS}
          onComplete={() => markTopicComplete('deep-rnn')}
        />
      </motion.div>
    </div>
  )
}
