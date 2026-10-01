import { useState } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

// ── Code blocks ────────────────────────────────────────────────────────────────

const CODE_ADDITIVE = `import numpy as np

# ── Bahdanau (additive) attention — step-by-step ──
# Decoder query q ∈ R^H, encoder keys K ∈ R^{T×H}
# Score(q, k_j) = v^T · tanh(W_q · q + W_k · k_j)

np.random.seed(0)
T, H, A = 5, 8, 6          # seq_len, hidden_dim, attention_dim

# Simulated encoder hidden states (keys = values in Bahdanau)
K = np.random.randn(T, H)  # (T, H)
q = np.random.randn(H)     # decoder hidden state (query)

# Learnable projections
W_q = np.random.randn(A, H) * 0.1
W_k = np.random.randn(A, H) * 0.1
v   = np.random.randn(A)   * 0.1

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# 1. Project query and each key into attention space
q_proj = W_q @ q                        # (A,)
k_proj = K @ W_k.T                      # (T, A)

# 2. Additive score: v · tanh(W_q·q + W_k·k_j)
scores = np.tanh(q_proj + k_proj) @ v   # (T,)

# 3. Softmax → attention weights
alpha = softmax(scores)                  # (T,) sums to 1
print("Attention weights α:", np.round(alpha, 4))

# 4. Context vector: weighted sum of encoder states
context = alpha @ K                      # (H,)
print("Context vector shape:", context.shape)`

const CODE_DOT_PRODUCT = `import numpy as np

# ── Dot-product attention (unscaled) ──
# Score(q, k) = q · k  (dot product of query and key)

np.random.seed(1)
T_src, T_tgt, D = 6, 4, 8   # source len, target len, dim

Q = np.random.randn(T_tgt, D)   # queries  (T_q × D)
K = np.random.randn(T_src, D)   # keys     (T_k × D)
V = np.random.randn(T_src, D)   # values   (T_k × D)

def softmax_rows(X):
    e = np.exp(X - X.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# 1. Compute all Q·K^T scores at once: (T_q × T_k)
scores = Q @ K.T                          # (T_q, T_k)
print("Score matrix shape:", scores.shape)

# 2. Softmax over key dimension
weights = softmax_rows(scores)            # (T_q, T_k)
print("Weight matrix (row sums):", weights.sum(axis=-1).round(3))

# 3. Weighted sum of values
output = weights @ V                      # (T_q, D)
print("Output shape:", output.shape)`

const CODE_SCALED = `import tensorflow as tf
import numpy as np

# ── Scaled dot-product attention ──
# Attn(Q,K,V) = softmax( Q K^T / sqrt(d_k) ) V

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q : (batch, heads, T_q, d_k)
    K : (batch, heads, T_k, d_k)
    V : (batch, heads, T_k, d_v)
    Returns: output (batch, heads, T_q, d_v), weights (batch, heads, T_q, T_k)
    """
    d_k = tf.cast(tf.shape(K)[-1], tf.float32)
    scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k)  # (…, T_q, T_k)

    if mask is not None:
        scores += mask * -1e9      # mask padding / future tokens

    weights = tf.nn.softmax(scores, axis=-1)
    output  = tf.matmul(weights, V)
    return output, weights

# Quick sanity check
B, H, T, dk = 2, 4, 6, 16
Q_ = tf.random.normal((B, H, T, dk))
K_ = tf.random.normal((B, H, T, dk))
V_ = tf.random.normal((B, H, T, dk))

out, w = scaled_dot_product_attention(Q_, K_, V_)
print("Output shape :", out.shape)    # (2, 4, 6, 16)
print("Weights shape:", w.shape)      # (2, 4, 6, 6)
print("Weight row sum:", w[0,0,0].numpy().sum().round(4))  # ≈ 1.0

# ── Why scale? ── 
# For large d_k, Q·K^T has large variance → softmax saturates → tiny gradients
# Dividing by sqrt(d_k) keeps variance ≈ 1 regardless of d_k
d_ks = [4, 16, 64, 256]
for d in d_ks:
    q = np.random.randn(d); k = np.random.randn(d)
    raw  = q @ k
    scal = raw / np.sqrt(d)
    print(f"d_k={d:3d}  raw dot={raw:+.2f}  scaled={scal:+.2f}")`

const CODE_BAHDANAU = `import tensorflow as tf

# ── Bahdanau attention layer (TF/Keras) ──
# Used inside an RNN decoder to attend over encoder outputs

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W_query = tf.keras.layers.Dense(units)   # project decoder state
        self.W_key   = tf.keras.layers.Dense(units)   # project encoder outputs
        self.V       = tf.keras.layers.Dense(1)        # scalar score

    def call(self, query, keys):
        """
        query : (batch, H)       — decoder hidden state at time t
        keys  : (batch, T, H)    — all encoder hidden states
        Returns: context (batch, H), weights (batch, T, 1)
        """
        # Expand query to broadcast over time axis: (batch, 1, units)
        q_exp = tf.expand_dims(self.W_query(query), 1)
        k_proj = self.W_key(keys)                     # (batch, T, units)

        score   = self.V(tf.nn.tanh(q_exp + k_proj))  # (batch, T, 1)
        weights = tf.nn.softmax(score, axis=1)         # (batch, T, 1) sums to 1 over T

        context = tf.reduce_sum(weights * keys, axis=1)  # (batch, H)
        return context, weights

# ── Minimal test ──
BATCH, T, H, A = 2, 7, 32, 16
attn  = BahdanauAttention(A)
query = tf.random.normal((BATCH, H))
keys  = tf.random.normal((BATCH, T, H))
ctx, w = attn(query, keys)
print("Context shape :", ctx.shape)     # (2, 32)
print("Weight shape  :", w.shape)       # (2, 7, 1)
print("Weight sum    :", w[0,:,0].numpy().sum().round(4))  # ≈ 1.0`

const CODE_SELF_ATTENTION = `import tensorflow as tf

# ── Self-attention: Q, K, V all come from the SAME sequence ──
# Each position attends to every other position in the sequence.

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)

    def call(self, x, mask=None):
        """x: (batch, T, d_model)  — input sequence (same for Q, K, V)"""
        Q = self.W_q(x)   # (batch, T, d_model)
        K = self.W_k(x)   # (batch, T, d_model)
        V = self.W_v(x)   # (batch, T, d_model)

        d_k = tf.cast(tf.shape(K)[-1], tf.float32)
        scores  = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)

        if mask is not None:
            scores += mask * -1e9

        weights = tf.nn.softmax(scores, axis=-1)  # (batch, T, T)
        output  = tf.matmul(weights, V)            # (batch, T, d_model)
        return output, weights

# Test: each of 5 tokens attends to all 5 tokens
sa = SelfAttention(d_model=16)
x  = tf.random.normal((2, 5, 16))
out, w = sa(x)
print("Self-attn output:", out.shape)   # (2, 5, 16)
print("Attention map   :", w.shape)     # (2, 5, 5)  — T × T matrix`

const CODE_CROSS_ATTENTION = `import tensorflow as tf

# ── Cross-attention: Q from decoder, K and V from encoder ──
# Connects decoder to encoder outputs in seq2seq models.

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)

    def call(self, decoder_seq, encoder_out):
        """
        decoder_seq : (batch, T_tgt, d_model)  — decoder hidden states
        encoder_out : (batch, T_src, d_model)   — encoder hidden states
        """
        Q = self.W_q(decoder_seq)   # (batch, T_tgt, d_model)
        K = self.W_k(encoder_out)   # (batch, T_src, d_model)
        V = self.W_v(encoder_out)   # (batch, T_src, d_model)

        d_k = tf.cast(tf.shape(K)[-1], tf.float32)
        scores  = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)  # (B, T_tgt, T_src)
        weights = tf.nn.softmax(scores, axis=-1)
        output  = tf.matmul(weights, V)                               # (B, T_tgt, d_model)
        return output, weights

ca = CrossAttention(d_model=16)
dec = tf.random.normal((2, 4, 16))   # 4 decoder tokens
enc = tf.random.normal((2, 6, 16))   # 6 encoder tokens
out, w = ca(dec, enc)
print("Cross-attn output:", out.shape)   # (2, 4, 16)
print("Attention weights:", w.shape)     # (2, 4, 6)  — each decoder token → all encoder`

const CODE_MULTIHEAD = `import tensorflow as tf

# ── Multi-head attention from scratch ──

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads     # depth per head

        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
        self.W_o = tf.keras.layers.Dense(d_model)   # output projection

    def split_heads(self, x, batch):
        """(B, T, d_model) → (B, heads, T, d_k)"""
        x = tf.reshape(x, (batch, -1, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q_in, K_in, V_in, mask=None):
        B = tf.shape(Q_in)[0]
        Q = self.split_heads(self.W_q(Q_in), B)   # (B, h, T_q, d_k)
        K = self.split_heads(self.W_k(K_in), B)   # (B, h, T_k, d_k)
        V = self.split_heads(self.W_v(V_in), B)   # (B, h, T_k, d_k)

        d_k = tf.cast(self.d_k, tf.float32)
        scores  = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
        if mask is not None:
            scores += mask * -1e9
        weights = tf.nn.softmax(scores, axis=-1)   # (B, h, T_q, T_k)
        attended = tf.matmul(weights, V)            # (B, h, T_q, d_k)

        # Concatenate all heads: (B, h, T_q, d_k) → (B, T_q, d_model)
        attended = tf.transpose(attended, perm=[0, 2, 1, 3])
        concat   = tf.reshape(attended, (B, -1, self.num_heads * self.d_k))
        return self.W_o(concat), weights             # (B, T_q, d_model)

# Test
mha = MultiHeadAttention(d_model=64, num_heads=8)
x   = tf.random.normal((2, 10, 64))
out, w = mha(x, x, x)                  # self-attention mode
print("MHA output:", out.shape)         # (2, 10, 64)
print("Weights   :", w.shape)           # (2, 8, 10, 10)  — 8 heads × T × T
print("Params    :", mha.count_params())`

const CODE_POSITIONAL = `import tensorflow as tf
import numpy as np

# ── Sinusoidal positional encoding (Vaswani et al., 2017) ──

def get_sinusoidal_encoding(seq_len, d_model):
    """
    Returns PE matrix of shape (1, seq_len, d_model).
    PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
    """
    positions = np.arange(seq_len)[:, None]          # (T, 1)
    dims      = np.arange(d_model)[None, :]           # (1, D)

    # Frequency for each dimension pair
    angle_rates = 1 / (10000 ** (2 * (dims // 2) / d_model))
    angles = positions * angle_rates                   # (T, D)

    # Even indices → sin, odd → cos
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return angles[None, :, :].astype(np.float32)      # (1, T, D)

PE = get_sinusoidal_encoding(seq_len=50, d_model=16)
print("PE shape:", PE.shape)    # (1, 50, 16)

# ── Learned positional embedding (alternative) ──
class LearnedPosEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_emb = tf.keras.layers.Embedding(max_len, d_model)

    def call(self, x):
        T = tf.shape(x)[1]
        positions = tf.range(T)
        return x + self.pos_emb(positions)   # add positional info to token embeddings

# Usage in a mini Transformer encoder
d_model = 32; vocab = 100; max_len = 50; T = 12

tok_emb = tf.keras.layers.Embedding(vocab, d_model)
pos_enc = LearnedPosEncoding(max_len, d_model)

tokens  = tf.random.uniform((2, T), 0, vocab, dtype=tf.int32)
x = pos_enc(tok_emb(tokens))           # (2, T, d_model) — tok + pos
print("With positional encoding:", x.shape)`

// ── Quiz ──────────────────────────────────────────────────────────────────────

const QUIZ_QUESTIONS = [
  {
    question: 'In the Q/K/V framework, what does the attention weight α_j represent?',
    options: [
      'The magnitude of key j',
      'How relevant key j is to the current query — how much to attend to value j',
      'The position of token j in the sequence',
      'The gradient flowing through token j',
    ],
    correct: 1,
    explanation: 'α_j = softmax(score(q, k_j)). It is the fraction of attention allocated to value j. Higher α_j means the output context vector borrows more from V_j.',
  },
  {
    question: 'Why does scaled dot-product attention divide scores by √d_k?',
    options: [
      'To ensure the output values stay between 0 and 1',
      'To normalise the number of attention heads',
      'To prevent softmax saturation caused by large dot products when d_k is large',
      'To make the computation independent of batch size',
    ],
    correct: 2,
    explanation: 'For large d_k, dot products grow in magnitude (variance ∝ d_k), pushing softmax into regions with tiny gradients. Dividing by √d_k keeps variance ≈ 1.',
  },
  {
    question: 'In self-attention, where do Q, K, and V come from?',
    options: [
      'Q from the decoder, K and V from the encoder',
      'Q, K, V all come from the same input sequence (different linear projections)',
      'Q from the encoder, K from the decoder, V from the encoder',
      'Q, K, V are fixed random matrices',
    ],
    correct: 1,
    explanation: 'Self-attention applies three separate linear projections (W_Q, W_K, W_V) to the same input sequence. Each token queries against every other token in the same sequence.',
  },
  {
    question: 'In cross-attention inside a Transformer decoder, Q comes from:',
    options: [
      'The encoder output',
      'The positional encoding',
      'The decoder\'s own hidden states',
      'A learned embedding table',
    ],
    correct: 2,
    explanation: 'In cross-attention (encoder–decoder attention), Q comes from the decoder sequence and K, V come from the encoder output. This lets the decoder "look up" relevant source information.',
  },
  {
    question: 'Multi-head attention with h heads and model dimension d_model uses a depth per head of:',
    options: ['d_model', 'd_model × h', 'd_model / h', '√d_model'],
    correct: 2,
    explanation: 'Each head works in a subspace of dimension d_k = d_model / h. Splitting into h independent heads lets the model jointly attend to different representation subspaces.',
  },
  {
    question: 'Sinusoidal positional encoding uses sin/cos at different frequencies so that:',
    options: [
      'The encoding sums to zero across positions',
      'Relative positions can be expressed as linear transformations of the encoding',
      'Each position gets a unique binary code',
      'Gradients flow more easily through the embedding',
    ],
    correct: 1,
    explanation: 'PE(pos + k) can be expressed as a linear function of PE(pos) — the model can learn to attend to relative offsets. Sinusoidal also generalises to sequence lengths longer than seen in training.',
  },
]

// ── Attention heatmap interactive visualiser ──────────────────────────────────

function AttentionHeatmap({ theme }) {
  const dark = theme === 'dark'
  const tokens = ['The', 'cat', 'sat', 'on', 'mat']
  const n = tokens.length

  // Fixed example attention weights (row = query, col = key)
  const weights = [
    [0.60, 0.15, 0.10, 0.08, 0.07],
    [0.12, 0.58, 0.14, 0.09, 0.07],
    [0.08, 0.18, 0.52, 0.12, 0.10],
    [0.07, 0.08, 0.13, 0.55, 0.17],
    [0.06, 0.09, 0.11, 0.16, 0.58],
  ]

  const [hovered, setHovered] = useState(null) // { row, col }

  const cellSize = 48
  const pad = 56

  return (
    <div className="overflow-x-auto my-4">
      <div className="flex flex-col items-center">
        <p className={`text-xs mb-3 ${dark ? 'text-slate-400' : 'text-slate-500'}`}>
          Self-attention map — hover a cell to inspect α<sub>ij</sub>
        </p>
        <svg
          width={pad + n * cellSize + 4}
          height={pad + n * cellSize + 4}
          style={{ userSelect: 'none' }}
        >
          {/* Column (key) labels */}
          {tokens.map((t, j) => (
            <text key={j}
              x={pad + j * cellSize + cellSize / 2}
              y={pad - 8}
              textAnchor="middle"
              fontSize="11"
              fontWeight="600"
              fill={dark ? '#94a3b8' : '#64748b'}
            >{t}</text>
          ))}
          {/* Row (query) labels */}
          {tokens.map((t, i) => (
            <text key={i}
              x={pad - 6}
              y={pad + i * cellSize + cellSize / 2 + 4}
              textAnchor="end"
              fontSize="11"
              fontWeight="600"
              fill={dark ? '#94a3b8' : '#64748b'}
            >{t}</text>
          ))}
          {/* Cells */}
          {weights.map((row, i) =>
            row.map((val, j) => {
              const isHov = hovered && hovered.row === i && hovered.col === j
              const fill = dark
                ? `rgba(99,102,241,${val.toFixed(2)})`
                : `rgba(99,102,241,${val.toFixed(2)})`
              return (
                <g key={`${i}-${j}`}>
                  <rect
                    x={pad + j * cellSize + 2}
                    y={pad + i * cellSize + 2}
                    width={cellSize - 4}
                    height={cellSize - 4}
                    rx="4"
                    fill={fill}
                    stroke={isHov ? (dark ? '#f472b6' : '#db2777') : 'transparent'}
                    strokeWidth="2"
                    style={{ cursor: 'pointer' }}
                    onMouseEnter={() => setHovered({ row: i, col: j })}
                    onMouseLeave={() => setHovered(null)}
                  />
                  <text
                    x={pad + j * cellSize + cellSize / 2}
                    y={pad + i * cellSize + cellSize / 2 + 4}
                    textAnchor="middle"
                    fontSize="10"
                    fontWeight="600"
                    fill={val > 0.35 ? '#fff' : (dark ? '#c7d2fe' : '#4338ca')}
                    style={{ pointerEvents: 'none' }}
                  >{val.toFixed(2)}</text>
                </g>
              )
            })
          )}
        </svg>
        {hovered && (
          <p className={`text-xs mt-2 font-mono ${dark ? 'text-indigo-300' : 'text-indigo-600'}`}>
            query="{tokens[hovered.row]}" attends to key="{tokens[hovered.col]}" with weight {weights[hovered.row][hovered.col].toFixed(2)}
          </p>
        )}
        {!hovered && (
          <p className={`text-xs mt-2 ${dark ? 'text-slate-500' : 'text-slate-400'}`}>
            Diagonal dominance: each token attends most strongly to itself
          </p>
        )}
      </div>
    </div>
  )
}

// ── Positional encoding visualiser ───────────────────────────────────────────

function PEVis({ theme }) {
  const dark = theme === 'dark'
  const seqLen = 20
  const dims = [0, 1, 4, 5, 8, 9] // show 6 dimensions

  function pe(pos, dim, dModel = 32) {
    const i = Math.floor(dim / 2)
    const freq = 1 / Math.pow(10000, (2 * i) / dModel)
    return dim % 2 === 0 ? Math.sin(pos * freq) : Math.cos(pos * freq)
  }

  const W = 340, H = 100
  const colors = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#f43f5e']

  return (
    <div className="overflow-x-auto my-4">
      <svg viewBox={`0 0 ${W} ${H + 20}`} className="w-full max-w-md mx-auto block">
        {dims.map((dim, di) => {
          const pts = Array.from({ length: seqLen }, (_, pos) => {
            const x = 20 + (pos / (seqLen - 1)) * (W - 40)
            const y = H / 2 - pe(pos, dim) * (H / 2 - 8)
            return `${x},${y}`
          }).join(' ')
          return (
            <polyline key={dim} points={pts}
              fill="none"
              stroke={colors[di]}
              strokeWidth="1.5"
              opacity="0.85" />
          )
        })}
        {/* Axis */}
        <line x1="20" y1={H / 2} x2={W - 10} y2={H / 2}
          stroke={dark ? '#334155' : '#e2e8f0'} strokeWidth="1" />
        {/* Legend */}
        {dims.map((dim, di) => (
          <g key={dim}>
            <rect x={20 + di * 52} y={H + 4} width="10" height="10" rx="2" fill={colors[di]} />
            <text x={33 + di * 52} y={H + 13} fontSize="9"
              fill={dark ? '#94a3b8' : '#64748b'}>dim {dim}</text>
          </g>
        ))}
      </svg>
      <p className={`text-center text-xs mt-1 ${dark ? 'text-slate-400' : 'text-slate-500'}`}>
        Sinusoidal PE across 20 positions — lower dims oscillate slowly, higher dims faster
      </p>
    </div>
  )
}

// ── QKV diagram ───────────────────────────────────────────────────────────────

function QKVDiagram({ theme }) {
  const dark = theme === 'dark'
  return (
    <svg viewBox="0 0 480 130" className="w-full max-w-lg mx-auto block my-4" style={{ minWidth: 320 }}>
      {/* Input */}
      <rect x="10" y="45" width="70" height="40" rx="8"
        fill={dark ? '#1e293b' : '#f1f5f9'} stroke={dark ? '#475569' : '#cbd5e1'} strokeWidth="1.5" />
      <text x="45" y="70" textAnchor="middle" fontSize="11" fontWeight="700"
        fill={dark ? '#94a3b8' : '#475569'}>Input x</text>

      {/* Arrows to W_Q, W_K, W_V */}
      {[['W_Q', 150, '#6366f1', dark ? '#312e81' : '#eef2ff', dark ? '#818cf8' : '#4338ca'],
        ['W_K', 230, '#06b6d4', dark ? '#083344' : '#ecfeff', dark ? '#22d3ee' : '#0891b2'],
        ['W_V', 310, '#10b981', dark ? '#052e16' : '#f0fdf4', dark ? '#34d399' : '#059669']
      ].map(([label, cx, stroke, bg, fg]) => (
        <g key={label}>
          <line x1="80" y1="65" x2={cx - 26} y2="65"
            stroke={dark ? '#475569' : '#cbd5e1'} strokeWidth="1.2" markerEnd="url(#arrq)" />
          <rect x={cx - 26} y="45" width="52" height="40" rx="8"
            fill={bg} stroke={stroke} strokeWidth="1.5" />
          <text x={cx} y="70" textAnchor="middle" fontSize="11" fontWeight="700"
            fill={fg}>{label}</text>
          {/* Arrow W → output */}
          <line x1={cx} y1="85" x2={cx} y2="110"
            stroke={stroke} strokeWidth="1.5" markerEnd="url(#arrq)" />
          {/* Output label */}
          <text x={cx} y="125" textAnchor="middle" fontSize="11" fontWeight="600"
            fill={fg}>{label.replace('W_', '')}</text>
        </g>
      ))}

      {/* Attention block */}
      <rect x="370" y="30" width="100" height="70" rx="10"
        fill={dark ? '#3b0764' : '#fdf4ff'} stroke={dark ? '#a855f7' : '#a855f7'} strokeWidth="1.5" />
      <text x="420" y="62" textAnchor="middle" fontSize="11" fontWeight="700"
        fill={dark ? '#e9d5ff' : '#7e22ce'}>Attention</text>
      <text x="420" y="78" textAnchor="middle" fontSize="10"
        fill={dark ? '#d8b4fe' : '#9333ea'}>softmax(QKᵀ/√dₖ)V</text>

      {/* Arrows Q,K,V → Attention */}
      {[150, 230, 310].map((cx, i) => (
        <line key={i} x1={cx + 26} y1="65" x2="370" y2="65"
          stroke={dark ? '#475569' : '#cbd5e1'} strokeWidth="1" strokeDasharray="4 2" />
      ))}

      <defs>
        <marker id="arrq" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill={dark ? '#475569' : '#94a3b8'} />
        </marker>
      </defs>
    </svg>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function Attention() {
  const { theme, markTopicComplete } = useApp()
  const dark = theme === 'dark'

  return (
    <div className="space-y-10">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3 mb-3">
          <span className="text-4xl">🔍</span>
          <div>
            <p className={`text-sm font-semibold uppercase tracking-widest ${dark ? 'text-violet-400' : 'text-violet-600'}`}>
              Session 11 · Deep Learning
            </p>
            <h1 className={`text-3xl font-black ${dark ? 'text-white' : 'text-slate-900'}`}>
              Attention Mechanism
            </h1>
          </div>
        </div>
        <p className={`text-lg leading-relaxed ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The encoder–decoder bottleneck forces the entire source sequence into one fixed-size vector.
          <strong> Attention</strong> breaks that constraint — at each decoding step the model dynamically
          selects which source positions to focus on. Self-attention then generalises this to within-sequence
          relationships, forming the backbone of the Transformer.
        </p>

        {/* Prerequisites callout */}
        <div className={`mt-4 rounded-xl p-4 border text-sm ${dark ? 'bg-amber-900/20 border-amber-700/40 text-amber-200' : 'bg-amber-50 border-amber-200 text-amber-800'}`}>
          <p className="font-bold mb-1">📚 Prerequisites covered in earlier sessions</p>
          <ul className="list-disc list-inside space-y-0.5 text-xs">
            <li><strong>Session 9</strong> — Encoder–decoder architecture, context vector, teacher forcing, greedy decoding</li>
            <li><strong>Session 9</strong> — Sequence tokenisation, embeddings, hidden states</li>
            <li><strong>Session 10</strong> — LSTM / GRU gating, bidirectional RNNs (used as encoder backbone here)</li>
          </ul>
        </div>
      </motion.div>

      {/* ── Section 11.1 ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 11.1</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Queries, Keys and Values</h2>
        </div>

        {/* 11.1.1 Intuition */}
        <h3 className={`text-lg font-bold mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          11.1.1 — Attention as Soft Retrieval from a Dictionary
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Imagine a key-value dictionary where instead of a hard lookup (one exact key matches), you perform a
          <strong> soft lookup</strong>: your query is compared to all keys simultaneously, and you retrieve a
          weighted blend of all values — with higher weight for keys closest to your query.
        </p>

        <TheoryBlock cols={2} items={[
          {
            title: '📖 Hard lookup (Python dict)',
            content: 'd["cat"] → returns exactly one value. Query must exactly match a key.',
          },
          {
            title: '🌊 Soft lookup (attention)',
            content: 'Compare query to ALL keys. Get a probability distribution over keys. Return a weighted sum of the corresponding values.',
          },
          {
            title: '🌍 Analogy: translation',
            content: 'Decoding the word "bank" — query is current decoder state. Keys are encoder states for "river", "money", "financial". Attention weight tells how relevant each source word is.',
          },
          {
            title: '📐 Why it works',
            content: 'Similarity is measured via dot products. High dot product → similar direction in embedding space → high attention weight → that value contributes more to the output.',
          },
        ]} />

        {/* 11.1.2 Q K V definitions */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          11.1.2 — Query (Q), Key (K), Value (V) — Definitions
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Every input representation is linearly projected into three distinct roles: a Query
          (what am I looking for?), a Key (what do I contain?), and a Value (what do I actually provide
          if selected?). The projections are learned independently.
        </p>

        <QKVDiagram theme={theme} />

        <TheoryBlock items={[
          { title: 'Q — Query', content: 'Represents the current information need. "What context do I need?" In cross-attention: the current decoder token.' },
          { title: 'K — Key', content: 'Represents what each source position advertises. "This is what I contain." Matched against Q to compute compatibility.' },
          { title: 'V — Value', content: 'The actual content retrieved when a key matches. Decoupling K from V lets a position advertise one thing while providing richer information.' },
          { title: 'Linear projections W_Q, W_K, W_V', content: 'Learned d_model × d_k matrices. Allow the model to project the same input into different representational spaces for querying vs. being queried.' },
        ]} />

        {/* 11.1.3 Scoring functions */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          11.1.3 — Attention Scoring Functions: Additive vs. Dot-Product
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The scoring function measures how compatible a query is with each key. Two families dominate:
        </p>

        <TheoryBlock cols={2} items={[
          {
            title: '➕ Additive (Bahdanau)',
            content: 'score(q, k) = vᵀ · tanh(W_q·q + W_k·k). Uses a learnable alignment network. More parameters but can capture asymmetric interactions.',
          },
          {
            title: '· Dot-product (Luong / Transformer)',
            content: 'score(q, k) = q · k. Just a dot product — extremely cheap to compute in batched matrix form. Preferred for its O(1) parameter overhead.',
          },
        ]} />

        <CodeBlock code={CODE_ADDITIVE} language="python" title="bahdanau_score.py" />
        <div className="mt-4">
          <CodeBlock code={CODE_DOT_PRODUCT} language="python" title="dot_product_attention.py" />
        </div>
      </motion.div>

      {/* ── Section 11.2 ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-violet-500/5 border-violet-500/20' : 'bg-violet-50 border-violet-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-violet-400' : 'text-violet-600'}`}>Section 11.2</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Attention Variants</h2>
        </div>

        {/* 11.2.1 Dot-product */}
        <h3 className={`text-lg font-bold mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          11.2.1 — Dot-Product Attention
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          All query–key scores can be computed in a single matrix multiplication: <code>QKᵀ</code>,
          giving a (T_q × T_k) score matrix. Applying softmax row-wise and multiplying by V gives the
          attended output in three lines of code.
        </p>

        <Callout type="formula" title="Dot-product attention">
          Attn(Q, K, V) = softmax(Q Kᵀ) · V
        </Callout>

        {/* 11.2.2 Scaled */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          11.2.2 — Scaled Dot-Product Attention — Division by √d_k
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          For large d_k the dot products grow in magnitude (variance ≈ d_k), pushing the softmax into
          flat saturation regions where gradients vanish. Dividing by √d_k normalises the variance back to 1.
        </p>

        <Callout type="formula" title="Scaled dot-product attention">
          Attn(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V
        </Callout>

        <CodeBlock code={CODE_SCALED} language="python" title="scaled_attention.py" />

        {/* 11.2.3 Bahdanau */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          11.2.3 — Bahdanau Attention Mechanism
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Bahdanau et al. (2015) introduced attention for neural machine translation. The decoder queries
          all encoder hidden states at every decoding step, producing a fresh context vector instead of
          relying on a single bottleneck state.
        </p>

        <TheoryBlock items={[
          { title: 'Problem solved', content: 'Vanilla seq2seq compresses the entire source into one vector. For long sentences, early tokens are forgotten. Attention lets the decoder look back at any encoder position.' },
          { title: 'Alignment matrix', content: 'For a sentence of length T_src decoded over T_tgt steps, the attention produces a (T_tgt × T_src) alignment matrix — interpretable as which source words each output word came from.' },
          { title: 'Training signal', content: 'No separate supervision needed. The attention weights are trained end-to-end via the translation loss — the network learns where to look by itself.' },
        ]} />

        <CodeBlock code={CODE_BAHDANAU} language="python" title="bahdanau_attention.py" />

        {/* 11.2.4 Self-attention */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          11.2.4 — Self-Attention: Q, K, V from the Same Sequence
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          In self-attention, <em>all three</em> of Q, K, and V are derived from the same input sequence.
          Every token attends to every other token — directly, in a single layer, with no recurrence.
          This enables full parallelism and captures long-range dependencies in O(1) layers.
        </p>

        <AttentionHeatmap theme={theme} />

        <Callout type="info" title="Self-attention vs recurrence">
          An RNN needs T sequential steps to relate token 1 to token T (path length = T).
          Self-attention relates any two tokens in a single layer (path length = 1) — at the cost of
          O(T²) memory for the attention map.
        </Callout>

        <CodeBlock code={CODE_SELF_ATTENTION} language="python" title="self_attention.py" />

        {/* 11.2.5 Cross-attention */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          11.2.5 — Cross-Attention: Q from Decoder, K/V from Encoder
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Cross-attention is the Transformer generalisation of Bahdanau attention. The decoder's hidden
          representations form the queries; the encoder output provides the keys and values. Each decoder
          position learns which encoder positions to retrieve from.
        </p>

        <TheoryBlock cols={2} items={[
          {
            title: '🔎 Query source',
            content: 'Q = W_Q · decoder_hidden. The decoder asks "what source information do I need to generate this output token?"',
          },
          {
            title: '📚 Key/Value source',
            content: 'K = W_K · encoder_out, V = W_V · encoder_out. The encoder exposes its full sequence as a memory bank that the decoder indexes into.',
          },
          {
            title: '📐 Shape',
            content: 'Score matrix is (T_tgt × T_src). Each of the T_tgt decoder positions computes a distribution over all T_src encoder positions.',
          },
          {
            title: '🔄 Used in Transformers',
            content: 'Transformer decoder blocks contain three sub-layers: masked self-attention → cross-attention → feed-forward. Cross-attention is the second sublayer.',
          },
        ]} />

        <CodeBlock code={CODE_CROSS_ATTENTION} language="python" title="cross_attention.py" />

        {/* 11.2.6 Multi-head attention */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          11.2.6 — Multi-Head Attention: Parallel Attention Heads
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          A single attention head may focus on one type of relationship (e.g. syntactic agreement).
          Multi-head attention runs <em>h</em> independent scaled dot-product attention operations in
          parallel, each in a subspace of dimension d_k = d_model / h, then concatenates and projects.
        </p>

        <Callout type="formula" title="Multi-head attention">
          MultiHead(Q, K, V) = Concat(head_1, …, head_h) · W_O<br />
          where head_i = Attn(Q·W_Qi, K·W_Ki, V·W_Vi)
        </Callout>

        <TheoryBlock items={[
          { title: 'Why multiple heads?', content: 'Different heads can independently learn different relationships: one for syntax (subject–verb), one for coreference (pronoun–antecedent), one for proximity. Single head is forced to average them.' },
          { title: 'Parameter cost', content: 'Each head has W_Qi, W_Ki, W_Vi ∈ R^{d_model × d_k} plus final W_O ∈ R^{d_model × d_model}. Total ≈ 4 × d_model² — same as a single full-dim head.' },
          { title: 'Parallelism', content: 'All h heads are computed simultaneously via a single batched matrix multiply (batch dimension = heads). No sequential dependency.' },
        ]} />

        <CodeBlock code={CODE_MULTIHEAD} language="python" title="multihead_attention.py" />

        <DeepDive title="Typical Transformer dimensions">
          <TheoryBlock items={[
            { title: '', content: 'BERT-base: d_model=768, h=12 heads → d_k=64. BERT-large: d_model=1024, h=16 → d_k=64. GPT-2 small: d_model=768, h=12. The d_k=64 sweet spot balances expressiveness and softmax stability.' },
          ]} />
        </DeepDive>

        {/* 11.2.7 Positional encoding */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          11.2.7 — Positional Encoding: Sinusoidal and Learned
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Self-attention is <strong>permutation-invariant</strong> — shuffling the tokens gives the same
          output. To inject order information, a positional encoding is <em>added</em> to the token
          embedding before the first attention layer.
        </p>

        <PEVis theme={theme} />

        <TheoryBlock cols={2} items={[
          {
            title: '〰️ Sinusoidal PE',
            content: 'PE(pos, 2i) = sin(pos / 10000^{2i/d}), PE(pos, 2i+1) = cos(…). Fixed — no parameters. Naturally generalises to longer sequences than seen in training. Relative offsets can be expressed as linear transforms.',
          },
          {
            title: '🎓 Learned PE',
            content: 'An Embedding(max_len, d_model) table where position indices are the inputs. Learned end-to-end. Slightly better on standard benchmarks but does not extrapolate beyond max_len.',
          },
          {
            title: '➕ Addition, not concatenation',
            content: 'PE is added to the token embedding (same dimension d_model). The network learns to separate positional from semantic information implicitly.',
          },
          {
            title: '🔁 RoPE / ALiBi (modern variants)',
            content: 'Rotary Positional Embedding (RoPE, used in LLaMA) and ALiBi bias attention scores directly. These extend context lengths far beyond training.',
          },
        ]} />

        <CodeBlock code={CODE_POSITIONAL} language="python" title="positional_encoding.py" />

        <Callout type="info" title="Why sin/cos at different frequencies?">
          Low frequencies (early dimensions) change slowly across positions — useful for absolute
          position. High frequencies (late dimensions) change rapidly — useful for relative proximity.
          Together they form a unique "fingerprint" for every position up to any sequence length.
        </Callout>
      </motion.div>

      {/* ── Quiz ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-cyan-500/5 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-cyan-400' : 'text-cyan-600'}`}>Knowledge Check</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Quiz — Sessions 11.1 &amp; 11.2</h2>
        </div>
        <Quiz
          questions={QUIZ_QUESTIONS}
          onComplete={() => markTopicComplete('attention')}
        />
      </motion.div>
    </div>
  )
}
