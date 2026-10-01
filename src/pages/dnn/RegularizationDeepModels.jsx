import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

// ── Code blocks ────────────────────────────────────────────────────────────────

const CODE_BIAS_VARIANCE = `import numpy as np

# ── Bias-Variance trade-off demo ──
# Fit polynomials of increasing degree to noisy sine data

np.random.seed(42)
N_train = 20
x_train = np.linspace(0, 1, N_train)
y_train = np.sin(2 * np.pi * x_train) + 0.3 * np.random.randn(N_train)

x_test = np.linspace(0, 1, 200)
y_true = np.sin(2 * np.pi * x_test)

def poly_mse(degree):
    # Fit polynomial of given degree
    coeffs  = np.polyfit(x_train, y_train, degree)
    y_pred_train = np.polyval(coeffs, x_train)
    y_pred_test  = np.polyval(coeffs, x_test)
    train_mse = np.mean((y_pred_train - y_train) ** 2)
    test_mse  = np.mean((y_pred_test  - y_true ) ** 2)
    return train_mse, test_mse

print(f"{'Degree':>8}  {'Train MSE':>12}  {'Test MSE':>12}  {'Status'}")
print("-" * 55)
for d in [1, 2, 3, 5, 9, 14, 19]:
    tr, te = poly_mse(d)
    if te > 0.5:   status = "OVERFIT ⚠️"
    elif tr > 0.2: status = "UNDERFIT"
    else:          status = "OK ✅"
    print(f"{d:>8}  {tr:>12.4f}  {te:>12.4f}  {status}")

# Degree 1-2: high train & test error   → underfitting (high bias)
# Degree 3-5: low train, low test error → good generalisation
# Degree 14+: near-zero train, huge test error → overfitting (high variance)`

const CODE_CROSS_VALIDATION = `import numpy as np

# ── k-Fold Cross-Validation ──

np.random.seed(0)
N = 100
X = np.random.randn(N, 3)
y = X @ np.array([1.5, -2.0, 0.5]) + 0.5 * np.random.randn(N)

def train_eval(X_tr, y_tr, X_val, y_val):
    """Closed-form linear regression."""
    w = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
    return np.mean((X_val @ w - y_val) ** 2)

k = 5
fold_size = N // k
idx = np.random.permutation(N)
val_mses = []

for fold in range(k):
    val_idx  = idx[fold * fold_size : (fold + 1) * fold_size]
    train_idx = np.concatenate([idx[:fold * fold_size],
                                idx[(fold + 1) * fold_size:]])
    mse = train_eval(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
    val_mses.append(mse)
    print(f"  Fold {fold+1}: val MSE = {mse:.4f}")

print(f"\\n  Mean CV MSE : {np.mean(val_mses):.4f}")
print(f"  Std  CV MSE : {np.std(val_mses):.4f}")
# Low std → stable model; high std → variance too high, model is sensitive to data split`

const CODE_WEIGHT_DECAY = `import numpy as np

# ── L2 Weight Decay (Ridge Regression) ──
# Loss = MSE + λ·‖w‖²
# Gradient = ∂MSE/∂w + 2λ·w
# Closed-form: w* = (XᵀX + λI)⁻¹ Xᵀy

np.random.seed(0)
N, D = 80, 20          # fewer samples than features → overfitting risk
X = np.random.randn(N, D)
w_true = np.random.randn(D) * 0.5
y = X @ w_true + 0.3 * np.random.randn(N)

# Hold-out test set
X_test = np.random.randn(200, D)
y_test = X_test @ w_true + 0.3 * np.random.randn(200)

print(f"{'λ':>10}  {'Train MSE':>12}  {'Test MSE':>12}  {'‖w‖':>8}")
print("-" * 48)
for lam in [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]:
    # Closed-form ridge solution
    A = X.T @ X + lam * np.eye(D)
    w = np.linalg.solve(A, X.T @ y)
    train_mse = np.mean((X @ w - y) ** 2)
    test_mse  = np.mean((X_test @ w - y_test) ** 2)
    print(f"{lam:>10.3f}  {train_mse:>12.4f}  {test_mse:>12.4f}  {np.linalg.norm(w):>8.3f}")

# λ=0 → severe overfitting. Optimal λ balances train/test gap.
# Larger λ → smaller weights (more regularised) but higher bias.`

const CODE_DROPOUT = `import numpy as np

# ── Dropout — forward pass (train vs. inference) ──

np.random.seed(42)

def dropout_train(x, p_drop=0.5):
    """
    p_drop: probability of zeroing a unit.
    Remaining units scaled by 1/(1-p_drop) — inverted dropout.
    Keeps expected activation magnitude constant.
    """
    mask = (np.random.rand(*x.shape) > p_drop).astype(float)
    return x * mask / (1 - p_drop), mask

def dropout_infer(x):
    """No dropout at inference — weights already scaled."""
    return x

# Simulated layer activations
h = np.array([0.8, -1.2, 0.3, 2.1, -0.5, 1.6, 0.0, 0.9])

print("Original activations: ", np.round(h, 2))
print()

for trial in range(4):
    h_dropped, mask = dropout_train(h, p_drop=0.5)
    print(f"  Trial {trial+1} | mask={mask.astype(int)}  | h_out={np.round(h_dropped,2)}")

print()
h_inf = dropout_infer(h)
print("Inference (no dropout):", np.round(h_inf, 2))

# Each trial uses a different random mask → forces the network to use
# redundant representations rather than co-adapting neurons.`

const CODE_BATCHNORM = `import numpy as np

# ── Batch Normalization — forward pass ──
# Normalise each feature across the mini-batch, then rescale.
#
# μ_B = mean over batch
# σ²_B = variance over batch
# x̂ = (x - μ_B) / sqrt(σ²_B + ε)
# y  = γ · x̂ + β          (γ, β are learned per-feature)

np.random.seed(0)
B, D = 8, 4       # batch size, feature dimension
eps  = 1e-5

X = np.random.randn(B, D) * np.array([10, 0.5, 3, 100])   # different scales!
gamma = np.ones(D)   # learned scale  (init to 1)
beta  = np.zeros(D)  # learned shift  (init to 0)

# ── Training: normalise over batch ──
mu  = X.mean(axis=0)            # (D,)
var = X.var(axis=0)             # (D,)
X_hat = (X - mu) / np.sqrt(var + eps)  # (B, D)  — unit normal per feature
Y     = gamma * X_hat + beta           # (B, D)

print("Input stats (per feature):")
print("  mean:", np.round(X.mean(0), 2))
print("  std: ", np.round(X.std(0),  2))
print()
print("After BN (per feature):")
print("  mean:", np.round(Y.mean(0), 4))  # ≈ 0
print("  std: ", np.round(Y.std(0),  4))  # ≈ 1

# ── At inference: use running statistics (exponential moving average) ──
# running_mean = 0.9 * running_mean + 0.1 * mu_batch
# (accumulated during training; batch statistics are NOT used at test time)`

const CODE_LAYERNORM = `import numpy as np

# ── Layer Normalization — forward pass ──
# Normalise each sample across its feature dimension (not across batch).
# Ideal for sequences where batch statistics are meaningless.

np.random.seed(1)
B, T, D = 4, 6, 8    # batch, sequence length, embedding dim
eps = 1e-5

X = np.random.randn(B, T, D) * 3 + 2   # various scales

gamma = np.ones(D)   # learned, shape (D,)
beta  = np.zeros(D)

# ── Normalise over last axis (feature dim D) ──
mu  = X.mean(axis=-1, keepdims=True)        # (B, T, 1)
var = X.var( axis=-1, keepdims=True)        # (B, T, 1)
X_hat = (X - mu) / np.sqrt(var + eps)       # (B, T, D)
Y     = gamma * X_hat + beta                # (B, T, D)

print("Before LN (sample 0, token 0):")
print(f"  mean={X[0,0].mean():.3f}  std={X[0,0].std():.3f}")
print()
print("After  LN (sample 0, token 0):")
print(f"  mean={Y[0,0].mean():.4f}  std={Y[0,0].std():.4f}")

# BN: statistics over batch  → shape (B,T,D) → mean/var over B axis
# LN: statistics over features → shape (B,T,D) → mean/var over D axis`

// ── Quiz questions ─────────────────────────────────────────────────────────────

const QUIZ_QUESTIONS = [
  {
    question: "A model achieves 98% training accuracy but only 62% test accuracy. This is best described as:",
    options: [
      "Underfitting — the model has too few parameters",
      "Overfitting — the model has memorised training data and fails to generalise",
      "Distribution shift — the test set was drawn from a different distribution",
      "A well-generalised model with an unusually hard test set",
    ],
    answer: 1,
    explanation:
      "A large gap between training accuracy (98%) and test accuracy (62%) is the hallmark of overfitting (high variance). The model has fit the training noise rather than the underlying pattern. Solutions include regularisation (dropout, weight decay), more data, or a simpler model.",
  },
  {
    question: "L2 weight decay penalises large weights by adding λ‖w‖² to the loss. The practical effect on the SGD update is:",
    options: [
      "The gradient is clipped whenever its norm exceeds λ",
      "Each weight is multiplied by (1 − η·2λ) before the gradient step — a multiplicative shrinkage",
      "Weights below λ are set to zero (sparsity)",
      "The learning rate is scaled down by a factor of λ at every step",
    ],
    answer: 1,
    explanation:
      "Adding λ‖w‖² to the loss adds 2λw to the gradient. The update becomes: w ← w − η(g + 2λw) = (1 − 2ηλ)w − ηg. The factor (1 − 2ηλ) is the weight decay — it multiplies each weight by a value slightly below 1 each step, continuously shrinking weight magnitudes toward zero.",
  },
  {
    question: "Dropout with p=0.5 is applied during training. Why are activations scaled by 1/(1−p) = 2× during training (inverted dropout)?",
    options: [
      "To compensate for the reduced learning rate caused by zeroed units",
      "So that the expected activation value is the same as without dropout, meaning inference requires no scaling",
      "To increase the gradient magnitude and avoid vanishing gradients",
      "To halve the effective batch size and improve generalisation",
    ],
    answer: 1,
    explanation:
      "Without scaling, the expected output of a unit with p=0.5 dropout would be 0.5× its undropped value. At inference (no dropout), the full activation is used, causing a 2× mismatch. Inverted dropout scales surviving units by 2× during training, so the expected value always equals the undropped activation — inference needs no special handling.",
  },
  {
    question: "Batch Normalization uses batch statistics (μ_B, σ²_B) during training but running averages at inference. Why can't it use batch statistics at inference?",
    options: [
      "Batch statistics are only accurate when batch size > 32",
      "At inference, samples are typically processed one at a time (batch size = 1), making batch statistics meaningless or unavailable; the running averages accumulated during training are used instead",
      "Batch statistics cause gradient explosions at inference time",
      "The γ and β parameters are not available outside of training",
    ],
    answer: 1,
    explanation:
      "Batch Norm normalises using the mean and variance of the current mini-batch. At inference we often predict single samples, making batch statistics undefined or noisy. During training, running exponential moving averages of μ and σ² are maintained; at test time these frozen population statistics are used instead.",
  },
  {
    question: "Layer Normalization is preferred over Batch Normalization in Transformers primarily because:",
    options: [
      "Layer Norm is computationally cheaper than Batch Norm",
      "In sequence models, each position has a different meaning; normalising over the feature dimension (LN) is well-defined for any batch size including 1, and doesn't mix statistics across positions",
      "Layer Norm eliminates the need for residual connections",
      "Batch Norm cannot be applied to embeddings with dimension > 512",
    ],
    answer: 1,
    explanation:
      "Batch Norm computes statistics over the batch dimension, which conflates different tokens and sequences. For variable-length sequences, RNNs (one step at a time), or batch size = 1 (common in inference and RL), batch statistics are unreliable. Layer Norm computes statistics across the feature vector of each individual token — fully independent of batch size and sequence length.",
  },
  {
    question: "The 'double descent' phenomenon in deep learning refers to:",
    options: [
      "Training loss decreasing twice — once per epoch boundary",
      "As model size increases beyond interpolation threshold, test error begins to decrease again after its peak, reaching lower error than classical models",
      "The requirement to decrease the learning rate twice during training",
      "Validation error oscillating between two values before convergence",
    ],
    answer: 1,
    explanation:
      "Classical bias-variance theory predicts a U-shaped test error curve (under → good → overfit). Double descent (Belkin et al., 2019) shows that when model capacity crosses the interpolation threshold (can fit training data exactly), test error peaks but then falls again as capacity grows further. Modern over-parameterised deep networks operate in this second descent regime.",
  },
]

// ── SVG Diagrams ──────────────────────────────────────────────────────────────

function BiasVarianceDiagram({ theme }) {
  const dark = theme === 'dark'
  const W = 340, H = 140, padL = 36, padR = 16, padT = 10, padB = 30

  // Stylised curves
  const n = 60
  const xs = Array.from({ length: n }, (_, i) => i / (n - 1))

  // Bias (decreasing): sqrt-like decay
  const bias = xs.map(x => 0.85 * Math.exp(-3.5 * x) + 0.05)
  // Variance (increasing): logistic-like growth
  const variance = xs.map(x => 0.05 + 0.8 * (1 / (1 + Math.exp(-10 * (x - 0.55)))))
  // Total = bias² + variance (scaled for display)
  const total = xs.map((_, i) => Math.sqrt(bias[i] ** 2 + variance[i] ** 2))

  const minTotalIdx = total.indexOf(Math.min(...total))

  function toSvgX(x) { return padL + x * (W - padL - padR) }
  function toSvgY(y) { return padT + (H - padT - padB) * (1 - Math.min(y, 1)) }

  function polyline(arr, color, dash = '') {
    const pts = arr.map((v, i) => `${toSvgX(xs[i])},${toSvgY(v)}`).join(' ')
    return <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeDasharray={dash} />
  }

  const sweetX = toSvgX(xs[minTotalIdx])

  return (
    <div className="my-4 overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H + 28}`} className="w-full max-w-md mx-auto block" style={{ minWidth: 280 }}>
        {/* Sweet-spot line */}
        <line x1={sweetX} y1={padT} x2={sweetX} y2={H - padB}
          stroke={dark ? '#475569' : '#cbd5e1'} strokeWidth="1" strokeDasharray="4 3" />
        <text x={sweetX + 3} y={padT + 10} fontSize="9" fill={dark ? '#64748b' : '#94a3b8'}>optimal</text>

        {/* Axes */}
        <line x1={padL} y1={padT} x2={padL} y2={H - padB}
          stroke={dark ? '#334155' : '#e2e8f0'} strokeWidth="1" />
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB}
          stroke={dark ? '#334155' : '#e2e8f0'} strokeWidth="1" />

        {polyline(bias,     '#6366f1')}
        {polyline(variance, '#ef4444')}
        {polyline(total,    '#10b981', '6 2')}

        {/* Axis labels */}
        <text x={(padL + W - padR) / 2} y={H + 2} textAnchor="middle" fontSize="10"
          fill={dark ? '#64748b' : '#94a3b8'}>Model complexity →</text>
        <text x={padL - 6} y={(padT + H - padB) / 2} textAnchor="middle" fontSize="10"
          fill={dark ? '#64748b' : '#94a3b8'} transform={`rotate(-90, ${padL - 14}, ${(padT + H - padB) / 2})`}>Error</text>

        {/* Legend */}
        {[['#6366f1','Bias²'], ['#ef4444','Variance'], ['#10b981','Total (test)']].map(([c, l], i) => (
          <g key={l}>
            <rect x={padL + i * 100} y={H + 10} width="10" height="10" rx="2" fill={c} />
            <text x={padL + i * 100 + 14} y={H + 19} fontSize="10" fill={dark ? '#cbd5e1' : '#475569'}>{l}</text>
          </g>
        ))}
      </svg>
    </div>
  )
}

function DoubleDescent({ theme }) {
  const dark = theme === 'dark'
  const W = 340, H = 140, padL = 36, padR = 16, padT = 10, padB = 30

  const n = 80
  const xs = Array.from({ length: n }, (_, i) => i / (n - 1))

  // Classical U-shape then second descent
  function testError(t) {
    if (t < 0.45) return 0.7 - 0.5 * t          // decreasing
    if (t < 0.55) return 0.47 + 2.0 * (t - 0.45) // sharp rise at interpolation threshold
    return 0.47 + 0.2 * Math.exp(-5 * (t - 0.55)) // second descent
  }
  // Train error: monotonically decreasing
  function trainError(t) { return Math.max(0.02, 0.7 * Math.exp(-4 * t)) }

  const testVals  = xs.map(testError)
  const trainVals = xs.map(trainError)

  function toSvgX(x) { return padL + x * (W - padL - padR) }
  function toSvgY(y) { return padT + (H - padT - padB) * (1 - Math.min(y / 1.1, 1)) }
  function polyline(arr, color, dash = '') {
    const pts = arr.map((v, i) => `${toSvgX(xs[i])},${toSvgY(v)}`).join(' ')
    return <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeDasharray={dash} />
  }

  const threshX = toSvgX(0.5)

  return (
    <div className="my-4 overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H + 28}`} className="w-full max-w-md mx-auto block" style={{ minWidth: 280 }}>
        {/* Interpolation threshold */}
        <line x1={threshX} y1={padT} x2={threshX} y2={H - padB}
          stroke={dark ? '#f59e0b' : '#d97706'} strokeWidth="1" strokeDasharray="4 3" />
        <text x={threshX + 3} y={padT + 10} fontSize="8.5" fill={dark ? '#fbbf24' : '#d97706'}>interpolation</text>
        <text x={threshX + 3} y={padT + 20} fontSize="8.5" fill={dark ? '#fbbf24' : '#d97706'}>threshold</text>

        {/* Axes */}
        <line x1={padL} y1={padT} x2={padL} y2={H - padB}
          stroke={dark ? '#334155' : '#e2e8f0'} strokeWidth="1" />
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB}
          stroke={dark ? '#334155' : '#e2e8f0'} strokeWidth="1" />

        {polyline(trainVals, '#6366f1')}
        {polyline(testVals,  '#ef4444')}

        <text x={(padL + W - padR) / 2} y={H + 2} textAnchor="middle" fontSize="10"
          fill={dark ? '#64748b' : '#94a3b8'}>Model capacity / #params →</text>

        {[['#6366f1','Train error'], ['#ef4444','Test error']].map(([c, l], i) => (
          <g key={l}>
            <rect x={padL + i * 120} y={H + 10} width="10" height="10" rx="2" fill={c} />
            <text x={padL + i * 120 + 14} y={H + 19} fontSize="10" fill={dark ? '#cbd5e1' : '#475569'}>{l}</text>
          </g>
        ))}
      </svg>
    </div>
  )
}

function DropoutDiagram({ theme }) {
  const dark = theme === 'dark'
  const units = [0, 1, 2, 3, 4]
  const dropped = [1, 3]         // indices zeroed out in this illustration
  const W = 320, H = 80

  return (
    <div className="my-4 overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H + 24}`} className="w-full max-w-sm mx-auto block" style={{ minWidth: 260 }}>
        {/* Layer label */}
        <text x={W / 2} y={14} textAnchor="middle" fontSize="11" fontWeight="700"
          fill={dark ? '#94a3b8' : '#64748b'}>Hidden Layer — 50% Dropout</text>

        {units.map(i => {
          const cx = 40 + i * 56
          const isDrop = dropped.includes(i)
          return (
            <g key={i}>
              <circle cx={cx} cy={52} r={18}
                fill={isDrop ? (dark ? '#1e293b' : '#f1f5f9') : (dark ? '#4338ca' : '#6366f1')}
                stroke={isDrop ? (dark ? '#334155' : '#cbd5e1') : (dark ? '#818cf8' : '#4338ca')}
                strokeWidth={isDrop ? '1.5' : '2'}
                opacity={isDrop ? 0.4 : 1}
                strokeDasharray={isDrop ? '4 2' : 'none'}
              />
              <text cx={cx} cy={52} x={cx} y={56} textAnchor="middle" fontSize="10" fontWeight="700"
                fill={isDrop ? (dark ? '#475569' : '#94a3b8') : '#fff'}
                opacity={isDrop ? 0.5 : 1}>
                {isDrop ? '✕' : `h${i+1}`}
              </text>
              {!isDrop && (
                <text x={cx} y={82} textAnchor="middle" fontSize="9"
                  fill={dark ? '#818cf8' : '#4338ca'}>×2</text>
              )}
            </g>
          )
        })}

        <text x={W / 2} y={H + 20} textAnchor="middle" fontSize="10"
          fill={dark ? '#475569' : '#94a3b8'}>
          Dashed = zeroed; ×2 = inverted-dropout rescaling
        </text>
      </svg>
    </div>
  )
}

function BatchNormVsLayerNorm({ theme }) {
  const dark = theme === 'dark'
  // Grid: rows = batch, cols = features
  const B = 4, D = 6
  const W = 320, H = 110
  const cellW = (W - 60) / D
  const cellH = (H - 30) / B

  const accent = {
    bn: dark ? 'rgba(99,102,241,0.55)' : 'rgba(99,102,241,0.45)',
    ln: dark ? 'rgba(16,185,129,0.55)' : 'rgba(16,185,129,0.45)',
  }

  return (
    <div className="my-4 overflow-x-auto">
      <svg viewBox={`0 0 ${W * 2 + 20} ${H + 40}`} className="w-full max-w-2xl mx-auto block" style={{ minWidth: 400 }}>
        {/* ── BatchNorm panel ── */}
        <text x={W / 2} y={14} textAnchor="middle" fontSize="12" fontWeight="700"
          fill={dark ? '#818cf8' : '#4338ca'}>Batch Norm</text>
        <text x={W / 2} y={26} textAnchor="middle" fontSize="9"
          fill={dark ? '#64748b' : '#94a3b8'}>normalise ↓ over BATCH dim</text>

        {Array.from({ length: B }, (_, r) =>
          Array.from({ length: D }, (_, c) => (
            <rect key={`${r}-${c}`}
              x={30 + c * cellW} y={32 + r * cellH}
              width={cellW - 2} height={cellH - 2}
              fill={dark ? '#1e293b' : '#f8fafc'}
              stroke={dark ? '#334155' : '#e2e8f0'}
              strokeWidth="0.8" rx="2" />
          ))
        )}
        {/* Highlight one column (one feature across all batch items) */}
        {Array.from({ length: B }, (_, r) => (
          <rect key={`bn-${r}`}
            x={30 + 2 * cellW} y={32 + r * cellH}
            width={cellW - 2} height={cellH - 2}
            fill={accent.bn} rx="2" />
        ))}
        {/* Axis labels */}
        <text x={14} y={32 + (B * cellH) / 2} textAnchor="middle" fontSize="9"
          fill={dark ? '#64748b' : '#94a3b8'} transform={`rotate(-90, 10, ${32 + (B * cellH) / 2})`}>batch</text>
        <text x={30 + (D * cellW) / 2} y={32 + B * cellH + 14} textAnchor="middle" fontSize="9"
          fill={dark ? '#64748b' : '#94a3b8'}>features</text>

        {/* ── LayerNorm panel ── */}
        <text x={W + 10 + W / 2} y={14} textAnchor="middle" fontSize="12" fontWeight="700"
          fill={dark ? '#34d399' : '#059669'}>Layer Norm</text>
        <text x={W + 10 + W / 2} y={26} textAnchor="middle" fontSize="9"
          fill={dark ? '#64748b' : '#94a3b8'}>normalise → over FEATURE dim</text>

        {Array.from({ length: B }, (_, r) =>
          Array.from({ length: D }, (_, c) => (
            <rect key={`${r}-${c}-ln`}
              x={W + 10 + 30 + c * cellW} y={32 + r * cellH}
              width={cellW - 2} height={cellH - 2}
              fill={dark ? '#1e293b' : '#f8fafc'}
              stroke={dark ? '#334155' : '#e2e8f0'}
              strokeWidth="0.8" rx="2" />
          ))
        )}
        {/* Highlight one row (one sample across all features) */}
        {Array.from({ length: D }, (_, c) => (
          <rect key={`ln-${c}`}
            x={W + 10 + 30 + c * cellW} y={32 + 1 * cellH}
            width={cellW - 2} height={cellH - 2}
            fill={accent.ln} rx="2" />
        ))}
        <text x={W + 10 + 14} y={32 + (B * cellH) / 2} textAnchor="middle" fontSize="9"
          fill={dark ? '#64748b' : '#94a3b8'} transform={`rotate(-90, ${W + 10 + 10}, ${32 + (B * cellH) / 2})`}>batch</text>
        <text x={W + 10 + 30 + (D * cellW) / 2} y={32 + B * cellH + 14} textAnchor="middle" fontSize="9"
          fill={dark ? '#64748b' : '#94a3b8'}>features</text>

        <text x={W * 2 + 10} y={H + 32} textAnchor="end" fontSize="9"
          fill={dark ? '#475569' : '#94a3b8'}>highlighted = normalisation group</text>
      </svg>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function RegularizationDeepModels() {
  const { theme, markTopicComplete } = useApp()
  const dark = theme === 'dark'

  return (
    <div className="space-y-10">
      {/* ── Header ── */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3 mb-3">
          <span className="text-4xl">🛡️</span>
          <div>
            <p className={`text-sm font-semibold uppercase tracking-widest ${dark ? 'text-violet-400' : 'text-violet-600'}`}>
              Session 15 · Deep Learning
            </p>
            <h1 className={`text-3xl font-black ${dark ? 'text-white' : 'text-slate-900'}`}>
              Regularization for Deep Models
            </h1>
          </div>
        </div>
        <p className={`text-lg leading-relaxed ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          A model that memorises training data is useless in deployment. This session covers the
          theory of <strong>generalisation</strong> and the practical techniques — weight decay,
          dropout, and normalisation layers — that close the gap between training and test performance.
        </p>

        <div className={`mt-4 rounded-xl p-4 border text-sm ${dark ? 'bg-amber-900/20 border-amber-700/40 text-amber-200' : 'bg-amber-50 border-amber-200 text-amber-800'}`}>
          <p className="font-bold mb-1">📚 Prerequisites</p>
          <ul className="list-disc list-inside space-y-0.5 text-xs">
            <li><strong>Session 14</strong> — Loss surfaces, SGD variants, mini-batch training</li>
            <li><strong>Backpropagation</strong> — Gradient flow, chain rule, weight updates</li>
            <li><strong>Session 12</strong> — Residual connections, LayerNorm in Transformers</li>
          </ul>
        </div>
      </motion.div>

      {/* ══════════════════════════════════════════
          Section 15.1 — Generalization & Model Selection
         ══════════════════════════════════════════ */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 15.1</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Generalization and Model Selection</h2>
        </div>

        {/* 15.1.1 Training vs. generalisation error */}
        <h3 className={`text-lg font-bold mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          15.1.1 — Training Error vs. Generalization Error
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The <strong>training error</strong> measures how well the model fits the training set —
          the data it was optimised on. The <strong>generalisation error</strong> (test error) measures
          performance on unseen data drawn from the same distribution. The goal of machine learning is
          to minimise generalisation error, not training error.
        </p>

        <Callout type="formula" title="Generalisation gap">
          Generalisation gap = E_test[L(f(x;θ), y)] − (1/N) Σᵢ L(f(xᵢ;θ), yᵢ)
        </Callout>

        <TheoryBlock items={[
          {
            title: 'Why they differ',
            content: 'The model was optimised on training data — it has "seen" those examples. Test data is unseen. Any pattern fit that is specific to training noise (not the true signal) increases the gap.',
          },
          {
            title: 'i.i.d. assumption',
            content: 'Standard theory assumes training and test data are drawn independently from the same distribution (i.i.d.). When this holds, a low training error + low gap = good generalisation.',
          },
          {
            title: 'Held-out test set',
            content: 'Never use the test set to make modelling decisions. It must remain unseen until final evaluation. Use a validation set (or cross-validation) for model selection.',
          },
          {
            title: 'Early stopping',
            content: 'Monitor validation loss during training; stop when it stops improving. Prevents the model from over-optimising the training set. The validation loss is a proxy for generalisation error.',
          },
        ]} />

        {/* 15.1.2 Underfitting / Overfitting */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          15.1.2 — Underfitting (High Bias) and Overfitting (High Variance)
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The <strong>bias-variance trade-off</strong> characterises the two failure modes of a learned model.
          A model with too little capacity cannot capture the true signal (underfitting). A model with too much
          capacity fits both signal and noise (overfitting). Optimal capacity sits between these extremes.
        </p>

        <BiasVarianceDiagram theme={theme} />

        <TheoryBlock cols={2} items={[
          {
            title: '📐 Underfitting — High Bias',
            content: 'Model is too simple to capture the data structure. High training error AND high test error. Symptom: training loss plateaus early. Fix: increase capacity, add features, train longer.',
          },
          {
            title: '📈 Overfitting — High Variance',
            content: 'Model memorises training noise. Very low training error, high test error. Symptom: training loss much lower than validation loss. Fix: regularisation, more data, simpler model.',
          },
          {
            title: 'Bias²',
            content: 'Systematic error from wrong assumptions. Average squared distance between expected prediction and true value. High bias → consistently wrong in the same direction.',
          },
          {
            title: 'Variance',
            content: 'Sensitivity of the prediction to fluctuations in the training set. High variance → different training sets produce wildly different models. Captures model instability.',
          },
        ]} />

        <CodeBlock code={CODE_BIAS_VARIANCE} language="python" title="bias_variance.py" />

        {/* 15.1.3 Model selection */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          15.1.3 — Model Selection: Validation Set and Cross-Validation
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Choosing among model architectures, hyperparameters, or regularisation strengths requires
          evaluating each candidate on data that was not used for training — but also not "saved" as
          the final test set. A <strong>validation set</strong> or <strong>k-fold cross-validation</strong>
          serves this purpose.
        </p>

        <TheoryBlock cols={2} items={[
          {
            title: '✂️ Train / Val / Test split',
            content: 'Common split: 60/20/20 or 70/15/15. Train → fit model. Val → select hyperparameters. Test → report final performance once. Never tune on test.',
          },
          {
            title: '🔁 k-Fold Cross-Validation',
            content: 'Split data into k folds. Train on k-1, validate on the remaining fold. Rotate k times. Average validation score. More reliable than a single split, especially for small datasets.',
          },
          {
            title: 'Leave-One-Out (LOO)',
            content: 'Extreme case of k-fold with k=N. Maximally uses data for training. Very expensive for large N. Useful for tiny datasets.',
          },
          {
            title: 'Nested CV',
            content: 'Outer loop: model evaluation. Inner loop: hyperparameter search. Prevents optimistic bias when both selecting and evaluating on the same validation set.',
          },
        ]} />

        <CodeBlock code={CODE_CROSS_VALIDATION} language="python" title="cross_validation.py" />

        {/* 15.1.4 Distribution shift */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          15.1.4 — Environment and Distribution Shift
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The i.i.d. assumption often fails in deployment. <strong>Distribution shift</strong> occurs when
          the test distribution differs from the training distribution. The model's training-time guarantees
          no longer hold.
        </p>

        <TheoryBlock items={[
          {
            title: '🔀 Covariate shift',
            content: 'Input distribution P(X) changes but the conditional P(Y|X) stays the same. Example: training on daytime images, deploying on nighttime images. Importance weighting can correct for this.',
          },
          {
            title: '🔄 Label shift (prior probability shift)',
            content: 'Output distribution P(Y) changes but P(X|Y) stays the same. Example: training on a balanced class dataset, deploying where one class dominates. Calibration methods apply.',
          },
          {
            title: '🌍 Concept drift',
            content: 'P(Y|X) itself changes over time. Example: spam patterns evolve, language meaning shifts. Requires periodic retraining or online learning.',
          },
          {
            title: '🏥 Domain generalisation',
            content: 'Training on multiple source domains; deploy on a new unseen domain. Goal: learn representations invariant to domain-specific nuisance factors. Active research area.',
          },
        ]} />

        {/* 15.1.5 Double descent */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          15.1.5 — Generalization in Deep Learning: Double Descent
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Classical statistical learning theory predicts a U-shaped test error curve as model complexity
          grows. Modern deep learning violates this: as model size grows beyond the point where it can
          perfectly fit the training data (<strong>interpolation threshold</strong>), test error rises
          sharply — then falls again in the <em>second descent</em> regime.
        </p>

        <DoubleDescent theme={theme} />

        <TheoryBlock items={[
          {
            title: 'Interpolation threshold',
            content: 'The point where the model can fit the training data exactly (zero training loss). Classical theory says this is where overfitting peaks. Double descent shows it is just a transition.',
          },
          {
            title: 'Over-parameterised regime',
            content: 'Beyond the interpolation threshold, many solutions perfectly fit the training data. Implicit regularisation (e.g., SGD noise, minimum-norm solutions) selects among them, often finding generalisable ones.',
          },
          {
            title: 'Implicit regularisation of SGD',
            content: 'SGD with small learning rate finds minimum-norm solutions. These solutions tend to be flatter and generalise better. This is why extremely large models (GPT-4, etc.) generalise despite zero training loss.',
          },
          {
            title: 'Epoch-wise double descent',
            content: 'The same phenomenon can appear along the training epoch axis (not just model size). Test loss rises temporarily before eventually falling — motivation for training longer than the loss plateau.',
          },
        ]} />

        <DeepDive title="Why over-parameterisation helps: benign overfitting">
          <TheoryBlock items={[
            {
              title: '',
              content: 'Bartlett et al. (2020) showed that in linear regression with more features than samples, if the signal lies in a low-dimensional subspace and the remaining directions have small weights, the model interpolates training data while still generalising. The "extra" parameters absorb noise without disturbing the signal. Deep networks exhibit an analogous structure: most parameters fit the training data while a small effective set captures the true signal — a form of implicit dimensionality reduction.',
            },
          ]} />
        </DeepDive>
      </motion.div>

      {/* ══════════════════════════════════════════
          Section 15.2 — Regularization Techniques
         ══════════════════════════════════════════ */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-violet-500/5 border-violet-500/20' : 'bg-violet-50 border-violet-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-violet-400' : 'text-violet-600'}`}>Section 15.2</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Regularization Techniques</h2>
        </div>

        <p className={`mb-6 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Regularisation encompasses any technique that reduces generalisation error without reducing
          training error (or reduces both but test error more). The goal is to shape the hypothesis
          space so that solutions with good inductive biases are preferred.
        </p>

        {/* 15.2.1 Weight Decay */}
        <h3 className={`text-lg font-bold mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          15.2.1 — Weight Decay: L2 Regularization
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          L2 regularisation adds a penalty proportional to the <strong>squared magnitude of the weights</strong>
          to the loss. This encourages the optimiser to prefer solutions with small weights — which correspond
          to smoother, less complex functions that are less likely to memorise noise.
        </p>

        <Callout type="formula" title="L2-regularised loss and gradient">
          L_reg = L + λ‖w‖² &nbsp;&nbsp;&nbsp; ∂L_reg/∂w = ∂L/∂w + 2λw &nbsp;&nbsp;&nbsp;
          update: w ← (1 − 2ηλ)·w − η·∂L/∂w
        </Callout>

        <TheoryBlock cols={2} items={[
          {
            title: 'L1 vs. L2 regularisation',
            content: 'L1 (‖w‖₁) produces sparse weights (many exactly zero) — useful for feature selection. L2 (‖w‖²) shrinks all weights smoothly toward zero but rarely makes them exactly zero. Deep learning almost exclusively uses L2 (weight decay).',
          },
          {
            title: 'Effect on loss surface',
            content: 'L2 adds a bowl-shaped ‖w‖² term to the loss. The minimum of the regularised loss is pulled toward the origin. Larger λ → minimum pulled further from the unregularised solution.',
          },
          {
            title: 'λ selection',
            content: 'Tuned on the validation set. Typical range: 1e-4 to 1e-2 for deep networks. Too small → no regularisation effect. Too large → underfitting (all weights near zero).',
          },
          {
            title: 'AdamW — decoupled weight decay',
            content: 'Standard Adam applies L2 via the gradient, conflating it with adaptive scaling. AdamW applies weight decay directly to the parameters (w ← w − η·λ·w) after the Adam gradient step, giving correct regularisation independent of the adaptive scaling.',
          },
        ]} />

        <CodeBlock code={CODE_WEIGHT_DECAY} language="python" title="weight_decay.py" />

        {/* 15.2.2 Dropout */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          15.2.2 — Dropout: Random Unit Deactivation During Training
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Dropout (Srivastava et al., 2014) randomly sets a fraction <strong>p</strong> of neuron
          activations to zero during each forward pass. A different random subset is dropped each
          mini-batch. This prevents neurons from <em>co-adapting</em> — relying on specific other
          neurons to be active — forcing the network to learn redundant, more robust representations.
        </p>

        <DropoutDiagram theme={theme} />

        <Callout type="formula" title="Dropout forward pass (inverted dropout)">
          mask ~ Bernoulli(1−p) &nbsp;&nbsp; h_out = (h · mask) / (1−p) &nbsp;&nbsp; [train only]
        </Callout>

        <TheoryBlock items={[
          {
            title: '🎲 Ensemble interpretation',
            content: 'Each forward pass samples a different sub-network of 2^N possible architectures (N = number of units). Inference with all units active approximates averaging the predictions of this exponentially large ensemble.',
          },
          {
            title: '🔢 Inverted dropout scaling',
            content: 'Surviving units are scaled by 1/(1-p) during training. This keeps the expected activation magnitude constant. At inference, no scaling is needed — weights are already correctly calibrated.',
          },
          {
            title: '📍 Where to apply',
            content: 'Typically applied after fully-connected layers (p=0.5) or after non-linearities. In Transformers, applied after attention and FFN sublayers (p=0.1). Not used in convolutional layers (spatial dropout preferred).',
          },
          {
            title: '⚠️ Dropout + Batch Norm conflict',
            content: 'Dropout introduces variance in activations that Batch Norm statistics depend on. Using both together can lead to "variance shift" at inference. Common fix: place BN before dropout, or use Layer Norm instead.',
          },
        ]} />

        <CodeBlock code={CODE_DROPOUT} language="python" title="dropout.py" />

        <DeepDive title="MC Dropout — uncertainty estimation at inference">
          <TheoryBlock items={[
            {
              title: '',
              content: 'Gal & Ghahramani (2016) showed that running dropout at inference time T times and averaging the T predictions approximates Bayesian inference in a Gaussian Process. The variance across T predictions estimates predictive uncertainty. This is "Monte Carlo Dropout" — a practical uncertainty quantification tool with no architectural changes beyond keeping dropout active at test time.',
            },
          ]} />
        </DeepDive>

        {/* 15.2.3 Batch Normalisation */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          15.2.3 — Batch Normalization: Normalizing Activations Per Mini-Batch
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Batch Normalization (Ioffe &amp; Szegedy, 2015) normalises each feature to have zero mean and
          unit variance across the current mini-batch, then re-scales with learned parameters γ and β.
          It dramatically accelerates training by reducing <strong>internal covariate shift</strong> — the
          change in activation distributions caused by parameter updates in earlier layers.
        </p>

        <Callout type="formula" title="Batch Norm forward pass">
          μ_B = (1/B) Σ xᵢ &nbsp;&nbsp; σ²_B = (1/B) Σ (xᵢ − μ_B)² &nbsp;&nbsp;
          x̂ᵢ = (xᵢ − μ_B) / √(σ²_B + ε) &nbsp;&nbsp; yᵢ = γ·x̂ᵢ + β
        </Callout>

        <TheoryBlock items={[
          {
            title: 'γ and β — learnable affine transform',
            content: 'After normalisation the network can recover any distribution by learning γ (scale) and β (shift). This means BN can undo the normalisation if needed — it does not restrict expressive power.',
          },
          {
            title: 'Running statistics at inference',
            content: 'During training, exponential moving averages of μ and σ² are maintained: μ_run ← 0.9·μ_run + 0.1·μ_B. At inference, these frozen population statistics are used — batch size = 1 is fine.',
          },
          {
            title: 'Regularisation effect',
            content: 'BN acts as a mild regulariser because each sample is normalised relative to other batch members — adding stochastic noise. This often reduces or eliminates the need for dropout in CNNs.',
          },
          {
            title: 'Placement: pre- vs. post-activation',
            content: 'Original paper: after linear layer, before activation (pre-activation). Modern practice: sometimes after activation. Pre-activation residual networks (He et al., 2016) apply BN before the non-linearity.',
          },
        ]} />

        <CodeBlock code={CODE_BATCHNORM} language="python" title="batch_norm.py" />

        {/* 15.2.4 Layer Normalisation */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          15.2.4 — Layer Normalization: Normalizing Across Features
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Layer Normalization (Ba et al., 2016) computes mean and variance across the <strong>feature
          dimension</strong> of a single sample — entirely independently of other samples in the batch.
          This makes it batch-size agnostic and the natural choice for sequence models and Transformers.
        </p>

        <Callout type="formula" title="Layer Norm forward pass">
          μ = (1/D) Σ_d x_d &nbsp;&nbsp; σ² = (1/D) Σ_d (x_d − μ)² &nbsp;&nbsp;
          x̂_d = (x_d − μ) / √(σ² + ε) &nbsp;&nbsp; y_d = γ_d·x̂_d + β_d
        </Callout>

        <TheoryBlock items={[
          {
            title: 'Batch-independent',
            content: 'Statistics are computed per-sample, over the feature vector. No cross-sample dependencies. Works identically for batch size = 1 or variable-length sequences. No running averages needed at inference.',
          },
          {
            title: 'Used in Transformers',
            content: 'BERT, GPT, T5, and virtually all Transformer-based models use Layer Norm. Each token\'s embedding vector (size d_model) is normalised independently. Sequence length and batch size have no effect.',
          },
          {
            title: 'RNN training',
            content: 'BN is awkward for RNNs (different sequence lengths, one-step-at-a-time processing). Layer Norm applies identically at every timestep regardless of sequence length.',
          },
          {
            title: 'γ and β are per-feature',
            content: 'Shape: (D,). One scale and shift per feature dimension. The model can learn to amplify or suppress specific dimensions while keeping the normalised representation.',
          },
        ]} />

        <CodeBlock code={CODE_LAYERNORM} language="python" title="layer_norm.py" />

        {/* 15.2.5 BatchNorm vs LayerNorm comparison */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          15.2.5 — Comparison: Batch Norm vs. Layer Norm
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The core difference is the <strong>axis of normalisation</strong>. Batch Norm operates across
          the batch dimension (for one feature, across all samples). Layer Norm operates across the feature
          dimension (for one sample, across all features).
        </p>

        <BatchNormVsLayerNorm theme={theme} />

        <div className="overflow-x-auto mb-6">
          <table className={`w-full text-sm border-collapse rounded-xl overflow-hidden ${dark ? 'text-slate-300' : 'text-slate-700'}`}>
            <thead>
              <tr className={dark ? 'bg-slate-800' : 'bg-slate-100'}>
                {['Property', 'Batch Norm', 'Layer Norm'].map(h => (
                  <th key={h} className={`px-3 py-2.5 text-left text-xs font-bold uppercase tracking-wide ${dark ? 'text-slate-400' : 'text-slate-500'}`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['Normalisation axis',    'Batch (B)',                     'Feature (D)'],
                ['Requires large batch',  '✅ (≥ 16–32 recommended)',      '❌ (any batch, even 1)'],
                ['Running averages',      '✅ (needed at inference)',       '❌ (none needed)'],
                ['Use in CNNs',           '✅ (standard choice)',           '❌ (rarely used)'],
                ['Use in Transformers',   '❌ (problematic)',               '✅ (universal choice)'],
                ['Use in RNNs',           '⚠️ (awkward)',                  '✅ (standard)'],
                ['Regularisation effect', 'Yes (stochastic batch noise)',   'Minimal'],
                ['Learnable params',      'γ, β per feature channel',       'γ, β per feature dim'],
              ].map((row, i) => (
                <tr key={i} className={`border-t ${dark ? 'border-slate-700/50 hover:bg-slate-800/50' : 'border-slate-100 hover:bg-slate-50'} transition-colors`}>
                  {row.map((cell, j) => (
                    <td key={j} className={`px-3 py-2 ${j === 0 ? 'font-bold' : ''}`}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <Callout type="info" title="Practical decision guide">
          CNN on vision tasks → Batch Norm.
          Transformer / NLP / variable-length sequences → Layer Norm.
          RNN → Layer Norm.
          Small batch (≤ 4) or online RL → Layer Norm or Group Norm.
          Want regularisation effect → Batch Norm (or add Dropout).
        </Callout>

        <DeepDive title="Instance Norm, Group Norm — when neither BN nor LN fits">
          <TheoryBlock items={[
            {
              title: 'Instance Norm',
              content: 'Normalises each sample × each channel independently (BN with B=1 per channel). Used in style transfer — removes style (global statistics) while preserving content (relative spatial structure).',
            },
            {
              title: 'Group Norm',
              content: 'Divides channels into G groups; normalises within each group. Interpolates between LN (G=1) and IN (G=C). More robust than BN for small batches. Preferred in object detection (FPN, Mask R-CNN) where batch size = 1–2 per GPU.',
            },
          ]} />
        </DeepDive>
      </motion.div>

      {/* ── Quiz ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <Quiz
          questions={QUIZ_QUESTIONS}
          onComplete={() => markTopicComplete('regularization-deep-models')}
        />
      </motion.div>
    </div>
  )
}
