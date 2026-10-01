import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

// ── Code blocks ────────────────────────────────────────────────────────────────

const CODE_LOSS_SURFACE = `import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Toy non-convex loss surface: Rastrigin function (2D) ──
# A classic benchmark with many local minima and saddle points

def rastrigin(x, y, A=1.0):
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + \
                   (y**2 - A * np.cos(2 * np.pi * y))

X = np.linspace(-3, 3, 300)
Y = np.linspace(-3, 3, 300)
Xg, Yg = np.meshgrid(X, Y)
Z = rastrigin(Xg, Yg)

fig, ax = plt.subplots(figsize=(5, 4))
cp = ax.contourf(Xg, Yg, Z, levels=30, cmap='viridis')
plt.colorbar(cp)
ax.set_title("Non-convex loss surface (Rastrigin)")
ax.set_xlabel("w₁"); ax.set_ylabel("w₂")

# Mark global minimum
ax.plot(0, 0, 'r*', markersize=14, label='Global min')
# Mark some local minima
for x0, y0 in [(1,1),(-1,1),(1,-1),(-1,-1),(2,0),(0,2)]:
    ax.plot(x0, y0, 'w^', markersize=7)

ax.legend()
plt.tight_layout()
plt.savefig('loss_surface.png', dpi=100)
print("Saved loss_surface.png")`

const CODE_VANISHING_EXPLODING = `import numpy as np

# ── Vanishing vs. Exploding gradients in a deep network ──
# Gradient magnitude after L layers of matrix multiplication

np.random.seed(0)
L   = 50   # depth
D   = 64   # dimension

# Case 1: weights initialised slightly below 1 → vanishing
W_small = np.random.randn(D, D) * (1.0 / np.sqrt(D)) * 0.5
g = np.ones(D)
norms_vanish = [np.linalg.norm(g)]
for _ in range(L):
    g = W_small.T @ g
    norms_vanish.append(np.linalg.norm(g))

# Case 2: weights initialised above 1 → exploding
W_large = np.random.randn(D, D) * (2.0 / np.sqrt(D))
g = np.ones(D)
norms_explode = [np.linalg.norm(g)]
for _ in range(L):
    g = W_large.T @ g
    norms_explode.append(np.linalg.norm(g))

print("Layer | Vanish grad | Explode grad")
print("-" * 40)
for l in [0, 5, 10, 20, 35, 49]:
    print(f"  {l:2d}  | {norms_vanish[l]:11.6f} | {norms_explode[l]:.3e}")

# Solutions: careful init (Xavier/He), BatchNorm, residual connections, gradient clipping`

const CODE_SGD_VARIANTS = `import numpy as np

# ── Batch GD vs SGD vs Mini-batch SGD ──

np.random.seed(42)
N = 1000       # dataset size
D = 2          # input dim
W_true = np.array([2.0, -1.5])

# Generate synthetic regression data: y = X·W_true + noise
X    = np.random.randn(N, D)
y    = X @ W_true + np.random.randn(N) * 0.5

def mse_grad(X_b, y_b, w):
    """MSE gradient on a batch."""
    pred = X_b @ w
    err  = pred - y_b
    return (2 / len(y_b)) * X_b.T @ err

def run(name, batch_size, epochs=5, lr=0.05):
    w = np.zeros(D)
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        for start in range(0, N, batch_size):
            b  = idx[start:start + batch_size]
            w -= lr * mse_grad(X[b], y[b], w)
        loss = np.mean((X @ w - y) ** 2)
        print(f"[{name}] epoch {epoch+1}  loss={loss:.4f}  w={np.round(w,3)}")

run("Batch GD  (B=N)",     batch_size=N,    epochs=5)
run("Mini-batch (B=64)",   batch_size=64,   epochs=5)
run("SGD       (B=1)",     batch_size=1,    epochs=2)   # noisy — fewer epochs`

const CODE_MOMENTUM = `import numpy as np

# ── SGD with Momentum ──
# v_t = β·v(t−1) + (1-β)·g_t   (EMA of gradients)
# w_t = w(t−1) - lr · v_t

np.random.seed(0)
N, D = 500, 2
X = np.random.randn(N, D)
y = X @ np.array([3.0, -2.0]) + 0.3 * np.random.randn(N)

def mse_grad(w):
    return (2/N) * X.T @ (X @ w - y)

def sgd_momentum(beta=0.9, lr=0.1, steps=30):
    w = np.zeros(D)
    v = np.zeros(D)
    print(f"\\n── Momentum β={beta}, lr={lr} ──")
    for t in range(1, steps + 1):
        g = mse_grad(w)
        v = beta * v + (1 - beta) * g   # EMA accumulation
        w = w - lr * v
        if t % 5 == 0:
            loss = np.mean((X @ w - y)**2)
            print(f"  step {t:3d}  loss={loss:.5f}  |v|={np.linalg.norm(v):.4f}")
    return w

sgd_momentum(beta=0.0)   # pure SGD: no momentum
sgd_momentum(beta=0.9)   # standard momentum
sgd_momentum(beta=0.99)  # high momentum — faster on plateaus, harder to stop`

const CODE_ADAGRAD = `import numpy as np

# ── Adagrad — per-parameter adaptive learning rate ──
# G_t  = G(t−1) + g_t²             (accumulated squared gradients)
# w_t  = w(t−1) - (η / sqrt(G_t + ε)) · g_t

np.random.seed(0)
N, D = 500, 2
X = np.random.randn(N, D)
y = X @ np.array([3.0, -2.0]) + 0.3 * np.random.randn(N)

def mse_grad(w):
    return (2/N) * X.T @ (X @ w - y)

def adagrad(lr=0.5, eps=1e-8, steps=30):
    w = np.zeros(D)
    G = np.zeros(D)   # accumulated squared gradients
    print(f"\\n── Adagrad lr={lr} ──")
    for t in range(1, steps + 1):
        g  = mse_grad(w)
        G += g ** 2
        w -= (lr / (np.sqrt(G) + eps)) * g
        if t % 5 == 0:
            loss = np.mean((X @ w - y)**2)
            lr_eff = lr / (np.sqrt(G) + eps)
            print(f"  step {t:3d}  loss={loss:.5f}  eff_lr≈{lr_eff.mean():.5f}")
    return w

adagrad()

# ── Problem: G grows monotonically → effective LR → 0 ──
# This is fine for sparse features but hurts dense problems over many steps`

const CODE_RMSPROP = `import numpy as np

# ── RMSProp — leaky Adagrad with exponential decay ──
# E[g²]_t = ρ · E[g²](t−1) + (1-ρ) · g_t²      (EMA of squared grads)
# w_t     = w(t−1) - (η / sqrt(E[g²]_t + ε)) · g_t

np.random.seed(0)
N, D = 500, 2
X = np.random.randn(N, D)
y = X @ np.array([3.0, -2.0]) + 0.3 * np.random.randn(N)

def mse_grad(w):
    return (2/N) * X.T @ (X @ w - y)

def rmsprop(rho=0.9, lr=0.1, eps=1e-8, steps=30):
    w   = np.zeros(D)
    Eg2 = np.zeros(D)   # EMA of squared gradients
    print(f"\\n── RMSProp ρ={rho}, lr={lr} ──")
    for t in range(1, steps + 1):
        g   = mse_grad(w)
        Eg2 = rho * Eg2 + (1 - rho) * g ** 2    # exponential decay!
        w  -= (lr / (np.sqrt(Eg2) + eps)) * g
        if t % 5 == 0:
            loss = np.mean((X @ w - y)**2)
            print(f"  step {t:3d}  loss={loss:.5f}")
    return w

rmsprop()   # robust on non-stationary objectives (RNNs, RL)`

const CODE_ADAM = `import numpy as np

# ── Adam — Adaptive Moment Estimation ──
# Combines Momentum (1st moment) + RMSProp (2nd moment) + bias correction
#
# m_t = β₁·m(t−1) + (1-β₁)·g_t         ← 1st moment (mean)
# v_t = β₂·v(t−1) + (1-β₂)·g_t²        ← 2nd moment (uncentred variance)
# m̂_t = m_t / (1 - β₁ᵗ)                 ← bias-corrected mean
# v̂_t = v_t / (1 - β₂ᵗ)                 ← bias-corrected variance
# w_t = w(t−1) - η · m̂_t / (√v̂_t + ε)

np.random.seed(0)
N, D = 500, 2
X = np.random.randn(N, D)
y = X @ np.array([3.0, -2.0]) + 0.3 * np.random.randn(N)

def mse_grad(w):
    return (2/N) * X.T @ (X @ w - y)

def adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, steps=40):
    w = np.zeros(D)
    m = np.zeros(D)   # 1st moment
    v = np.zeros(D)   # 2nd moment
    print(f"\\n── Adam lr={lr}, β₁={beta1}, β₂={beta2} ──")
    for t in range(1, steps + 1):
        g  = mse_grad(w)
        m  = beta1 * m + (1 - beta1) * g         # biased 1st moment
        v  = beta2 * v + (1 - beta2) * g ** 2    # biased 2nd moment
        m_hat = m / (1 - beta1 ** t)             # bias correction
        v_hat = v / (1 - beta2 ** t)             # bias correction
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if t % 5 == 0:
            loss = np.mean((X @ w - y)**2)
            print(f"  step {t:3d}  loss={loss:.5f}  w≈{np.round(w,3)}")
    return w

adam()
# Default hyperparameters (lr=1e-3, β₁=0.9, β₂=0.999) work well on most tasks`

// ── Quiz questions ─────────────────────────────────────────────────────────────

const QUIZ_QUESTIONS = [
  {
    question: "What is the key difference between Adagrad and RMSProp?",
    options: [
      "Adagrad uses a first moment estimate; RMSProp does not",
      "RMSProp uses an exponential moving average of squared gradients, preventing the learning rate from decaying to zero",
      "RMSProp accumulates all squared gradients since t=0 like Adagrad",
      "Adagrad applies bias correction; RMSProp does not",
    ],
    answer: 1,
    explanation:
      "Adagrad monotonically accumulates ALL squared gradients (G_t grows forever), so the effective learning rate eventually collapses to zero. RMSProp uses an EMA with decay ρ, so old gradients are exponentially forgotten — the effective learning rate remains non-zero.",
  },
  {
    question: "In Adam, what problem does bias correction solve?",
    options: [
      "It prevents exploding gradients in early training",
      "It compensates for the fact that m and v are initialised to zero, causing underestimates in the first steps",
      "It reduces the variance of gradient estimates across mini-batches",
      "It ensures the learning rate schedule follows a cosine annealing curve",
    ],
    answer: 1,
    explanation:
      "At t=1, m₁ = (1-β₁)·g₁ which is much smaller than g₁ (since β₁=0.9). Dividing by (1−β₁ᵗ) corrects this cold-start bias. As t→∞ the correction factor → 1 and has no effect.",
  },
  {
    question: "Why do saddle points pose a greater practical challenge than local minima in deep networks?",
    options: [
      "Saddle points have higher loss values than any local minimum",
      "At a saddle point gradients are zero but the curvature is mixed — SGD noise can escape; local minima are typically good solutions in high-dimensional spaces",
      "Local minima are exponentially more common than saddle points in high dimensions",
      "Saddle points only occur when the batch size equals 1",
    ],
    answer: 1,
    explanation:
      "In high-dimensional loss surfaces, true local minima (all directions curve up) are rare — the probability that all D eigenvalues are positive is 2^{-D}. Saddle points (mixed curvature) are exponentially more common and are where gradient-based optimisers can stall, because ∇L=0 but the point is not a minimum.",
  },
  {
    question: "Which of the following best describes the role of momentum in gradient descent?",
    options: [
      "It adapts the learning rate for each parameter independently",
      "It accumulates an exponential moving average of past gradients to build up velocity, smoothing oscillations and accelerating convergence in consistent directions",
      "It resets the gradient to zero when the gradient sign flips",
      "It applies weight decay to all parameters at each step",
    ],
    answer: 1,
    explanation:
      "Momentum maintains a velocity vector v_t = β·v(t−1) + (1-β)·g_t. In directions where gradients consistently point the same way, v grows; in oscillating directions, opposing gradients cancel. This damps zig-zag behaviour (e.g. narrow ravines) and speeds up progress along flat valleys.",
  },
  {
    question: "Mini-batch SGD is preferred over full-batch gradient descent primarily because:",
    options: [
      "It always converges to a lower loss than batch GD",
      "It provides noisy gradient estimates that help escape sharp minima and allows parameter updates much more frequently per epoch",
      "It requires no learning rate schedule",
      "It eliminates the need for data shuffling",
    ],
    answer: 1,
    explanation:
      "Full-batch GD computes an exact gradient but updates weights only once per epoch and is expensive for large datasets. Mini-batch updates are noisier (which is regularising) but happen N/B times per epoch. The noise also helps escape sharp minima, and modern hardware (GPUs) is optimised for batched matrix operations.",
  },
]

// ── Diagram: optimizer trajectories (SVG) ─────────────────────────────────────

function OptimizerTrajectoryDiagram({ theme }) {
  const dark = theme === 'dark'

  // Contour ellipse parameters — simulating a narrow ravine (elongated loss surface)
  const cx = 200, cy = 130
  // Trajectories: [name, color, points]
  const trajectories = [
    {
      name: 'Batch GD',
      color: '#64748b',
      pts: [[40,40],[80,90],[110,108],[140,118],[165,124],[185,128],[198,130]],
    },
    {
      name: 'SGD (noisy)',
      color: '#f59e0b',
      pts: [[40,40],[65,75],[95,65],[120,95],[105,115],[140,110],[160,120],[180,125],[198,130]],
    },
    {
      name: 'Momentum',
      color: '#6366f1',
      pts: [[40,40],[90,95],[130,112],[160,122],[185,128],[198,130]],
    },
    {
      name: 'Adam',
      color: '#10b981',
      pts: [[40,40],[100,105],[145,118],[175,126],[198,130]],
    },
  ]

  return (
    <div className="my-4 overflow-x-auto">
      <svg viewBox="0 0 400 260" className="w-full max-w-lg mx-auto block" style={{ minWidth: 300 }}>
        {/* Contour ellipses — loss surface */}
        {[1.0, 0.75, 0.5, 0.3, 0.15].map((s, i) => (
          <ellipse key={i}
            cx={cx} cy={cy}
            rx={180 * s} ry={80 * s}
            fill="none"
            stroke={dark ? `rgba(99,102,241,${0.12 + i*0.04})` : `rgba(99,102,241,${0.1 + i*0.04})`}
            strokeWidth="1.2"
          />
        ))}

        {/* Global minimum marker */}
        <circle cx={cx} cy={cy} r="5" fill="#ef4444" />
        <text x={cx + 8} y={cy + 4} fontSize="10" fill={dark ? '#fca5a5' : '#dc2626'} fontWeight="700">min</text>

        {/* Starting point */}
        <circle cx={40} cy={40} r="4" fill={dark ? '#94a3b8' : '#64748b'} />
        <text x={46} y={38} fontSize="10" fill={dark ? '#94a3b8' : '#64748b'}>start</text>

        {/* Trajectories */}
        {trajectories.map((t, ti) => {
          const d = t.pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join(' ')
          return (
            <g key={ti}>
              <path d={d} fill="none" stroke={t.color} strokeWidth="2" strokeLinejoin="round"
                strokeDasharray={ti === 1 ? '4 3' : 'none'} opacity="0.9" />
              {/* Arrowhead at last point */}
              <circle cx={t.pts[t.pts.length-1][0]} cy={t.pts[t.pts.length-1][1]} r="3" fill={t.color} />
            </g>
          )
        })}

        {/* Legend */}
        {trajectories.map((t, i) => (
          <g key={i}>
            <line x1={10} y1={180 + i * 18} x2={30} y2={180 + i * 18}
              stroke={t.color} strokeWidth="2.5"
              strokeDasharray={i === 1 ? '4 3' : 'none'} />
            <text x={35} y={184 + i * 18} fontSize="11" fill={dark ? '#cbd5e1' : '#475569'}>{t.name}</text>
          </g>
        ))}

        <text x="200" y="252" textAnchor="middle" fontSize="10" fill={dark ? '#475569' : '#94a3b8'}>
          Stylised contour — narrower = harder (ravine)
        </text>
      </svg>
    </div>
  )
}

// ── Diagram: gradient magnitude across layers ─────────────────────────────────

function GradientFlowDiagram({ theme }) {
  const dark = theme === 'dark'
  const layers = ['L₁','L₂','L₃','L₄','L₅','L₆','L₇','L₈']
  const vanish  = [1.0, 0.55, 0.28, 0.13, 0.06, 0.025, 0.01, 0.003]
  const explode = [1.0, 1.9,  3.5,  6.5,  12,   22,    40,   74]
  const healthy = [1.0, 0.95, 0.92, 0.88, 0.85, 0.83,  0.8,  0.78]

  const W = 360, H = 140, padL = 40, padB = 30, padT = 14

  function barY(val, maxVal) {
    const availH = H - padB - padT
    return padT + availH * (1 - Math.min(val / maxVal, 1))
  }
  function barH(val, maxVal) {
    const availH = H - padB - padT
    return availH * Math.min(val / maxVal, 1)
  }

  const maxVal = 80
  const barW   = 22
  // gap reserved for future padding adjustments
  const groupW = barW * 3 + 4

  return (
    <div className="my-4 overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H + 40}`} className="w-full max-w-md mx-auto block" style={{ minWidth: 300 }}>
        {layers.map((lbl, i) => {
          const gx = padL + i * (groupW + 6)
          return (
            <g key={i}>
              {/* Vanishing */}
              <rect x={gx} y={barY(vanish[i], maxVal)} width={barW} height={barH(vanish[i], maxVal)}
                fill="#6366f1" opacity="0.85" rx="2" />
              {/* Exploding */}
              <rect x={gx + barW + 2} y={barY(explode[i], maxVal)} width={barW} height={barH(explode[i], maxVal)}
                fill="#ef4444" opacity="0.75" rx="2" />
              {/* Healthy */}
              <rect x={gx + barW * 2 + 4} y={barY(healthy[i], maxVal)} width={barW} height={barH(healthy[i], maxVal)}
                fill="#10b981" opacity="0.85" rx="2" />
              {/* Layer label */}
              <text x={gx + barW * 1.5 + 2} y={H + 4} textAnchor="middle" fontSize="10"
                fill={dark ? '#94a3b8' : '#64748b'}>{lbl}</text>
            </g>
          )
        })}
        {/* Baseline */}
        <line x1={padL} y1={H - padB} x2={W - 10} y2={H - padB}
          stroke={dark ? '#334155' : '#e2e8f0'} strokeWidth="1" />

        {/* Legend */}
        {[['#6366f1','Vanishing'], ['#ef4444','Exploding'], ['#10b981','Healthy']].map(([c,l], i) => (
          <g key={l}>
            <rect x={10 + i * 110} y={H + 16} width="12" height="12" rx="2" fill={c} opacity="0.85" />
            <text x={26 + i * 110} y={H + 26} fontSize="10" fill={dark ? '#cbd5e1' : '#475569'}>{l}</text>
          </g>
        ))}
        <text x={W / 2} y={H + 54} textAnchor="middle" fontSize="10" fill={dark ? '#475569' : '#94a3b8'}>
          |∂L/∂W| at each layer (layer 1 = closest to output)
        </text>
      </svg>
    </div>
  )
}

// ── Diagram: Adam vs SGD bias correction ──────────────────────────────────────

function BiasCorrectionDiagram({ theme }) {
  const dark = theme === 'dark'
  const steps = 20
  const beta1 = 0.9, beta2 = 0.999
  // Simulate m_t and m_hat_t given a constant gradient g=1
  const g = 1.0
  let m = 0, v = 0
  const raw = [], corrected = []
  for (let t = 1; t <= steps; t++) {
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    raw.push(m)
    corrected.push(m / (1 - Math.pow(beta1, t)))
  }

  const W = 320, H = 110, padL = 36, padB = 24, padT = 10

  function toSVG(val, maxVal) {
    return padT + (H - padB - padT) * (1 - Math.min(val / maxVal, 1))
  }
  const maxVal = 1.05

  function polyline(arr, color) {
    const pts = arr.map((v, i) => {
      const x = padL + (i / (steps - 1)) * (W - padL - 10)
      const y = toSVG(v, maxVal)
      return `${x},${y}`
    }).join(' ')
    return <polyline key={color} points={pts} fill="none" stroke={color} strokeWidth="2" />
  }

  // True value line (g=1)
  const trueY = toSVG(1.0, maxVal)

  return (
    <div className="my-4 overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H + 36}`} className="w-full max-w-sm mx-auto block" style={{ minWidth: 260 }}>
        {/* True gradient reference */}
        <line x1={padL} y1={trueY} x2={W - 10} y2={trueY}
          stroke={dark ? '#475569' : '#cbd5e1'} strokeWidth="1" strokeDasharray="6 3" />
        <text x={W - 8} y={trueY - 3} textAnchor="end" fontSize="9"
          fill={dark ? '#64748b' : '#94a3b8'}>true g=1</text>

        {/* Axis */}
        <line x1={padL} y1={padT} x2={padL} y2={H - padB}
          stroke={dark ? '#334155' : '#e2e8f0'} strokeWidth="1" />
        <line x1={padL} y1={H - padB} x2={W - 10} y2={H - padB}
          stroke={dark ? '#334155' : '#e2e8f0'} strokeWidth="1" />

        {polyline(raw,       '#f59e0b')}
        {polyline(corrected, '#6366f1')}

        {/* X labels */}
        {[1,5,10,15,20].map(t => {
          const x = padL + ((t-1) / (steps - 1)) * (W - padL - 10)
          return <text key={t} x={x} y={H - padB + 12} textAnchor="middle" fontSize="9"
            fill={dark ? '#64748b' : '#94a3b8'}>{t}</text>
        })}
        <text x={W / 2} y={H - padB + 22} textAnchor="middle" fontSize="9"
          fill={dark ? '#64748b' : '#94a3b8'}>step t</text>

        {/* Legend */}
        <rect x={padL} y={H + 6} width="10" height="10" rx="2" fill="#f59e0b" />
        <text x={padL + 14} y={H + 15} fontSize="10" fill={dark ? '#cbd5e1' : '#475569'}>m_t (biased)</text>
        <rect x={padL + 100} y={H + 6} width="10" height="10" rx="2" fill="#6366f1" />
        <text x={padL + 114} y={H + 15} fontSize="10" fill={dark ? '#cbd5e1' : '#475569'}>m̂_t (corrected)</text>
      </svg>
      <p className={`text-center text-xs mt-1 ${dark ? 'text-slate-500' : 'text-slate-400'}`}>
        At t=1 the biased estimate is 10× too small — bias correction fixes this
      </p>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function OptimizationDeepModels() {
  const { theme, markTopicComplete } = useApp()
  const dark = theme === 'dark'

  return (
    <div className="space-y-10">
      {/* ── Header ── */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3 mb-3">
          <span className="text-4xl">⚙️</span>
          <div>
            <p className={`text-sm font-semibold uppercase tracking-widest ${dark ? 'text-violet-400' : 'text-violet-600'}`}>
              Session 14 · Deep Learning
            </p>
            <h1 className={`text-3xl font-black ${dark ? 'text-white' : 'text-slate-900'}`}>
              Optimization of Deep Models
            </h1>
          </div>
        </div>
        <p className={`text-lg leading-relaxed ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Training a deep network is a high-dimensional non-convex optimization problem.
          This session covers the <strong>challenges</strong> that make it hard and the family of
          <strong> adaptive optimizers</strong> — from Momentum through Adam — that make it tractable.
        </p>

        {/* Prerequisites */}
        <div className={`mt-4 rounded-xl p-4 border text-sm ${dark ? 'bg-amber-900/20 border-amber-700/40 text-amber-200' : 'bg-amber-50 border-amber-200 text-amber-800'}`}>
          <p className="font-bold mb-1">📚 Prerequisites</p>
          <ul className="list-disc list-inside space-y-0.5 text-xs">
            <li><strong>Backpropagation &amp; chain rule</strong> — how gradients flow backward through a computation graph</li>
            <li><strong>Gradient descent fundamentals</strong> — weight update rule, learning rate, loss landscape</li>
            <li><strong>Vanishing gradients in RNNs</strong> — covered in Session 9/10</li>
          </ul>
        </div>
      </motion.div>

      {/* ══════════════════════════════════════════
          Section 14.1 — Optimization Goals & Challenges
         ══════════════════════════════════════════ */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 14.1</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Optimization Goals and Challenges</h2>
        </div>

        {/* 14.1.1 Goal */}
        <h3 className={`text-lg font-bold mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          14.1.1 — Goal of Optimization in Deep Learning
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The objective is to find a parameter vector <strong>θ</strong> that minimises the
          expected loss on the data distribution. In practice we minimise the empirical loss on
          the training set and rely on regularisation &amp; generalisation theory to ensure this
          also minimises test loss.
        </p>

        <Callout type="formula" title="Empirical Risk Minimization (ERM)">
          θ* = argmin_θ  (1/N) Σᵢ L(f(xᵢ; θ), yᵢ) + λ · Ω(θ)
        </Callout>

        <TheoryBlock items={[
          {
            title: 'Loss function L',
            content: 'Measures prediction error per sample. Cross-entropy for classification, MSE for regression. Gradients ∇L drive parameter updates.',
          },
          {
            title: 'Regularizer Ω(θ)',
            content: 'Penalty on parameter magnitude (L2 = weight decay, L1 = sparsity). Controlled by λ. Keeps model from overfitting the empirical loss.',
          },
          {
            title: 'Optimisation vs. learning',
            content: 'Optimisation minimises training loss. Learning requires that training-loss minima generalise. These are related but distinct goals.',
          },
          {
            title: 'Why it is hard',
            content: 'θ lives in millions-to-billions of dimensions. The loss landscape is non-convex with local minima, saddle points, flat regions, and sharp cliffs.',
          },
        ]} />

        {/* 14.1.2 Non-convex surfaces */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          14.1.2 — Non-Convex Loss Surfaces: Local Minima &amp; Saddle Points
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Deep network loss surfaces are <strong>non-convex</strong> — no single global minimum is
          guaranteed to be reachable. The two most important non-convex features are
          <strong> local minima</strong> (all directions curve up, but loss is not globally minimal)
          and <strong>saddle points</strong> (gradient is zero but the point is neither a min nor max).
        </p>

        <CodeBlock code={CODE_LOSS_SURFACE} language="python" title="loss_surface.py" />

        <TheoryBlock cols={2} items={[
          {
            title: '📉 Local minima',
            content: 'All eigenvalues of the Hessian are positive. In practice, local minima in large networks tend to have similar loss to the global minimum — the loss landscape is relatively "flat" at the bottom.',
          },
          {
            title: '🔀 Saddle points',
            content: 'Some Hessian eigenvalues are positive, some negative. ∇L=0 but not a minimum. In D dimensions, the probability a random critical point is a local minimum is ~2^{-D}. Saddle points dominate.',
          },
          {
            title: '🏔 Sharp vs. flat minima',
            content: 'Sharp minima generalise poorly (tiny param change → large loss increase). Flat minima generalise better. SGD noise biases toward flatter basins — a useful implicit regulariser.',
          },
          {
            title: '🏜 Plateaus',
            content: 'Regions where the gradient is near zero everywhere but the loss is not at a minimum. Can arise from symmetry (permutation of neurons) or from saturating activations.',
          },
        ]} />

        <DeepDive title="Why local minima are less feared than saddle points">
          <TheoryBlock items={[
            {
              title: '',
              content: 'Dauphin et al. (2014) showed empirically and theoretically that in high-dimensional loss surfaces, the ratio of negative eigenvalues of the Hessian at a critical point scales with the loss value. Minima of high loss (bad solutions) are exponentially rare. The dangerous critical points are saddle points — gradient is zero, so first-order methods stall. SGD noise and adaptive optimizers both help escape.',
            },
          ]} />
        </DeepDive>

        {/* 14.1.3 Vanishing / exploding gradients */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          14.1.3 — Vanishing and Exploding Gradients
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          During backpropagation, gradients are multiplied by the weight matrix (and activation
          derivative) at every layer. If the <strong>spectral radius</strong> (largest eigenvalue
          magnitude) of those matrices is &lt; 1, gradients shrink exponentially with depth
          (vanishing). If &gt; 1, they grow exponentially (exploding).
        </p>

        <GradientFlowDiagram theme={theme} />

        <Callout type="formula" title="Gradient at layer l via chain rule">
          ∂L/∂W_l = (∂L/∂h_L) · (∏ ∂h_k/∂h_(k-1)) · ∂h_l/∂W_l
        </Callout>

        <CodeBlock code={CODE_VANISHING_EXPLODING} language="python" title="vanishing_exploding.py" />

        <TheoryBlock items={[
          {
            title: '🛠 Fixes for vanishing gradients',
            content: 'ReLU activations (derivative = 1 for positive inputs), careful initialisation (Xavier/He), Batch Normalisation, residual connections (gradient highway), LSTM/GRU gates.',
          },
          {
            title: '🛠 Fixes for exploding gradients',
            content: 'Gradient clipping — cap ‖g‖ ≤ threshold before the update. Weight initialisation with small σ. Spectral normalisation (constrain ‖W‖₂ ≤ 1). Batch Norm.',
          },
          {
            title: 'Gradient clipping rule',
            content: 'If ‖g‖ > clip_val: g ← g · clip_val / ‖g‖. This rescales the gradient direction without changing it. Typical clip_val = 1.0 for transformers, 5.0 for RNNs.',
          },
        ]} />

        {/* 14.1.4 Batch GD, SGD, mini-batch */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          14.1.4 — Review: Batch GD, SGD, and Mini-Batch SGD
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          All gradient-based optimizers differ in <em>how much data</em> they use to estimate the
          gradient at each step. The trade-off is between gradient accuracy and update frequency.
        </p>

        <TheoryBlock cols={3} items={[
          {
            title: '📦 Batch (Full) GD',
            content: 'Uses all N samples. Exact gradient, stable convergence. One update per epoch. Impractical for large N. Cannot escape sharp minima.',
          },
          {
            title: '🎲 Stochastic GD (B=1)',
            content: 'One sample per update. Very noisy gradient but N updates per epoch. Noise can escape saddle points and sharp minima. Unstable without LR decay.',
          },
          {
            title: '⚖️ Mini-Batch GD',
            content: 'Batch size B ∈ [32, 512]. Best of both worlds: vectorised computation (GPU-efficient), N/B updates per epoch, moderate noise. Standard in practice.',
          },
        ]} />

        <Callout type="info" title="Rule of thumb for batch size">
          Larger batches → better gradient estimates but may converge to sharp minima (worse generalisation). 
          Hoffer et al. (2017) and Keskar et al. (2017): B=256–512 is a common sweet spot. 
          When scaling batch size by k×, scale learning rate by √k (linear scaling also common).
        </Callout>

        <CodeBlock code={CODE_SGD_VARIANTS} language="python" title="sgd_variants.py" />
      </motion.div>

      {/* ══════════════════════════════════════════
          Section 14.2 — Adaptive Optimizers
         ══════════════════════════════════════════ */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-violet-500/5 border-violet-500/20' : 'bg-violet-50 border-violet-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-violet-400' : 'text-violet-600'}`}>Section 14.2</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Adaptive Optimizers</h2>
        </div>

        <p className={`mb-6 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Plain SGD uses a single global learning rate η for all parameters. Adaptive optimizers
          maintain <strong>per-parameter running statistics</strong> — allowing frequently-updated
          parameters to take smaller steps and rarely-updated parameters to take larger steps.
          This is especially powerful in NLP where input features are sparse.
        </p>

        <OptimizerTrajectoryDiagram theme={theme} />

        {/* 14.2.1 Momentum */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          14.2.1 — Momentum: Exponential Moving Average of Gradients
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Momentum introduces a <strong>velocity</strong> term that accumulates an exponential moving
          average of past gradients. Consistent gradient directions build up velocity (acceleration);
          oscillating directions cancel out (damping). The result: faster convergence on elongated
          loss surfaces, reduced zig-zagging.
        </p>

        <Callout type="formula" title="SGD + Momentum">
          v_t = β · v(t−1) + (1-β) · g_t &nbsp;&nbsp;&nbsp; θ_t = θ(t−1) − η · v_t
        </Callout>

        <TheoryBlock cols={2} items={[
          {
            title: 'β = 0.9 (typical)',
            content: 'Half-life ≈ 6.6 steps — roughly the last 7 gradients contribute significantly. β=0.99 ≈ 69-step memory. Larger β → smoother trajectory but slower reaction to curvature changes.',
          },
          {
            title: 'Nesterov Momentum (NAG)',
            content: 'Evaluates gradient at the "anticipated" position θ + β·v instead of the current θ. Provides look-ahead correction; slightly faster convergence in theory and practice.',
          },
          {
            title: 'Classical vs. EMA formulation',
            content: 'Classical: v_t = β·v(t−1) + g_t (no 1-β). EMA: v_t = β·v(t−1) + (1-β)·g_t. Adam uses the EMA form so the scale of v matches the gradient scale.',
          },
          {
            title: 'Intuition: ball on slope',
            content: 'Momentum acts like a ball rolling down a hill — it accelerates on long flat stretches and overshoots slightly, but damps oscillations in narrow valleys.',
          },
        ]} />

        <CodeBlock code={CODE_MOMENTUM} language="python" title="sgd_momentum.py" />

        {/* 14.2.2 Adagrad */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          14.2.2 — Adagrad: Per-Parameter Adaptive Learning Rate
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Adagrad (Duchi et al., 2011) maintains a <strong>cumulative sum of squared gradients</strong>
          for each parameter. Parameters that receive large or frequent gradients automatically get a
          smaller effective learning rate; rare/small-gradient parameters get a larger one. Ideal for
          sparse features (e.g. word embeddings).
        </p>

        <Callout type="formula" title="Adagrad update">
          G_t = G(t−1) + g_t²  &nbsp;&nbsp;&nbsp;  θ_t = θ(t−1) − (η / √(G_t + ε)) · g_t
        </Callout>

        <TheoryBlock items={[
          {
            title: '✅ Advantage: automatic per-parameter LR',
            content: 'Eliminates manual LR tuning for different feature groups. Word embeddings for rare words get larger updates; common words get smaller ones.',
          },
          {
            title: '⚠️ Disadvantage: monotonically shrinking LR',
            content: 'G_t is monotonically non-decreasing. Eventually η/√G_t → 0 for ALL parameters, even if the gradient is still informative. Training stalls for long-running jobs.',
          },
          {
            title: 'ε (epsilon)',
            content: 'Small constant (1e-8) added for numerical stability to prevent division by zero when G_t = 0 for a never-updated parameter.',
          },
        ]} />

        <CodeBlock code={CODE_ADAGRAD} language="python" title="adagrad.py" />

        {/* 14.2.3 RMSProp */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          14.2.3 — RMSProp: Leaky Adagrad with Exponential Decay
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          RMSProp (Hinton, 2012) fixes Adagrad's dying learning rate by replacing the cumulative sum
          with an <strong>exponential moving average</strong> of squared gradients. Old gradients are
          exponentially forgotten, so the denominator stays bounded and the learning rate remains
          adaptive throughout training.
        </p>

        <Callout type="formula" title="RMSProp update">
          E[g²]_t = ρ · E[g²](t−1) + (1-ρ) · g_t² &nbsp;&nbsp;&nbsp; θ_t = θ(t−1) − (η / √(E[g²]_t + ε)) · g_t
        </Callout>

        <TheoryBlock cols={2} items={[
          {
            title: 'ρ = 0.9 (decay rate)',
            content: 'Effective window size ≈ 1/(1-ρ) = 10 steps. The moving average remembers roughly the last 10 gradient magnitudes. Larger ρ → smoother but slower to adapt.',
          },
          {
            title: 'Non-stationary objectives',
            content: 'The forgetting mechanism makes RMSProp well-suited for RNNs and RL where the effective loss changes as the policy or sequence context changes.',
          },
          {
            title: 'No bias correction',
            content: 'Unlike Adam, RMSProp does not correct the initialisation bias (E[g²]_0=0). In practice this is minor because ρ=0.9 warms up quickly.',
          },
          {
            title: 'Hinton unpublished',
            content: 'RMSProp was introduced in Hinton\'s Coursera lecture slides (2012) — no formal paper. It was widely adopted and became the precursor to Adam.',
          },
        ]} />

        <CodeBlock code={CODE_RMSPROP} language="python" title="rmsprop.py" />

        {/* 14.2.4 Adadelta */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          14.2.4 — Adadelta: Second Moment Variant
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Adadelta (Zeiler, 2012) extends RMSProp by also tracking an EMA of the
          <strong> squared parameter updates</strong> (Δθ), not just squared gradients. This gives
          the update the correct units (matching the units of θ) without needing to set a global
          learning rate η at all.
        </p>

        <Callout type="formula" title="Adadelta update">
          E[g²]_t = ρ·E[g²](t−1) + (1-ρ)·g_t² &nbsp;&nbsp;&nbsp;
          Δθ_t = −(√(E[Δθ²](t−1) + ε) / √(E[g²]_t + ε)) · g_t &nbsp;&nbsp;&nbsp;
          E[Δθ²]_t = ρ·E[Δθ²](t−1) + (1-ρ)·Δθ_t²
        </Callout>

        <TheoryBlock items={[
          {
            title: 'No learning rate hyperparameter',
            content: 'The ratio √E[Δθ²]/√E[g²] acts as an adaptive learning rate derived entirely from the curvature of the loss — in theory no η needed. In practice a small η is often reintroduced.',
          },
          {
            title: 'Units match',
            content: 'Gradients have units of loss/θ. Dividing by √E[g²] gives units of θ (approximately). The Δθ EMA in the numerator makes the units exact. Adadelta was motivated by this dimensional analysis.',
          },
          {
            title: 'When to use',
            content: 'Adadelta is rarely the first choice today (Adam is simpler and usually better). It can be useful when you want to avoid tuning a learning rate entirely.',
          },
        ]} />

        {/* 14.2.5 Adam */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          14.2.5 — Adam: Combining Momentum and RMSProp with Bias Correction
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Adam (Kingma &amp; Ba, 2015) is the de-facto standard optimizer for deep learning. It
          combines <strong>first-moment estimation</strong> (Momentum) with <strong>second-moment
          estimation</strong> (RMSProp) and applies <strong>bias correction</strong> to both,
          compensating for the zero initialisation of the running averages.
        </p>

        <Callout type="formula" title="Adam update rule">
          m_t = β₁·m(t−1) + (1-β₁)·g_t &nbsp;&nbsp; v_t = β₂·v(t−1) + (1-β₂)·g_t² &nbsp;&nbsp;
          m̂_t = m_t/(1-β₁ᵗ) &nbsp;&nbsp; v̂_t = v_t/(1-β₂ᵗ) &nbsp;&nbsp;
          θ_t = θ(t−1) − η · m̂_t / (√v̂_t + ε)
        </Callout>

        <BiasCorrectionDiagram theme={theme} />

        <TheoryBlock cols={2} items={[
          {
            title: 'Default hyperparameters',
            content: 'η=1e-3, β₁=0.9, β₂=0.999, ε=1e-8. These defaults work well on most tasks without tuning — a key reason for Adam\'s widespread adoption.',
          },
          {
            title: 'Bias correction intuition',
            content: 'At t=1: m₁=(1-β₁)·g₁≈0.1g₁. Dividing by (1-0.9¹)=0.1 recovers g₁. Without correction, early updates are systematically too small and training is slow to start.',
          },
          {
            title: 'Why β₂=0.999?',
            content: 'Very slow second-moment decay. The denominator √v̂ tracks long-term gradient variance — a stable baseline. β₂=0.999 gives a 1000-step effective window.',
          },
          {
            title: 'Adam vs. SGD+Momentum',
            content: 'Adam converges faster and requires less LR tuning. SGD+Momentum with a well-tuned LR schedule can achieve slightly better generalisation (Wilson et al., 2017). Large-scale LLM training typically uses Adam.',
          },
        ]} />

        <DeepDive title="AdamW — Adam with decoupled weight decay">
          <TheoryBlock items={[
            {
              title: '',
              content: 'Standard Adam applies L2 weight decay by adding λ·θ to the gradient before computing m and v. This conflates the adaptive scaling with the regulariser — large-gradient parameters are under-regularised. AdamW (Loshchilov & Hutter, 2019) decouples them: update = Adam_update − η·λ·θ. This is the default in PyTorch (AdamW) and is used in BERT, GPT-2, and most modern LLMs.',
            },
          ]} />
        </DeepDive>

        <CodeBlock code={CODE_ADAM} language="python" title="adam.py" />

        {/* Optimizer comparison table */}
        <h3 className={`text-lg font-bold mt-8 mb-4 ${dark ? 'text-white' : 'text-slate-800'}`}>
          Optimizer Comparison Summary
        </h3>
        <div className="overflow-x-auto mb-6">
          <table className={`w-full text-sm border-collapse rounded-xl overflow-hidden ${dark ? 'text-slate-300' : 'text-slate-700'}`}>
            <thead>
              <tr className={dark ? 'bg-slate-800' : 'bg-slate-100'}>
                {['Optimizer','1st Moment','2nd Moment','Bias Correction','η Required','Key Strength'].map(h => (
                  <th key={h} className={`px-3 py-2.5 text-left text-xs font-bold uppercase tracking-wide ${dark ? 'text-slate-400' : 'text-slate-500'}`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['SGD',         '—',        '—',        '—',   '✅','Simple baseline'],
                ['Momentum',    'EMA of g', '—',        '—',   '✅','Escapes ravines'],
                ['Adagrad',     '—',        'Σg²',      '—',   '✅','Sparse features'],
                ['RMSProp',     '—',        'EMA of g²','—',   '✅','Non-stationary'],
                ['Adadelta',    '—',        'EMA of g²','—',   '❌','No LR needed'],
                ['Adam',        'EMA of g', 'EMA of g²','✅',  '✅','General purpose'],
                ['AdamW',       'EMA of g', 'EMA of g²','✅',  '✅','Adam + correct WD'],
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
      </motion.div>

      {/* ── Quiz ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <Quiz
          questions={QUIZ_QUESTIONS}
          onComplete={() => markTopicComplete('optimization-deep-models')}
        />
      </motion.div>
    </div>
  )
}
