import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

const PYTHON_CODE_CLASSIFY = `from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Always scale features for kNN!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Classification — k=5, Euclidean distance (p=2)
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)

print(f"Accuracy: {knn.score(X_test, y_test):.2%}")
print(classification_report(y_test, knn.predict(X_test),
      target_names=['setosa','versicolor','virginica']))

# Find neighbours of a new point
sample = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
distances, indices = knn.kneighbors(sample)
print("5 nearest neighbours (indices):", indices[0])
print("Distances:", distances[0].round(3))`

const PYTHON_CODE_REGRESSION = `from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Toy dataset: x → y = sin(x) + noise
np.random.seed(42)
X = np.sort(np.random.uniform(0, 6, 80)).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.normal(0, 0.2, 80)

# kNN regression — predicts mean of k nearest neighbours' targets
knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
scores = cross_val_score(knn_reg, X, y, cv=5, scoring='r2')
print(f"R² (CV): {scores.mean():.4f} ± {scores.std():.4f}")

# Compare k values
for k in [1, 3, 5, 10, 20]:
    r2 = cross_val_score(KNeighborsRegressor(k), X, y, cv=5).mean()
    print(f"  k={k:2d}  R²={r2:.4f}")`

const PYTHON_CODE_CURSE = `from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

results = []
for n_features in [2, 5, 10, 20, 50, 100, 200]:
    X, y = make_classification(
        n_samples=500, n_features=n_features,
        n_informative=min(n_features, 5),  # only 5 truly useful
        n_redundant=0, random_state=42
    )
    X = StandardScaler().fit_transform(X)
    acc = cross_val_score(KNeighborsClassifier(n_neighbors=5),
                          X, y, cv=5).mean()
    results.append((n_features, acc))
    print(f"  dims={n_features:3d}  accuracy={acc:.2%}")

# As dimensions grow, accuracy degrades because
# all points become equidistant — k neighbours lose meaning`

const QUIZ_QUESTIONS = [
  { question: 'kNN is a __ algorithm — it stores all training data.', options: ['Eager', 'Lazy', 'Parametric', 'Generative'], correct: 1, explanation: 'kNN is lazy — it defers all computation to prediction time. There is no explicit training phase; the model is the training data itself.' },
  { question: 'In kNN classification, the predicted class is:', options: ['The class of the single nearest neighbour', 'The mean of k neighbours\' classes', 'The majority class among k neighbours', 'The median class among k neighbours'], correct: 2, explanation: 'Each of the k nearest neighbours votes, and the majority class wins. Ties are broken by distance or randomly.' },
  { question: 'What does Minkowski distance with p=1 equal?', options: ['Euclidean', 'Manhattan', 'Chebyshev', 'Cosine'], correct: 1, explanation: 'Minkowski(p=1) = Σ|xᵢ−yᵢ| = Manhattan (city-block) distance. p=2 gives Euclidean. p→∞ gives Chebyshev.' },
  { question: 'Why must features be scaled before kNN?', options: ['To speed up computation', 'kNN uses distances — unscaled features dominate by magnitude', 'kNN requires normalised probabilities', 'Scaling reduces overfitting'], correct: 1, explanation: 'A feature with range [0,1000] will dominate over one with range [0,1] in Euclidean distance. StandardScaler or MinMaxScaler makes distances meaningful.' },
  { question: 'Small k (e.g. k=1) leads to:', options: ['High bias, low variance', 'Low bias, high variance', 'Underfitting', 'Smooth decision boundary'], correct: 1, explanation: 'k=1 memorises training data — decision boundary is jagged and overfits noise (high variance). Large k smooths the boundary (higher bias, lower variance).' },
  { question: 'What is the time complexity of kNN prediction for n training samples and d features?', options: ['O(1)', 'O(log n)', 'O(n·d)', 'O(d²)'], correct: 2, explanation: 'Brute-force kNN computes distance to every training point — O(n·d) per query. KD-trees improve this to O(d·log n) for low dimensions.' },
  { question: 'The curse of dimensionality primarily causes:', options: ['Slower training', 'All points to become equidistant in high dimensions', 'Larger decision boundaries', 'Fewer neighbours to find'], correct: 1, explanation: 'In high dimensions, the ratio of max to min distance approaches 1 — all points are nearly equidistant. The k-nearest concept loses meaning.' },
  { question: 'kNN regression predicts:', options: ['The class of the nearest neighbour', 'The mean (or weighted mean) of k neighbours\' target values', 'The mode of k neighbours', 'A linear combination of all training targets'], correct: 1, explanation: 'For regression, kNN averages the target values of the k nearest neighbours. Distance-weighted averaging gives closer neighbours more influence.' },
  { question: 'Distance-weighted kNN gives:', options: ['All k neighbours equal weight', 'Closer neighbours more weight (1/distance)', 'Farther neighbours more weight', 'Only the nearest neighbour weight'], correct: 1, explanation: 'Weighted kNN uses w=1/distance, so the closest neighbour has the most influence. This often outperforms uniform weighting.' },
  { question: 'Which data structure speeds up kNN search in low dimensions?', options: ['Hash table', 'KD-tree', 'Priority queue', 'Heap'], correct: 1, explanation: 'A KD-tree partitions the feature space into a binary tree. For d < ~20, neighbour search is O(d·log n) instead of O(n·d). Ball trees work better for higher d.' },
]

// ── Interactive kNN classifier ─────────────────────────────────────────────────
function KNNCanvas({ theme }) {
  const canvasRef = useRef(null)
  const [k, setK] = useState(3)
  const [metric, setMetric] = useState('euclidean')
  const [points] = useState([
    { x: 80,  y: 80,  cls: 0 }, { x: 110, y: 60,  cls: 0 }, { x: 60,  y: 110, cls: 0 },
    { x: 130, y: 90,  cls: 0 }, { x: 90,  y: 140, cls: 0 },
    { x: 260, y: 200, cls: 1 }, { x: 290, y: 170, cls: 1 }, { x: 240, y: 230, cls: 1 },
    { x: 310, y: 210, cls: 1 }, { x: 270, y: 240, cls: 1 },
    { x: 170, y: 160, cls: 0 }, { x: 200, y: 130, cls: 1 },
  ])
  const [query, setQuery] = useState({ x: 190, y: 170 })
  const [dragging, setDragging] = useState(false)
  const W = 380, H = 280

  const dist = (a, b) => {
    if (metric === 'euclidean') return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y)  // manhattan
  }

  const sorted = [...points].sort((a, b) => dist(a, query) - dist(b, query))
  const neighbours = sorted.slice(0, k)
  const votes = [0, 1].map(c => neighbours.filter(n => n.cls === c).length)
  const predicted = votes[0] > votes[1] ? 0 : votes[1] > votes[0] ? 1 : -1
  const COLORS = ['#6366F1', '#10B981']
  const LABELS = ['Class A', 'Class B']

  const getXY = (e) => {
    const rect = canvasRef.current.getBoundingClientRect()
    const scaleX = W / rect.width
    const scaleY = H / rect.height
    return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY }
  }

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)

    // Decision boundary background (flood fill approximation via grid)
    const step = 6
    for (let gx = 0; gx < W; gx += step) {
      for (let gy = 0; gy < H; gy += step) {
        const pt = { x: gx, y: gy }
        const near = [...points].sort((a, b) => dist(a, pt) - dist(b, pt)).slice(0, k)
        const v = [0, 1].map(c => near.filter(n => n.cls === c).length)
        const pred = v[0] > v[1] ? 0 : 1
        ctx.fillStyle = pred === 0 ? 'rgba(99,102,241,0.07)' : 'rgba(16,185,129,0.07)'
        ctx.fillRect(gx, gy, step, step)
      }
    }

    // Radius circle from query to k-th neighbour
    const kDist = dist(neighbours[neighbours.length - 1] || query, query)
    ctx.beginPath()
    ctx.arc(query.x, query.y, kDist, 0, Math.PI * 2)
    ctx.strokeStyle = 'rgba(251,191,36,0.4)'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([])

    // Lines to neighbours
    neighbours.forEach(n => {
      ctx.beginPath(); ctx.moveTo(query.x, query.y); ctx.lineTo(n.x, n.y)
      ctx.strokeStyle = 'rgba(251,191,36,0.5)'; ctx.lineWidth = 1.5; ctx.stroke()
    })

    // Training points
    points.forEach(p => {
      const isNeighbour = neighbours.includes(p)
      ctx.beginPath(); ctx.arc(p.x, p.y, isNeighbour ? 9 : 7, 0, Math.PI * 2)
      ctx.fillStyle = COLORS[p.cls]; ctx.fill()
      ctx.strokeStyle = isNeighbour ? '#FBF24A' : 'white'
      ctx.lineWidth = isNeighbour ? 2.5 : 1.5; ctx.stroke()
    })

    // Query point
    ctx.beginPath(); ctx.arc(query.x, query.y, 10, 0, Math.PI * 2)
    ctx.fillStyle = predicted === 0 ? 'rgba(99,102,241,0.9)' : predicted === 1 ? 'rgba(16,185,129,0.9)' : '#6B7280'
    ctx.fill(); ctx.strokeStyle = '#FBF24A'; ctx.lineWidth = 3; ctx.stroke()
    ctx.fillStyle = 'white'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center'
    ctx.fillText('?', query.x, query.y + 4)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, query, k, metric, theme])

  return (
    <div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>k = <span className="text-yellow-400 font-bold">{k}</span></label>
          <input type="range" min="1" max="11" step="2" value={k} onChange={e => setK(+e.target.value)} className="w-full accent-yellow-500" />
          <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>{k <= 3 ? 'Low k — sensitive' : k >= 9 ? 'High k — smooth' : 'Balanced'}</p>
        </div>
        <div className="col-span-2 flex gap-2 items-end pb-1">
          {[['euclidean', 'Euclidean'], ['manhattan', 'Manhattan']].map(([v, l]) => (
            <button key={v} onClick={() => setMetric(v)}
              className={`flex-1 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                metric === v
                  ? theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-indigo-100 border-indigo-300 text-indigo-700'
                  : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'
              }`}>{l}</button>
          ))}
        </div>
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border cursor-crosshair touch-none ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }}
        onMouseDown={e => { setDragging(true); setQuery(getXY(e)) }}
        onMouseMove={e => { if (dragging) setQuery(getXY(e)) }}
        onMouseUp={() => setDragging(false)}
        onMouseLeave={() => setDragging(false)}
        onTouchStart={e => { setDragging(true); setQuery(getXY(e.touches[0])) }}
        onTouchMove={e => { e.preventDefault(); setQuery(getXY(e.touches[0])) }}
        onTouchEnd={() => setDragging(false)}
      />
      <div className="flex gap-3 mt-3 text-xs flex-wrap">
        <div className={`flex-1 px-3 py-2 rounded-lg ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-100'}`}>
          <span className={theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}>Votes: </span>
          <span className="text-indigo-400 font-bold">{votes[0]}A</span>
          <span className={`mx-1 ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>vs</span>
          <span className="text-emerald-400 font-bold">{votes[1]}B</span>
        </div>
        <div className={`flex-1 px-3 py-2 rounded-lg ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-100'}`}>
          <span className={theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}>Prediction: </span>
          <span className={`font-bold ${predicted === 0 ? 'text-indigo-400' : predicted === 1 ? 'text-emerald-400' : 'text-gray-400'}`}>
            {predicted === -1 ? 'Tie' : LABELS[predicted]}
          </span>
        </div>
        <div className={`flex-1 px-3 py-2 rounded-lg ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-100'} ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
          Drag the <span className="text-yellow-400 font-bold">?</span> point to classify it
        </div>
      </div>
    </div>
  )
}

// ── Distance metrics visualiser ────────────────────────────────────────────────
function DistanceViz({ theme }) {
  const [p, setP] = useState(2)
  const [ax, setAx] = useState(1)
  const [ay, setAy] = useState(2)
  const [bx, setBx] = useState(4)
  const [by, setBy] = useState(6)

  const dx = Math.abs(bx - ax), dy = Math.abs(by - ay)
  const euclidean = Math.sqrt(dx ** 2 + dy ** 2)
  const manhattan = dx + dy
  const minkowski = (dx ** p + dy ** p) ** (1 / p)
  const chebyshev = Math.max(dx, dy)

  const canvasRef = useRef(null)
  const W = 380, H = 260

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)
    const pad = 30
    const scale = (W - 2 * pad) / 9
    const toX = v => pad + v * scale
    const toY = v => H - pad - v * scale

    // Grid
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'
    ctx.lineWidth = 1; ctx.setLineDash([2, 3])
    for (let i = 0; i <= 9; i++) {
      ctx.beginPath(); ctx.moveTo(toX(i), pad); ctx.lineTo(toX(i), H - pad); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(pad, toY(i)); ctx.lineTo(W - pad, toY(i)); ctx.stroke()
    }
    ctx.setLineDash([])

    // Axes
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); ctx.stroke()

    // Manhattan path
    ctx.beginPath(); ctx.moveTo(toX(ax), toY(ay)); ctx.lineTo(toX(bx), toY(ay)); ctx.lineTo(toX(bx), toY(by))
    ctx.strokeStyle = '#F59E0B'; ctx.lineWidth = 2; ctx.setLineDash([5, 3]); ctx.stroke(); ctx.setLineDash([])

    // Euclidean line
    ctx.beginPath(); ctx.moveTo(toX(ax), toY(ay)); ctx.lineTo(toX(bx), toY(by))
    ctx.strokeStyle = '#6366F1'; ctx.lineWidth = 2.5; ctx.stroke()

    // Minkowski "unit ball" contour around A (just illustrative if p ≠ 2)
    if (Math.abs(p - 2) > 0.1 && Math.abs(p - 1) > 0.1) {
      const r = minkowski
      ctx.beginPath()
      for (let angle = 0; angle <= Math.PI * 2; angle += 0.05) {
        // Approximate Lp ball: |x|^p + |y|^p = r^p
        // Use polar parametric: x = r·|cos|^(2/p)·sign(cos), same for y
        const cosA = Math.cos(angle), sinA = Math.sin(angle)
        const lx = Math.sign(cosA) * (Math.abs(cosA) ** (2 / p)) * r
        const ly = Math.sign(sinA) * (Math.abs(sinA) ** (2 / p)) * r
        const cx = toX(ax + lx), cy = toY(ay + ly)
        angle === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy)
      }
      ctx.strokeStyle = 'rgba(6,182,212,0.4)'; ctx.lineWidth = 1.5; ctx.stroke()
    }

    // Points
    [[ax, ay, '#6366F1', 'A'], [bx, by, '#10B981', 'B']].forEach(([x, y, color, label]) => {
      ctx.beginPath(); ctx.arc(toX(x), toY(y), 7, 0, Math.PI * 2)
      ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.stroke()
      ctx.fillStyle = theme === 'dark' ? '#E2E8F0' : '#1E293B'
      ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center'
      ctx.fillText(label, toX(x), toY(y) - 12)
    })

    // Labels
    ctx.fillStyle = theme === 'dark' ? '#6B7280' : '#9CA3AF'
    ctx.font = '10px monospace'; ctx.textAlign = 'center'
    for (let i = 0; i <= 9; i += 2) {
      ctx.fillText(i, toX(i), H - pad + 14)
      if (i > 0) ctx.fillText(i, pad - 14, toY(i) + 3)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ax, ay, bx, by, p, theme])

  return (
    <div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <p className={`text-xs font-semibold mb-2 text-indigo-400`}>Point A</p>
          <div className="flex gap-2">
            <div className="flex-1">
              <label className={`text-xs block mb-1 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>x={ax}</label>
              <input type="range" min="0" max="8" step="0.5" value={ax} onChange={e => setAx(+e.target.value)} className="w-full accent-indigo-500" />
            </div>
            <div className="flex-1">
              <label className={`text-xs block mb-1 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>y={ay}</label>
              <input type="range" min="0" max="8" step="0.5" value={ay} onChange={e => setAy(+e.target.value)} className="w-full accent-indigo-500" />
            </div>
          </div>
        </div>
        <div>
          <p className={`text-xs font-semibold mb-2 text-emerald-400`}>Point B</p>
          <div className="flex gap-2">
            <div className="flex-1">
              <label className={`text-xs block mb-1 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>x={bx}</label>
              <input type="range" min="0" max="8" step="0.5" value={bx} onChange={e => setBx(+e.target.value)} className="w-full accent-emerald-500" />
            </div>
            <div className="flex-1">
              <label className={`text-xs block mb-1 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>y={by}</label>
              <input type="range" min="0" max="8" step="0.5" value={by} onChange={e => setBy(+e.target.value)} className="w-full accent-emerald-500" />
            </div>
          </div>
        </div>
      </div>
      <div className="mb-4">
        <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
          Minkowski p = <span className="text-cyan-400 font-bold">{p.toFixed(1)}</span>
          <span className={`ml-2 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
            {p <= 1.05 ? '(= Manhattan)' : p >= 1.95 && p <= 2.05 ? '(= Euclidean)' : p >= 9.9 ? '(≈ Chebyshev)' : ''}
          </span>
        </label>
        <input type="range" min="1" max="10" step="0.1" value={p} onChange={e => setP(+e.target.value)} className="w-full accent-cyan-500" />
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3">
        {[
          { label: 'Euclidean (p=2)', val: euclidean.toFixed(3), color: 'text-indigo-400' },
          { label: 'Manhattan (p=1)', val: manhattan.toFixed(3), color: 'text-yellow-400' },
          { label: `Minkowski (p=${p.toFixed(1)})`, val: minkowski.toFixed(3), color: 'text-cyan-400' },
          { label: 'Chebyshev (p→∞)', val: chebyshev.toFixed(3), color: 'text-purple-400' },
        ].map(m => (
          <div key={m.label} className={`rounded-xl p-2.5 border text-center ${theme === 'dark' ? 'bg-slate-800 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className={`font-mono font-bold text-base ${m.color}`}>{m.val}</p>
            <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{m.label}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Bias-variance k demo ───────────────────────────────────────────────────────
function BiasVarianceDemo({ theme }) {
  const [k, setK] = useState(5)
  const canvasRef = useRef(null)
  const W = 380, H = 240

  // Fixed toy dataset: noisy sine wave
  const seed = 42
  const rng = (i) => ((Math.sin(seed + i * 127.1) + 1) / 2)
  const data = Array.from({ length: 20 }, (_, i) => ({
    x: (i / 19) * 340 + 20,
    y: Math.sin((i / 19) * Math.PI * 2) * 80 + 120 + (rng(i) - 0.5) * 50,
  }))

  const dist1d = (a, b) => Math.abs(a.x - b.x)

  const predict = (px) => {
    const sorted = [...data].sort((a, b) => dist1d(a, { x: px }) - dist1d(b, { x: px }))
    const neighbours = sorted.slice(0, k)
    return neighbours.reduce((s, n) => s + n.y, 0) / k
  }

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)

    // True curve
    ctx.beginPath()
    for (let px = 20; px <= 360; px++) {
      const fy = Math.sin(((px - 20) / 340) * Math.PI * 2) * 80 + 120
      px === 20 ? ctx.moveTo(px, fy) : ctx.lineTo(px, fy)
    }
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)'
    ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([])

    // kNN prediction curve
    const grad = ctx.createLinearGradient(0, 0, W, 0)
    grad.addColorStop(0, '#6366F1'); grad.addColorStop(0.5, '#8B5CF6'); grad.addColorStop(1, '#06B6D4')
    ctx.beginPath()
    for (let px = 20; px <= 360; px++) {
      const py = predict(px)
      px === 20 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)
    }
    ctx.strokeStyle = grad; ctx.lineWidth = 2.5; ctx.stroke()

    // Data points
    data.forEach(p => {
      ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2)
      ctx.fillStyle = '#10B981'; ctx.fill()
      ctx.strokeStyle = 'white'; ctx.lineWidth = 1.5; ctx.stroke()
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [k, theme])

  const label = k <= 2 ? 'Overfitting — jagged, follows noise' : k >= 14 ? 'Underfitting — too smooth, misses pattern' : 'Good fit — captures signal, ignores noise'
  const labelColor = k <= 2 ? 'text-red-400' : k >= 14 ? 'text-amber-400' : 'text-emerald-400'

  return (
    <div>
      <div className="mb-4">
        <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
          k = <span className="text-yellow-400 font-bold">{k}</span>
          <span className={`ml-3 font-semibold ${labelColor}`}>{label}</span>
        </label>
        <input type="range" min="1" max="19" step="1" value={k} onChange={e => setK(+e.target.value)} className="w-full accent-yellow-500" />
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
      <div className="flex gap-3 mt-3 text-xs">
        <div className={`flex gap-2 items-center ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          <div className="w-6 h-0.5 bg-white/20 border border-dashed" />
          <span>True curve</span>
        </div>
        <div className={`flex gap-2 items-center ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          <div className="w-6 h-0.5 bg-indigo-500" />
          <span>kNN prediction</span>
        </div>
        <div className={`flex gap-2 items-center ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          <div className="w-3 h-3 rounded-full bg-emerald-500" />
          <span>Data points</span>
        </div>
      </div>
    </div>
  )
}

// ── Curse of dimensionality visualiser ────────────────────────────────────────
function CurseDemo({ theme }) {
  const dims = [1, 2, 3, 5, 10, 20, 50, 100]
  // Expected fraction of hypercube volume covered to capture 10% of data
  const fraction = (d) => 0.1 ** (1 / d)
  return (
    <div>
      <p className={`text-xs mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
        To capture 10% of training data in a d-dimensional unit hypercube, the neighbourhood must cover this fraction of each axis:
      </p>
      <div className="space-y-2 mb-5">
        {dims.map(d => {
          const f = fraction(d)
          const pct = (f * 100).toFixed(1)
          return (
            <div key={d} className="flex items-center gap-3">
              <span className={`text-xs font-mono w-16 shrink-0 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>d = {d}</span>
              <div className={`flex-1 h-5 rounded-full overflow-hidden ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-200'}`}>
                <motion.div
                  className={`h-full rounded-full ${f > 0.8 ? 'bg-red-500' : f > 0.5 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${f * 100}%` }}
                  transition={{ duration: 0.5, delay: 0.05 * dims.indexOf(d) }}
                />
              </div>
              <span className={`text-xs font-mono w-14 shrink-0 text-right font-bold ${f > 0.8 ? 'text-red-400' : f > 0.5 ? 'text-amber-400' : 'text-emerald-400'}`}>{pct}%</span>
            </div>
          )
        })}
      </div>
      <Callout type="warning" title="What this means for kNN">
        In 100 dimensions, you must search 63% of the feature space just to find 10% of data points. Your "neighbours" are almost as far away as the entire dataset — the concept of locality breaks down completely.
      </Callout>
      <div className="grid grid-cols-2 gap-3 mt-4">
        {[
          { label: 'Low dims (d ≤ 5)', icon: '✅', desc: 'Neighbours are truly local. kNN works well. KD-trees give fast lookup.', color: 'text-emerald-400' },
          { label: 'High dims (d > 20)', icon: '⚠️', desc: 'All distances converge. Neighbourhood radius must cover most of the space. kNN accuracy degrades.', color: 'text-red-400' },
        ].map(m => (
          <div key={m.label} className={`rounded-xl border p-3 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <div className="flex items-center gap-2 mb-1">
              <span>{m.icon}</span>
              <span className={`text-xs font-semibold ${m.color}`}>{m.label}</span>
            </div>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{m.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function KNN() {
  const { theme } = useApp()
  const S = `rounded-2xl border p-6 mb-6 ${theme === 'dark' ? 'bg-slate-900/80 border-white/10' : 'bg-white border-gray-200'}`
  const LBL = `text-xs font-semibold uppercase tracking-wider mb-3 ${theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}`
  const H2 = `text-xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`
  const BODY = `text-sm leading-relaxed ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-3xl">

      {/* Hero */}
      <div className="mb-8">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-500/20 border border-cyan-500/30 mb-4">
          <span className="text-xs text-cyan-400 font-medium">Machine Learning • Instance-Based Learning</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          k-Nearest <span className="gradient-text">Neighbors</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          The simplest ML algorithm: classify or predict by looking at the <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>k most similar training examples</strong>. No training phase — just memory and distance.
        </p>
        <Callout type="analogy" title="Analogy: Ask your neighbours">
          You move to a new city and want to know if a neighbourhood is safe. You ask the 5 closest residents. If 4 say "safe" and 1 says "unsafe", you conclude it's safe — majority vote. That's kNN. The only question is: how do you measure "closest"?
        </Callout>
      </div>

      <TheoryBlock title="Core Concepts" cards={[
        { icon: '💤', title: 'Lazy Learning', body: 'kNN does zero computation at training time — it just stores all examples. All work happens at prediction time. No model is fitted; the training set IS the model.', mono: 'train: O(1)  predict: O(n·d)' },
        { icon: '📏', title: 'Distance Metric', body: 'Classification depends entirely on distance. Euclidean is standard. Manhattan is robust to outliers. Always scale features first — units and ranges matter.', mono: 'd(x,y) = (Σ|xᵢ−yᵢ|ᵖ)^(1/p)' },
        { icon: '🗳️', title: 'Majority Vote', body: 'For classification: count classes among the k neighbours and take the majority. For regression: average (or distance-weighted average) of neighbours\' target values.', mono: 'ŷ = mode({y₁,...,yₖ})' },
        { icon: '🎚️', title: 'Choosing k', body: 'Small k → high variance (overfits noise). Large k → high bias (ignores local structure). Optimal k found via cross-validation. Usually odd to avoid ties.', mono: 'k★ = argmin CV-error(k)' },
        { icon: '📐', title: 'Feature Scaling', body: 'Critical for kNN. A feature on scale [0,1000] dominates one on [0,1] in distance calculations. Always apply StandardScaler or MinMaxScaler before kNN.', mono: 'x_scaled = (x − μ) / σ' },
        { icon: '💀', title: 'Curse of Dimensionality', body: 'In high-dimensional spaces, all points become roughly equidistant. The k "nearest" neighbours are no longer meaningfully close. kNN degrades rapidly with dimension.', mono: 'E[d_max/d_min] → 1 as d→∞' },
      ]} />

      {/* Topic 1: Algorithm */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-cyan-500/5 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}`}>Topic 1</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>kNN Algorithm — Classification and Regression</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>How kNN predicts, from raw algorithm to weighted voting.</p>
      </div>

      <div className={S}>
        <p className={LBL}>Algorithm walkthrough</p>
        <h2 className={H2}>Four steps to a prediction</h2>
        <p className={`${BODY} mb-4`}>
          kNN is one of the most transparent ML algorithms. Given a new query point, it literally looks up similar training examples and aggregates their answers.
        </p>
        <TheoryBlock title="kNN prediction steps" cards={[
          { icon: '1️⃣', title: 'Compute distances', body: 'Calculate the distance between the query point and every training point using the chosen metric (Euclidean, Manhattan, etc.).', mono: 'd(xᵢ, x_query)  for all i' },
          { icon: '2️⃣', title: 'Sort and select k', body: 'Sort all training points by distance. Keep the k closest ones. These are the "neighbours".', mono: 'neighbours = argsort(d)[:k]' },
          { icon: '3️⃣', title: 'Aggregate', body: 'Classification: majority vote. Regression: mean (or weighted mean). Distance weighting gives closer neighbours more influence.', mono: 'ŷ = Σ wᵢ·yᵢ / Σ wᵢ  (weighted)' },
          { icon: '4️⃣', title: 'Return prediction', body: 'Output the class with most votes (classification) or the aggregated value (regression). No model parameters were ever updated.', mono: 'no weights, no fitting' },
        ]} />
        <Callout type="info" title="Classification vs Regression">
          The only difference is step 3. Classification takes the mode (majority class). Regression takes the mean (average target). Both benefit from distance weighting: w = 1/d, so closer points have more influence.
        </Callout>
        <p className={`${BODY} mb-4`}>Drag the <span className="text-yellow-400 font-bold">?</span> query point on the canvas to classify it. Change k and the distance metric to see how the decision changes:</p>
        <KNNCanvas theme={theme} />
        <DeepDive title="Weighted kNN vs uniform kNN">
          <p className={`text-sm ${BODY} mb-2`}>Uniform kNN gives equal weight to all k neighbours. Distance-weighted kNN uses w = 1/d, so a neighbour at distance 0.1 has 10× more influence than one at distance 1.0.</p>
          <TheoryBlock title="" cards={[
            { icon: '⚖️', title: 'Uniform weights', body: 'All k neighbours vote equally. Simple and robust. Can be influenced by distant neighbours in the k-set.', mono: 'ŷ = (1/k) Σ yᵢ' },
            { icon: '🎯', title: 'Distance weights', body: 'Closer neighbours have proportionally more say. Better accuracy on most datasets. Handles varying density better.', mono: 'ŷ = Σ (1/dᵢ)·yᵢ / Σ (1/dᵢ)' },
          ]} />
        </DeepDive>
      </div>

      <div className={S}>
        <p className={LBL}>kNN Regression</p>
        <h2 className={H2}>Predicting continuous values</h2>
        <p className={`${BODY} mb-4`}>
          kNN regression works identically to classification but aggregates numeric targets. It is a non-parametric, non-linear regressor — it can approximate any smooth function given enough data. The prediction is a locally weighted average.
        </p>
        <TheoryBlock title="Regression properties" cards={[
          { icon: '📈', title: 'Non-linear', body: 'kNN regression can fit any smooth curve — no assumption about the functional form. The prediction surface adapts to local density.', mono: 'f̂(x) = (1/k) Σ yᵢ' },
          { icon: '🌍', title: 'Local model', body: 'The model is different at every query point — it only looks at local neighbours. This contrasts with global models like linear regression which fit a single line.', mono: 'local average vs global fit' },
          { icon: '📉', title: 'Large k → smoother', body: 'As k increases, the prediction averages over more neighbours and becomes smoother (less responsive to individual points). Equivalent to increasing regularisation.', mono: 'k = n → predicts global mean' },
        ]} />
        <Callout type="warning" title="kNN regression pitfall: extrapolation">
          kNN regression cannot extrapolate beyond the range of the training data. A query point outside the training range will always return the average of the boundary neighbours — a constant. For extrapolation, use a parametric model.
        </Callout>
      </div>

      {/* Topic 2: Distance metrics */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>Topic 2</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Distance Metrics</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Euclidean, Manhattan, Minkowski — and when to use each.</p>
      </div>

      <div className={S}>
        <p className={LBL}>Minkowski family</p>
        <h2 className={H2}>One formula, infinite metrics</h2>
        <p className={`${BODY} mb-4`}>
          Euclidean, Manhattan, and Chebyshev distances are all special cases of the Minkowski distance with different values of p. Adjusting p changes the "shape" of the distance ball around a point.
        </p>
        <TheoryBlock title="Distance metric family" cards={[
          { icon: '📐', title: 'Euclidean (p=2)', body: 'Straight-line distance. Standard in most geometric contexts. Sensitive to large differences in any one dimension. Most common for continuous features.', mono: 'd = √(Σ(xᵢ−yᵢ)²)' },
          { icon: '🏙️', title: 'Manhattan (p=1)', body: 'Sum of absolute differences. Robust to outliers and extreme values. Geometrically: distance when you can only move along grid lines (city blocks).', mono: 'd = Σ|xᵢ−yᵢ|' },
          { icon: '♾️', title: 'Chebyshev (p→∞)', body: 'Maximum absolute difference across dimensions. Useful when the largest single-dimension deviation matters most. Used in board game (king) moves.', mono: 'd = max|xᵢ−yᵢ|' },
        ]} />
        <Callout type="formula" mono="d_Minkowski(x,y) = (Σᵢ |xᵢ−yᵢ|ᵖ)^(1/p)">
          p=1 → Manhattan. p=2 → Euclidean. As p→∞, only the largest dimension difference survives → Chebyshev. p&lt;1 is technically not a metric (violates triangle inequality) but is sometimes used in sparse high-dimensional settings.
        </Callout>
        <p className={`${BODY} mb-4`}>Drag the two points and sweep p from 1 to 10 to see all four distance values update live:</p>
        <DistanceViz theme={theme} />
        <DeepDive title="Other distance metrics">
          <TheoryBlock title="" cards={[
            { icon: '📊', title: 'Cosine similarity', body: 'Measures angle between vectors, not magnitude. Ideal for text/NLP where document length varies. Two documents with same proportions = distance 0.', mono: 'cos(x,y) = (x·y) / (|x||y|)' },
            { icon: '🧬', title: 'Hamming distance', body: 'Count of positions where two strings differ. Used for categorical features and binary data. kNN with Hamming = nearest binary neighbour.', mono: 'H(x,y) = Σ 𝟙[xᵢ ≠ yᵢ]' },
            { icon: '📈', title: 'Mahalanobis', body: 'Euclidean distance normalised by the covariance matrix. Scale-invariant and accounts for correlations between features. Expensive to compute.', mono: 'd = √((x−y)ᵀ Σ⁻¹ (x−y))' },
          ]} />
        </DeepDive>
      </div>

      {/* Topic 3: Choosing k */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-purple-500/5 border-purple-500/20' : 'bg-purple-50 border-purple-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-purple-400' : 'text-purple-600'}`}>Topic 3</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Choosing k — Bias-Variance Trade-off</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>The most important hyperparameter in kNN — and how to pick it.</p>
      </div>

      <div className={S}>
        <p className={LBL}>Effect of k on model complexity</p>
        <h2 className={H2}>k controls the smoothness of the decision boundary</h2>
        <p className={`${BODY} mb-4`}>
          k is the sole hyperparameter of kNN (apart from the distance metric). It directly controls the model's complexity: small k = complex, wiggly boundary; large k = smooth, simple boundary.
        </p>
        <TheoryBlock title="k and bias-variance" cards={[
          { icon: '🎯', title: 'k=1 (lowest bias)', body: 'Every training point is its own class. Perfect training accuracy. Extremely jagged boundary that overfits noise. Test accuracy is poor.', mono: 'train acc = 100%, test acc ↓' },
          { icon: '⚖️', title: 'k=√n (rule of thumb)', body: 'A common starting point. Balances local sensitivity and global smoothness. √n grows slowly — keeps neighbourhood small relative to dataset.', mono: 'k★ ≈ √n  (heuristic)' },
          { icon: '📊', title: 'k=n (maximum bias)', body: 'All training points are neighbours. Prediction is always the global majority class. Zero variance — but completely ignores local structure.', mono: 'ŷ = global majority class' },
        ]} />
        <Callout type="warning" title="Use odd k for binary classification">
          Even k can produce ties (equal votes for each class). Odd k avoids this. If you must use even k, break ties by distance to nearest point.
        </Callout>
        <p className={`${BODY} mb-4`}>Drag k to see how the kNN regression fit changes — from jagged overfit at k=1 to flat underfit at high k:</p>
        <BiasVarianceDemo theme={theme} />
        <DeepDive title="How to find optimal k in practice">
          <p className={`text-sm ${BODY} mb-2`}>Use k-fold cross-validation. Train kNN for k = 1, 3, 5, ..., √n·2. For each k, compute validation error. Pick the k that minimises it.</p>
          <Callout type="formula" mono="k★ = argmin_{k odd} CV_error(k)">
            In sklearn: <span className="font-mono">GridSearchCV(KNeighborsClassifier(), {'{'}'n_neighbors': range(1, 30, 2){'}'}, cv=5)</span>
          </Callout>
          <p className={`text-sm ${BODY} mt-2`}>The optimal k is typically in the range [3, 20] for most datasets. Very large k is rarely useful unless the dataset is tiny.</p>
        </DeepDive>
      </div>

      {/* Topic 4: Curse of dimensionality */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-red-500/5 border-red-500/20' : 'bg-red-50 border-red-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-red-400' : 'text-red-600'}`}>Topic 4</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Curse of Dimensionality in kNN</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Why kNN struggles in high-dimensional spaces — and how to mitigate it.</p>
      </div>

      <div className={S}>
        <p className={LBL}>The volume problem</p>
        <h2 className={H2}>Neighbours stop being neighbours</h2>
        <p className={`${BODY} mb-4`}>
          In a 1D space, "nearby" is intuitive. In 100D, the concept breaks down: the volume of the neighbourhood grows exponentially with dimension. To capture even 10% of data, you must cover most of the feature space.
        </p>
        <Callout type="formula" mono="r = (fraction)^(1/d)">
          To capture a fraction f of data from a d-dimensional unit hypercube, your neighbourhood must extend r = f^(1/d) along each axis. At d=100 and f=0.1: r = 0.1^(1/100) ≈ 0.977 — almost the entire space.
        </Callout>
        <CurseDemo theme={theme} />
        <DeepDive title="Distance concentration phenomenon">
          <p className={`text-sm ${BODY} mb-2`}>As dimensionality grows, all pairwise distances in a dataset converge to the same value. Formally:</p>
          <Callout type="formula" mono="E[d_max − d_min] / E[d_min] → 0 as d → ∞">
            This means the nearest and farthest neighbours become indistinguishable. The relative contrast of distances vanishes. kNN loses all discriminative power.
          </Callout>
          <p className={`text-sm ${BODY} mt-2`}>This is not just a kNN problem — it affects all distance-based algorithms including SVMs with RBF kernels, k-means clustering, and anomaly detection.</p>
        </DeepDive>
      </div>

      <div className={S}>
        <p className={LBL}>Mitigation strategies</p>
        <h2 className={H2}>Making kNN work in high dimensions</h2>
        <p className={`${BODY} mb-4`}>
          Several strategies can recover kNN performance when the number of features is large.
        </p>
        <TheoryBlock title="High-dimension strategies" cards={[
          { icon: '📉', title: 'Dimensionality reduction', body: 'Apply PCA, t-SNE, or UMAP before kNN. Compress d features into d\' ≪ d dimensions that preserve the most variance. kNN in d\' = 10–50 often works well.', mono: 'PCA: X → X_reduced ∈ ℝ^{d\'}' },
          { icon: '✂️', title: 'Feature selection', body: 'Remove irrelevant or redundant features. Mutual information, ANOVA F-test, or recursive feature elimination (RFE). Fewer informative features → better neighbourhoods.', mono: 'select top-d\' by MI score' },
          { icon: '⚡', title: 'Approximate kNN', body: 'Use FAISS, Annoy, or HNSW for approximate nearest-neighbour search. Trades small accuracy loss for massive speed-up in high dimensions.', mono: 'FAISS: GPU-accelerated ANN' },
          { icon: '🌲', title: 'Ball trees', body: 'KD-trees degrade in high-d; Ball trees partition via hyperspheres and generalise better. Still breaks down above ~30 dimensions but better than KD-tree.', mono: 'sklearn: algorithm="ball_tree"' },
          { icon: '🔢', title: 'Increase training data', body: 'The required data to maintain neighbourhood density grows exponentially with d. Doubling dimensions requires squaring the dataset. Often impractical.', mono: 'n ∝ ε^{−d}  (coverage bound)' },
          { icon: '🔄', title: 'Try other algorithms', body: 'For high-d tabular data, tree ensembles (Random Forest, XGBoost) are usually better. kNN is best suited for low-to-medium dimensional, dense datasets.', mono: 'd > 20 → prefer tree ensembles' },
        ]} />
        <Callout type="success" title="When kNN shines">
          kNN is excellent for: (1) low-dimensional data with clear clusters, (2) recommendation systems with user–item vectors, (3) image search via embedding vectors (after neural net feature extraction), (4) anomaly detection via nearest-neighbour distance.
        </Callout>
      </div>

      {/* Code */}
      <div className={S}>
        <p className={LBL}>Python Example — Classification</p>
        <h2 className={H2}>kNN on Iris with feature scaling</h2>
        <p className={`${BODY} mb-3`}>Full classification workflow including standardisation, prediction, and finding nearest neighbours.</p>
        <CodeBlock code={PYTHON_CODE_CLASSIFY} />
      </div>

      <div className={S}>
        <p className={LBL}>Python Example — Regression and k selection</p>
        <h2 className={H2}>kNN regression with cross-validated k</h2>
        <p className={`${BODY} mb-3`}>Regression on a noisy sine wave — compare R² across different k values to find the optimal one.</p>
        <CodeBlock code={PYTHON_CODE_REGRESSION} />
      </div>

      <div className={S}>
        <p className={LBL}>Python Example — Curse of Dimensionality</p>
        <h2 className={H2}>Accuracy vs number of dimensions</h2>
        <p className={`${BODY} mb-3`}>Empirically observe how kNN accuracy degrades as features grow from 2 to 200 dimensions.</p>
        <CodeBlock code={PYTHON_CODE_CURSE} />
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme === 'dark' ? 'bg-cyan-500/10 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>10 questions covering all topics • +100 XP on completion</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="knn" />
      </div>
    </motion.div>
  )
}
