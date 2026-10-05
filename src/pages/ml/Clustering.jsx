import { useState, useRef, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

// Deterministic PRNG
function makePRNG(seed = 42) {
  let s = seed
  return () => {
    s = (s * 16807 + 11) % 2147483647
    return (s - 1) / 2147483646
  }
}

// ── Python Code Examples ────────────────────────────────────────────────────────
const PYTHON_CODE_KMEANS = `from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset with 4 clusters
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.7, random_state=42)

# 1. K-Means++: Distance-weighted initialization prevents poor local optima
# WCSS (Within-Cluster Sum of Squares) is stored in kmeans.inertia_
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

print(f"Final WCSS (Inertia): {kmeans.inertia_:.2f}")
print(f"Cluster Centroids:\\n{kmeans.cluster_centers_.round(2)}")

# 2. Silhouette Score for cluster quality assessment: s = (b - a) / max(a, b)
score = silhouette_score(X, labels)
print(f"Silhouette Score (optimal k=4): {score:.3f}")

# 3. Mini-Batch K-Means for massive datasets (stochastic mini-batch centroid updates)
mini_kmeans = MiniBatchKMeans(n_clusters=4, batch_size=64, random_state=42)
mini_labels = mini_kmeans.fit_predict(X)
print(f"Mini-Batch Inertia: {mini_kmeans.inertia_:.2f}")`

const PYTHON_CODE_GMM_EM = `from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import numpy as np

# Create anisotropic (elongated elliptical) clusters where K-Means fails
X, _ = make_blobs(n_samples=400, centers=3, random_state=42)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)

# Gaussian Mixture Model (GMM) with Expectation-Maximization (EM)
# covariance_type='full' allows arbitrary orientation and elongation
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_aniso)

# Soft probabilistic assignment: returns responsibility gamma_ik for each cluster
probs = gmm.predict_proba(X_aniso)
hard_preds = gmm.predict(X_aniso)

# Sample query point: examine soft probabilities
query_idx = 0
print(f"Point {query_idx} Soft Probabilities:")
for c, p in enumerate(probs[query_idx]):
    print(f"  Cluster {c}: {p:.2%}")

# Anomaly Detection via GMM Density
# Low log-likelihood indicates samples in sparse, unobserved regions
log_densities = gmm.score_samples(X_aniso)
threshold = np.percentile(log_densities, 5) # bottom 5% as outliers
anomalies = X_aniso[log_densities < threshold]
print(f"\\nDetected {len(anomalies)} potential anomalies (< 5th percentile log-likelihood)")`

const QUIZ_QUESTIONS = [
  {
    question: 'What is the objective function that the standard K-Means algorithm minimizes?',
    options: [
      'Between-Cluster Maximum Margin',
      'Within-Cluster Sum of Squares (WCSS / Inertia): J = Σ Σ ||x_i - μ_k||²',
      'Total dataset mutual information',
      'Maximum likelihood of a Poisson distribution'
    ],
    correct: 1,
    explanation: 'K-Means optimizes the Within-Cluster Sum of Squares (Inertia), computing the sum of squared Euclidean distances from every point to its assigned centroid μ_k.'
  },
  {
    question: 'How does K-Means++ initialization improve upon standard random centroid initialization?',
    options: [
      'It places all centroids at the origin (0, 0)',
      'It seeds the first centroid randomly, then chooses subsequent centroids with probability proportional to the squared distance D(x)² from the nearest existing centroid',
      'It uses deep neural networks to place centroids',
      'It sorts all points along the first principal component'
    ],
    correct: 1,
    explanation: 'By selecting subsequent centroids with probability D(x)² / Σ D(x)², K-Means++ spreads initial centroids far apart across the data space, preventing poor local minima and speeding up convergence.'
  },
  {
    question: 'In the Elbow Method for selecting the number of clusters k, what is the practitioner looking for on the Inertia vs k plot?',
    options: [
      'The point where inertia drops strictly to zero',
      'The inflection point (elbow) where the marginal reduction in inertia drops sharply and begins leveling off',
      'The highest peak on the curve',
      'The point where k equals the number of features d'
    ],
    correct: 1,
    explanation: 'As k increases, inertia continually decreases. The "elbow" is the point of diminishing returns where adding another cluster provides negligible improvement in compactness.'
  },
  {
    question: 'What does a Silhouette Coefficient close to +1.0 indicate for a sample?',
    options: [
      'The sample is on the borderline between two clusters',
      'The sample is tightly clustered within its assigned group (small intra-cluster distance a) and well separated from neighboring clusters (large b)',
      'The sample has been misclassified into the wrong cluster',
      'The dataset contains zero variance'
    ],
    correct: 1,
    explanation: 'The silhouette formula s = (b - a) / max(a, b) ranges from -1 to +1. Values near +1 indicate dense, well-isolated clusters. Values near 0 indicate borderline points; negative values indicate likely wrong cluster assignment.'
  },
  {
    question: 'What happens in the Expectation (E-step) of the Expectation-Maximization (EM) algorithm for GMMs?',
    options: [
      'The centroid locations and covariance matrices are updated',
      'Soft &ldquo;responsibilities&rdquo; γ_ik are computed, representing the posterior probability that point i was generated by Gaussian component k',
      'Data points with low probabilities are deleted from memory',
      'The learning rate is decayed'
    ],
    correct: 1,
    explanation: 'In the E-step, current parameters (μ, Σ, π) are held fixed, and Bayes\' rule is used to evaluate the posterior responsibility γ_ik = P(Z_i = k | x_i, θ).'
  },
  {
    question: 'What happens in the Maximization (M-step) of the EM algorithm for GMMs?',
    options: [
      'Points are assigned to their single nearest centroid',
      'The model parameters (cluster priors π_k, means μ_k, and covariance matrices Σ_k) are re-estimated using the newly computed responsibilities as weights',
      'The number of clusters k is automatically incremented',
      'The gradients of all neural layers are backpropagated'
    ],
    correct: 1,
    explanation: 'In the M-step, responsibilities γ_ik act as soft weights to update means μ_k = (1/N_k) Σ γ_ik x_i, covariance matrices Σ_k, and mixing proportions π_k.'
  },
  {
    question: 'Why does K-Means struggle with elongated (elliptical) or unequal-variance clusters compared to GMM?',
    options: [
      'K-Means assumes hard spherical clusters with equal isotropic variance because it uses Euclidean distance, while GMM with full covariance can model arbitrary ellipsoids and orientations',
      'K-Means cannot compute distances in 2D',
      'GMM always uses fewer parameters than K-Means',
      'K-Means requires labeled data'
    ],
    correct: 0,
    explanation: 'Because Euclidean distance weights all dimensions equally without correlation, K-Means carves out rigid spherical Voronoi partitions. GMM models full covariance matrices Σ_k, capturing elliptical shapes and cross-feature correlations.'
  },
  {
    question: 'How can Gaussian Mixture Models be used directly for Anomaly / Outlier Detection?',
    options: [
      'By removing any cluster with fewer than 10 points',
      'By evaluating the overall probability density p(x) for each sample; points falling below a low density threshold (e.g. bottom 1-5%) are flagged as anomalies',
      'By measuring the silhouette score of outliers',
      'By running K-Means on the residual vectors'
    ],
    correct: 1,
    explanation: 'GMM provides a full continuous probability density function p(x) = Σ π_k N(x|μ_k, Σ_k). Points that fall in extremely low-density tails represent rare, anomalous events.'
  },
  {
    question: 'What is the primary operational advantage of Mini-Batch K-Means over standard Lloyd\'s K-Means?',
    options: [
      'It is guaranteed to find the global optimum every run',
      'It drastically speeds up computation and memory throughput by updating centroids on small stochastic subsamples rather than scanning the full dataset every iteration',
      'It does not require specifying the parameter k',
      'It produces perfect circular clusters'
    ],
    correct: 1,
    explanation: 'Standard K-Means scans all N data points every iteration. Mini-Batch K-Means uses small random batches (e.g. 100-1000 points) to perform incremental sliding updates, converging orders of magnitude faster on massive datasets.'
  },
  {
    question: 'In the mathematical formulation of GMMs, what constraint must the mixing proportions (priors) π_k satisfy?',
    options: [
      'All π_k must equal 1.0',
      '0 ≤ π_k ≤ 1 for all k, and their sum Σ_{k=1}^K π_k must equal 1.0',
      'π_k must be negative for outlier clusters',
      'π_k is always equal to 1 / k'
    ],
    correct: 1,
    explanation: 'The mixing weights π_k represent the prior probability that an unobserved sample belongs to component k. As valid probabilities, they must be non-negative and sum to 1.0 across all components.'
  }
]

// ── Interactive Component 1: Step-by-Step K-Means Simulator ──────────────────
function KMeansSimulatorCanvas({ theme }) {
  const W = 460
  const H = 260
  const K = 3
  const [initMethod, setInitMethod] = useState('kmeanspp') // 'random' or 'kmeanspp'
  const [stepState, setStepState] = useState('init') // 'init', 'assigned', 'updated'
  const [iteration, setIteration] = useState(0)

  // 3 distinct ground-truth cluster blobs
  const points = useMemo(() => {
    const rand = makePRNG(12345)
    const pts = []
    const centers = [
      { x: 110, y: 80 },
      { x: 340, y: 90 },
      { x: 220, y: 195 }
    ]
    centers.forEach(c => {
      for (let i = 0; i < 18; i++) {
        const r = rand() * 45
        const theta = rand() * Math.PI * 2
        pts.push({
          x: c.x + r * Math.cos(theta),
          y: c.y + r * Math.sin(theta),
          cluster: -1
        })
      }
    })
    return pts
  }, [])

  // Pure initialization helper
  const computeInitialCentroids = (method, seedVal) => {
    const rand = makePRNG(seedVal)
    let newCentroids = []

    if (method === 'random') {
      const indices = []
      while (indices.length < K) {
        const idx = Math.floor(rand() * points.length)
        if (!indices.includes(idx)) indices.push(idx)
      }
      newCentroids = indices.map(idx => ({ x: points[idx].x, y: points[idx].y }))
    } else {
      // K-Means++ initialization
      const firstIdx = Math.floor(rand() * points.length)
      newCentroids.push({ x: points[firstIdx].x, y: points[firstIdx].y })

      while (newCentroids.length < K) {
        const distsSq = points.map(p => {
          let minDistSq = Infinity
          newCentroids.forEach(c => {
            const d2 = (p.x - c.x) ** 2 + (p.y - c.y) ** 2
            if (d2 < minDistSq) minDistSq = d2
          })
          return minDistSq
        })

        const sumD2 = distsSq.reduce((a, b) => a + b, 0)
        let r = rand() * sumD2
        let chosenIdx = 0
        for (let i = 0; i < points.length; i++) {
          r -= distsSq[i]
          if (r <= 0) {
            chosenIdx = i
            break
          }
        }
        newCentroids.push({ x: points[chosenIdx].x, y: points[chosenIdx].y })
      }
    }
    return newCentroids
  }

  // Centroids state
  const [centroids, setCentroids] = useState(() => computeInitialCentroids('kmeanspp', 42))
  const [assignments, setAssignments] = useState(() => new Array(points.length).fill(-1))

  const handleReseed = (method = initMethod) => {
    const nextSeed = Math.floor(Math.random() * 99999) + 1
    const newCentroids = computeInitialCentroids(method, nextSeed)
    setCentroids(newCentroids)
    setAssignments(new Array(points.length).fill(-1))
    setStepState('init')
    setIteration(0)
  }

  const handleInitMethodChange = (method) => {
    setInitMethod(method)
    handleReseed(method)
  }

  // Perform one step of Lloyd's algorithm
  const handleNextStep = () => {
    if (centroids.length < K) return

    if (stepState === 'init' || stepState === 'updated') {
      // ASSIGNMENT STEP: assign each point to closest centroid
      const newAssignments = points.map(p => {
        let minDistSq = Infinity
        let bestCluster = 0
        centroids.forEach((c, idx) => {
          const d2 = (p.x - c.x) ** 2 + (p.y - c.y) ** 2
          if (d2 < minDistSq) {
            minDistSq = d2
            bestCluster = idx
          }
        })
        return bestCluster
      })
      setAssignments(newAssignments)
      setStepState('assigned')
    } else {
      // UPDATE STEP: move centroids to cluster centers of mass
      const newCentroids = centroids.map((c, k) => {
        let sumX = 0, sumY = 0, count = 0
        points.forEach((p, idx) => {
          if (assignments[idx] === k) {
            sumX += p.x
            sumY += p.y
            count++
          }
        })
        return count > 0 ? { x: sumX / count, y: sumY / count } : c
      })
      setCentroids(newCentroids)
      setStepState('updated')
      setIteration(it => it + 1)
    }
  }

  // Compute current WCSS / Inertia
  const currentInertia = useMemo(() => {
    if (assignments.length === 0 || centroids.length === 0) return 0
    let wcss = 0
    points.forEach((p, idx) => {
      const cIdx = assignments[idx]
      if (cIdx >= 0 && centroids[cIdx]) {
        wcss += (p.x - centroids[cIdx].x) ** 2 + (p.y - centroids[cIdx].y) ** 2
      }
    })
    return wcss
  }, [points, assignments, centroids])

  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'
    const clusterColors = ['#6366F1', '#10B981', '#F59E0B']

    // Draw background Voronoi cells if assigned
    if (stepState !== 'init' && centroids.length === K) {
      const step = 8
      for (let x = 0; x < W; x += step) {
        for (let y = 0; y < H; y += step) {
          let minDistSq = Infinity
          let nearestK = 0
          centroids.forEach((c, idx) => {
            const d2 = (x - c.x) ** 2 + (y - c.y) ** 2
            if (d2 < minDistSq) {
              minDistSq = d2
              nearestK = idx
            }
          })
          const baseColor = clusterColors[nearestK]
          ctx.fillStyle = dark
            ? baseColor + '18'
            : baseColor + '12'
          ctx.fillRect(x, y, step, step)
        }
      }
    }

    // Draw assignment connection lines
    if (stepState === 'assigned') {
      points.forEach((p, idx) => {
        const cIdx = assignments[idx]
        if (cIdx >= 0 && centroids[cIdx]) {
          ctx.beginPath()
          ctx.moveTo(p.x, p.y)
          ctx.lineTo(centroids[cIdx].x, centroids[cIdx].y)
          ctx.strokeStyle = dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)'
          ctx.lineWidth = 1
          ctx.stroke()
        }
      })
    }

    // Draw points
    points.forEach((p, idx) => {
      const cIdx = assignments[idx]
      ctx.beginPath()
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2)
      ctx.fillStyle = cIdx >= 0 ? clusterColors[cIdx] : (dark ? '#94A3B8' : '#64748B')
      ctx.fill()
      ctx.strokeStyle = dark ? '#0F172A' : '#FFFFFF'
      ctx.lineWidth = 1.5
      ctx.stroke()
    })

    // Draw centroids (Large pulsating crosshairs)
    centroids.forEach((c, idx) => {
      ctx.beginPath()
      ctx.arc(c.x, c.y, 11, 0, Math.PI * 2)
      ctx.fillStyle = clusterColors[idx]
      ctx.fill()
      ctx.strokeStyle = '#FFFFFF'
      ctx.lineWidth = 2.5
      ctx.stroke()

      // Center cross
      ctx.strokeStyle = '#000000'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(c.x - 5, c.y)
      ctx.lineTo(c.x + 5, c.y)
      ctx.moveTo(c.x, c.y - 5)
      ctx.lineTo(c.x, c.y + 5)
      ctx.stroke()
    })

    // Legend
    ctx.font = '10px monospace'
    ctx.fillStyle = dark ? '#94A3B8' : '#64748B'
    ctx.fillText(
      stepState === 'init'
        ? 'Step 0: Centroids Initialized'
        : stepState === 'assigned'
        ? `Iteration ${iteration}: Assignment Step (Voronoi partitions)`
        : `Iteration ${iteration}: Update Step (Centroids shifted to means)`,
      12,
      20
    )
  }, [points, centroids, assignments, stepState, iteration, theme, W, H, K])

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <label className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Init:
          </label>
          <select
            value={initMethod}
            onChange={e => handleInitMethodChange(e.target.value)}
            className={`px-2.5 py-1.5 rounded-lg text-xs font-medium border outline-none ${
              theme === 'dark' ? 'bg-slate-800 border-white/10 text-white' : 'bg-white border-gray-200 text-gray-800'
            }`}
          >
            <option value="kmeanspp">K-Means++ (Smart D² seeding)</option>
            <option value="random">Random Sampling (Standard)</option>
          </select>

          <button
            onClick={() => handleReseed(initMethod)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
              theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-300 hover:text-white' : 'bg-gray-100 border-gray-200 text-gray-700 hover:bg-gray-200'
            }`}
          >
            ↺ Re-seed
          </button>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleNextStep}
            className="px-4 py-1.5 rounded-xl bg-indigo-500 hover:bg-indigo-600 text-white text-xs font-bold shadow-sm transition-all"
          >
            {stepState === 'init' || stepState === 'updated' ? '▶ Run Assignment Step' : '▶ Run Update Step'}
          </button>
        </div>
      </div>

      <div className={`rounded-2xl border overflow-hidden relative mb-2 ${
        theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
      }`}>
        <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
      </div>

      <div className="flex flex-wrap items-center justify-between text-xs py-1 px-2">
        <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>
          Current Status: <strong className="font-mono text-indigo-400">
            {stepState === 'init' ? 'Awaiting 1st Step' : stepState === 'assigned' ? 'Points Partitioned' : 'Means Recalculated'}
          </strong>
        </span>
        <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>
          Inertia (WCSS): <strong className="font-mono text-emerald-400">{currentInertia.toFixed(0)}</strong>
        </span>
      </div>
    </div>
  )
}

// ── Interactive Component 2: Elbow Method & Silhouette Curve Analyzer ────────
function ElbowSilhouetteAnalyzer({ theme }) {
  const [metric, setMetric] = useState('elbow') // 'elbow' vs 'silhouette'
  const [selectedK, setSelectedK] = useState(3)
  const W = 460
  const H = 220

  // Precomputed realistic curves for 3 ground-truth blobs
  const data = useMemo(() => [
    { k: 1, inertia: 18200, silhouette: 0.12 },
    { k: 2, inertia: 9100,  silhouette: 0.48 },
    { k: 3, inertia: 2800,  silhouette: 0.74 }, // optimal kink & peak!
    { k: 4, inertia: 2200,  silhouette: 0.58 },
    { k: 5, inertia: 1750,  silhouette: 0.49 },
    { k: 6, inertia: 1350,  silhouette: 0.41 },
    { k: 7, inertia: 1050,  silhouette: 0.36 },
    { k: 8, inertia: 820,   silhouette: 0.32 },
  ], [])

  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'
    const padL = 45, padR = 25, padT = 25, padB = 35
    const plotW = W - padL - padR
    const plotH = H - padT - padB

    const maxVal = metric === 'elbow' ? 20000 : 1.0
    const minVal = 0

    const toX = (k) => padL + ((k - 1) / 7) * plotW
    const toY = (val) => padT + (1 - (val - minVal) / (maxVal - minVal)) * plotH

    // Grid lines
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
    ctx.lineWidth = 1
    for (let k = 1; k <= 8; k++) {
      ctx.beginPath()
      ctx.moveTo(toX(k), padT)
      ctx.lineTo(toX(k), padT + plotH)
      ctx.stroke()
    }

    // Draw main curve
    ctx.beginPath()
    ctx.strokeStyle = metric === 'elbow' ? '#F59E0B' : '#6366F1'
    ctx.lineWidth = 2.5

    data.forEach((d, idx) => {
      const val = metric === 'elbow' ? d.inertia : d.silhouette
      const cx = toX(d.k)
      const cy = toY(val)
      if (idx === 0) ctx.moveTo(cx, cy)
      else ctx.lineTo(cx, cy)
    })
    ctx.stroke()

    // Draw dots
    data.forEach(d => {
      const val = metric === 'elbow' ? d.inertia : d.silhouette
      const cx = toX(d.k)
      const cy = toY(val)
      const isSelected = d.k === selectedK

      ctx.beginPath()
      ctx.arc(cx, cy, isSelected ? 6.5 : 4, 0, Math.PI * 2)
      ctx.fillStyle = isSelected ? '#10B981' : (metric === 'elbow' ? '#F59E0B' : '#6366F1')
      ctx.fill()
      ctx.strokeStyle = dark ? '#0F172A' : '#FFFFFF'
      ctx.lineWidth = 2
      ctx.stroke()
    })

    // Optimal k = 3 annotation line
    const optX = toX(3)
    ctx.beginPath()
    ctx.moveTo(optX, padT)
    ctx.lineTo(optX, padT + plotH)
    ctx.strokeStyle = '#10B981'
    ctx.lineWidth = 1.5
    ctx.setLineDash([3, 3])
    ctx.stroke()
    ctx.setLineDash([])

    // Ticks & labels
    ctx.font = '10px monospace'
    ctx.fillStyle = dark ? '#94A3B8' : '#64748B'
    ctx.textAlign = 'center'
    for (let k = 1; k <= 8; k++) {
      ctx.fillText(`k=${k}`, toX(k), padT + plotH + 15)
    }
    ctx.fillText('Number of Clusters (k)', W / 2, H - 4)

    ctx.textAlign = 'right'
    if (metric === 'elbow') {
      ctx.fillText('20k', padL - 6, toY(20000) + 4)
      ctx.fillText('10k', padL - 6, toY(10000) + 4)
      ctx.fillText('0', padL - 6, toY(0) + 4)
    } else {
      ctx.fillText('1.0', padL - 6, toY(1.0) + 4)
      ctx.fillText('0.5', padL - 6, toY(0.5) + 4)
      ctx.fillText('0.0', padL - 6, toY(0.0) + 4)
    }

    // Legend
    ctx.textAlign = 'left'
    ctx.fillStyle = '#10B981'
    ctx.fillText('★ Optimal k=3 (Elbow inflection & Silhouette peak)', 50, 16)
  }, [metric, selectedK, data, theme, W, H])

  const selectedData = data.find(d => d.k === selectedK)

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setMetric('elbow')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              metric === 'elbow'
                ? 'bg-amber-500 text-white shadow-sm'
                : theme === 'dark' ? 'bg-slate-800 text-gray-400' : 'bg-gray-100 text-gray-700'
            }`}
          >
            📉 Elbow Method (Inertia)
          </button>
          <button
            onClick={() => setMetric('silhouette')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              metric === 'silhouette'
                ? 'bg-indigo-500 text-white shadow-sm'
                : theme === 'dark' ? 'bg-slate-800 text-gray-400' : 'bg-gray-100 text-gray-700'
            }`}
          >
            📊 Silhouette Score (Peak)
          </button>
        </div>

        <div className="flex items-center gap-2">
          <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Inspect k:
          </span>
          <select
            value={selectedK}
            onChange={e => setSelectedK(parseInt(e.target.value, 10))}
            className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold border outline-none ${
              theme === 'dark' ? 'bg-slate-800 border-white/10 text-white' : 'bg-white border-gray-200 text-gray-800'
            }`}
          >
            {[1, 2, 3, 4, 5, 6, 7, 8].map(k => (
              <option key={k} value={k}>k = {k}</option>
            ))}
          </select>
        </div>
      </div>

      <div className={`rounded-2xl border overflow-hidden relative mb-2 ${
        theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
      }`}>
        <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
      </div>

      <div className={`p-3 rounded-xl border flex flex-wrap items-center justify-between text-xs ${
        theme === 'dark' ? 'bg-slate-900/60 border-white/10' : 'bg-gray-50 border-gray-200'
      }`}>
        <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
          At <strong>k = {selectedK}</strong>: Inertia = <span className="font-mono text-amber-400 font-bold">{selectedData?.inertia}</span> | Silhouette = <span className="font-mono text-indigo-400 font-bold">{selectedData?.silhouette.toFixed(2)}</span>
        </span>
        <span className="text-emerald-400 font-medium">
          {selectedK === 3 ? '✓ True ground-truth cluster count identified!' : 'Sub-optimal clustering partition'}
        </span>
      </div>
    </div>
  )
}

// ── Interactive Component 3: GMM vs K-Means Soft Assignment & Ellipsoid Canvas ─
function GmmVsKMeansCanvas({ theme }) {
  const [modelType, setModelType] = useState('gmm') // 'kmeans' vs 'gmm'
  const W = 460
  const H = 260

  // 2 anisotropic elongated clusters angled diagonally
  const dataset = useMemo(() => {
    const rand = makePRNG(998877)
    const pts = []
    // Cluster A: elongated along x = y
    for (let i = 0; i < 30; i++) {
      const u = (rand() - 0.5) * 140
      const v = (rand() - 0.5) * 28
      pts.push({
        x: 140 + u * 0.707 - v * 0.707,
        y: 110 + u * 0.707 + v * 0.707,
        trueCluster: 0
      })
    }
    // Cluster B: elongated horizontally below
    for (let i = 0; i < 30; i++) {
      const u = (rand() - 0.5) * 130
      const v = (rand() - 0.5) * 30
      pts.push({
        x: 310 + u * 0.9 - v * 0.2,
        y: 180 + u * 0.2 + v * 0.9,
        trueCluster: 1
      })
    }
    return pts
  }, [])

  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'

    // Centroids for K-Means (spherical centers)
    const kmCenters = [
      { x: 140, y: 110 },
      { x: 310, y: 180 }
    ]

    // GMM Gaussian component parameters: means and 2x2 covariance ellipses
    const gmmComponents = [
      { meanX: 140, meanY: 110, rx: 65, ry: 20, angle: Math.PI / 4, color: '#6366F1' },
      { meanX: 310, meanY: 180, rx: 60, ry: 20, angle: 0.15, color: '#10B981' }
    ]

    // Rasterize background
    const step = 8
    for (let x = 0; x < W; x += step) {
      for (let y = 0; y < H; y += step) {
        if (modelType === 'kmeans') {
          // Hard Voronoi Euclidean partition
          const d0 = (x - kmCenters[0].x) ** 2 + (y - kmCenters[0].y) ** 2
          const d1 = (x - kmCenters[1].x) ** 2 + (y - kmCenters[1].y) ** 2
          const isA = d0 < d1
          ctx.fillStyle = isA
            ? (dark ? 'rgba(99, 102, 241, 0.12)' : 'rgba(99, 102, 241, 0.08)')
            : (dark ? 'rgba(16, 185, 129, 0.12)' : 'rgba(16, 185, 129, 0.08)')
        } else {
          // Soft GMM probability gradient
          // Mahalanobis distance to component 0
          const dx0 = x - gmmComponents[0].meanX
          const dy0 = y - gmmComponents[0].meanY
          const c0 = Math.cos(-gmmComponents[0].angle)
          const s0 = Math.sin(-gmmComponents[0].angle)
          const rotX0 = dx0 * c0 - dy0 * s0
          const rotY0 = dx0 * s0 + dy0 * c0
          const dist0 = (rotX0 / gmmComponents[0].rx) ** 2 + (rotY0 / gmmComponents[0].ry) ** 2
          const prob0 = Math.exp(-0.5 * dist0)

          const dx1 = x - gmmComponents[1].meanX
          const dy1 = y - gmmComponents[1].meanY
          const c1 = Math.cos(-gmmComponents[1].angle)
          const s1 = Math.sin(-gmmComponents[1].angle)
          const rotX1 = dx1 * c1 - dy1 * s1
          const rotY1 = dx1 * s1 + dy1 * c1
          const dist1 = (rotX1 / gmmComponents[1].rx) ** 2 + (rotY1 / gmmComponents[1].ry) ** 2
          const prob1 = Math.exp(-0.5 * dist1)

          const softRatio = (prob0 + 1e-6) / (prob0 + prob1 + 2e-6)
          ctx.fillStyle = dark
            ? `rgba(${Math.round(99 * softRatio + 16 * (1 - softRatio))}, ${Math.round(102 * softRatio + 185 * (1 - softRatio))}, ${Math.round(241 * softRatio + 129 * (1 - softRatio))}, 0.15)`
            : `rgba(${Math.round(99 * softRatio + 16 * (1 - softRatio))}, ${Math.round(102 * softRatio + 185 * (1 - softRatio))}, ${Math.round(241 * softRatio + 129 * (1 - softRatio))}, 0.12)`
        }
        ctx.fillRect(x, y, step, step)
      }
    }

    // Draw GMM covariance ellipses if active
    if (modelType === 'gmm') {
      gmmComponents.forEach(comp => {
        [1, 2].forEach(sigmaMultiplier => {
          ctx.save()
          ctx.translate(comp.meanX, comp.meanY)
          ctx.rotate(comp.angle)
          ctx.beginPath()
          ctx.ellipse(0, 0, comp.rx * sigmaMultiplier, comp.ry * sigmaMultiplier, 0, 0, Math.PI * 2)
          ctx.strokeStyle = comp.color
          ctx.lineWidth = sigmaMultiplier === 1 ? 2 : 1
          if (sigmaMultiplier === 2) ctx.setLineDash([4, 4])
          ctx.stroke()
          ctx.restore()
        })
      })
    } else {
      // Draw K-Means rigid circular boundary
      ctx.beginPath()
      ctx.moveTo(W / 2 + 10, 0)
      ctx.lineTo(W / 2 - 40, H)
      ctx.strokeStyle = dark ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.4)'
      ctx.lineWidth = 2
      ctx.setLineDash([4, 4])
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Render Data Points
    dataset.forEach(p => {
      ctx.beginPath()
      ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2)
      ctx.fillStyle = p.trueCluster === 0 ? '#6366F1' : '#10B981'
      ctx.fill()
      ctx.strokeStyle = dark ? '#0F172A' : '#FFFFFF'
      ctx.lineWidth = 1.5
      ctx.stroke()
    })

    // Legend
    ctx.font = '10px monospace'
    ctx.fillStyle = '#6366F1'
    ctx.fillText('Cluster A (Diagonal)', 12, 20)
    ctx.fillStyle = '#10B981'
    ctx.fillText('Cluster B (Horizontal)', 12, 34)
    ctx.fillStyle = dark ? '#94A3B8' : '#64748B'
    ctx.fillText(
      modelType === 'kmeans'
        ? 'K-Means: Rigid spherical Voronoi cut (Misallocates tail points!)'
        : 'GMM: Elliptical full covariance contours with soft responsibilities γ_ik',
      12,
      48
    )
  }, [dataset, modelType, theme, W, H])

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <label className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
          Clustering Paradigm:
        </label>
        <div className="flex gap-2">
          <button
            onClick={() => setModelType('kmeans')}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              modelType === 'kmeans'
                ? 'bg-amber-500 text-white shadow-sm'
                : theme === 'dark' ? 'bg-slate-800 text-gray-400' : 'bg-gray-100 text-gray-700'
            }`}
          >
            K-Means (Hard &amp; Spherical)
          </button>
          <button
            onClick={() => setModelType('gmm')}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              modelType === 'gmm'
                ? 'bg-indigo-500 text-white shadow-sm'
                : theme === 'dark' ? 'bg-slate-800 text-gray-400' : 'bg-gray-100 text-gray-700'
            }`}
          >
            GMM with EM (Soft &amp; Elliptical)
          </button>
        </div>
      </div>

      <div className={`rounded-2xl border overflow-hidden relative mb-2 ${
        theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
      }`}>
        <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
      </div>

      <p className={`text-xs text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        Notice how K-Means awkwardly slices elongated clusters in half with a straight Voronoi plane, while GMM naturally contours around their elliptical shapes via covariance matrices Σ_k!
      </p>
    </div>
  )
}

// ── Main Page Component ───────────────────────────────────────────────────────
export default function Clustering() {
  const { theme } = useApp()

  const S = `rounded-2xl border p-6 mb-6 ${theme === 'dark' ? 'bg-slate-900/80 border-white/10' : 'bg-white border-gray-200'}`
  const LBL = `text-xs font-semibold uppercase tracking-wider mb-3 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`
  const H2 = `text-xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`
  const BODY = `text-sm leading-relaxed ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-4xl mx-auto">
      {/* Hero Header */}
      <div className="mb-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-purple-500/20 border border-purple-500/30 mb-4"
        >
          <span className="text-xs text-purple-400 font-medium">Machine Learning • Unsupervised Discovery &amp; Density Estimation</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Unsupervised Learning <span className="gradient-text">(K-Means &amp; GMM)</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Without supervisor labels or reward signals, unsupervised learning uncovers the <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>latent geometric structure and probability distributions</strong> hidden inside raw data, powering customer segmentation, anomaly detection, and automated feature discovery.
        </p>

        <Callout type="analogy" title="Analogy: The Alien Archaeologist">
          Imagine an alien discovering an ancient human library with no translation key. They cannot read the titles, but by measuring book dimensions, binding styles, and word frequencies, they cluster texts into poetry, encyclopedias, and novels! That is <strong>Unsupervised Clustering</strong>: discovering organic groupings purely from feature geometry.
        </Callout>
      </div>

      {/* Core Concepts */}
      <TheoryBlock
        title="Unsupervised Clustering Pillars"
        cards={[
          {
            icon: '🎯',
            title: 'K-Means Algorithm',
            body: 'Iterates between Assignment (mapping points to nearest centroid) and Update (recalculating centroids as cluster means). Minimizes inertia J.',
            mono: 'J = Σ Σ ||x_i - μ_k||²'
          },
          {
            icon: '✨',
            title: 'K-Means++',
            body: 'Arthur & Vassilvitskii\'s smart initialization: chooses initial seeds with probability proportional to D(x)², preventing catastrophic local minima.',
            mono: 'P(x) ∝ D(x)²'
          },
          {
            icon: '📉',
            title: 'Choosing k',
            body: 'Elbow Method identifies diminishing returns in inertia reduction; Silhouette Score computes cohesion vs separation s = (b - a) / max(a, b).',
            mono: 's ∈ [-1, +1]'
          },
          {
            icon: '🫧',
            title: 'Gaussian Mixture (GMM)',
            body: 'Generative model representing data as a weighted mixture of K normal densities with arbitrary elliptical covariance shapes Σ_k.',
            mono: 'p(x) = Σ π_k N(x|μ_k, Σ_k)'
          },
          {
            icon: '🔄',
            title: 'EM Algorithm',
            body: 'Expectation-Maximization: E-step calculates soft responsibilities γ_ik; M-step updates weights π_k, means μ_k, and covariances Σ_k.',
            mono: 'E-step ↔ M-step'
          },
          {
            icon: '🚨',
            title: 'Anomaly Detection',
            body: 'Evaluates the continuous density function p(x); instances falling in the extreme low-probability tails are flagged as high-risk anomalies.',
            mono: 'Flag if p(x) < ε'
          }
        ]}
      />

      {/* ========================================================================= */}
      {/* SECTION 1: K-MEANS CLUSTERING */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>
          Part 1 — Geometric Partitioning
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          K-Means Clustering &amp; WCSS Optimization
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Lloyd&apos;s alternating optimization, Within-Cluster Sum of Squares, and smart initialization.
        </p>
      </div>

      <div className={S}>
        <p className={LBL}>1.1 — The Assignment &amp; Update Loop</p>
        <h2 className={H2}>Minimizing Within-Cluster Sum of Squares (Inertia)</h2>
        <p className={`${BODY} mb-4`}>
          Given <code className="font-mono">N</code> unlabeled points, K-Means seeks to partition them into <code className="font-mono">K</code> non-overlapping clusters <code className="font-mono">C₁, ..., C_K</code> to minimize total squared Euclidean distortion:
        </p>

        <Callout type="formula" mono="J = Σ_{k=1}^K Σ_{x_i ∈ C_k} ||x_i - μ_k||²">
          Lloyd&apos;s algorithm solves this non-convex problem by alternating two steps until centroid movement drops below tolerance:
        </Callout>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <span className="text-indigo-400 font-bold text-xs uppercase block mb-1">Step 1: Assignment</span>
            <p className={`text-xs ${BODY} mb-2`}>
              Hold centroids <code className="font-mono">μ_k</code> fixed. Assign each point <code className="font-mono">x_i</code> to its nearest centroid, establishing Voronoi polygonal boundaries.
            </p>
            <div className={`p-2 rounded font-mono text-[11px] ${theme === 'dark' ? 'bg-slate-900 text-indigo-300' : 'bg-white text-indigo-800'}`}>
              c_i = argmin_k ||x_i - μ_k||²
            </div>
          </div>

          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <span className="text-emerald-400 font-bold text-xs uppercase block mb-1">Step 2: Update</span>
            <p className={`text-xs ${BODY} mb-2`}>
              Hold cluster assignments fixed. Recompute each centroid <code className="font-mono">μ_k</code> as the geometric mean (center of mass) of all assigned points.
            </p>
            <div className={`p-2 rounded font-mono text-[11px] ${theme === 'dark' ? 'bg-slate-900 text-emerald-300' : 'bg-white text-emerald-800'}`}>
              {"μ_k = (1 / |C_k|) Σ_{x_i ∈ C_k} x_i"}
            </div>
          </div>
        </div>

        <p className={`${BODY} mb-4`}>
          Interactive Simulator: Alternate between the Assignment Step and Update Step to observe Voronoi partitions form and centroids migrate toward stability:
        </p>

        <KMeansSimulatorCanvas theme={theme} />
      </div>

      <div className={S}>
        <p className={LBL}>1.2 — Choosing k: Elbow Method &amp; Silhouette Score</p>
        <h2 className={H2}>Evaluating cluster validity without ground truth labels</h2>
        <p className={`${BODY} mb-4`}>
          Because K-Means cannot evaluate test accuracy, selecting the optimal number of clusters <code className="font-mono">k</code> requires unsupervised heuristic metrics:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <h3 className="font-semibold text-xs text-amber-400 mb-1">The Elbow Method</h3>
            <p className={`text-xs ${BODY}`}>
              Plots Inertia (WCSS) against <code className="font-mono">k</code>. As <code className="font-mono">k</code> increases, inertia drops monotonically. The optimal <code className="font-mono">k</code> is the &ldquo;elbow&rdquo; inflection point where marginal gains plateau.
            </p>
          </div>

          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <h3 className="font-semibold text-xs text-indigo-400 mb-1">Silhouette Coefficient</h3>
            <p className={`text-xs ${BODY}`}>
              Quantifies both cohesion within cluster <code className="font-mono">a(i)</code> and separation from neighboring clusters <code className="font-mono">b(i)</code>:
              <br />
              <code className="font-mono font-bold text-indigo-300">s(i) = (b(i) - a(i)) / max(a(i), b(i))</code>.
              <br />
              Peaking near +1 indicates dense, well-separated clusters.
            </p>
          </div>
        </div>

        <ElbowSilhouetteAnalyzer theme={theme} />
      </div>

      {/* ========================================================================= */}
      {/* SECTION 2: GMM AND EM */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-purple-500/5 border-purple-500/20' : 'bg-purple-50 border-purple-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-purple-400' : 'text-purple-600'}`}>
          Part 2 — Density Estimation &amp; Soft Clustering
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Gaussian Mixture Models (GMM) &amp; EM
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Overcoming spherical limitations via soft assignments and full covariance ellipsoids.
        </p>
      </div>

      <div className={S}>
        <p className={LBL}>2.1 — Gaussian Mixtures &amp; The EM Algorithm</p>
        <h2 className={H2}>A weighted superposition of K Gaussian bells</h2>
        <p className={`${BODY} mb-4`}>
          A <strong>Gaussian Mixture Model</strong> treats data generation as a two-stage probabilistic process: first choose component <code className="font-mono">k</code> with prior probability <code className="font-mono">π_k</code>, then draw <code className="font-mono">x</code> from that component&apos;s Gaussian distribution <code className="font-mono">N(μ_k, Σ_k)</code>:
        </p>

        <Callout type="formula" mono="p(x) = Σ_{k=1}^K π_k · N(x | μ_k, Σ_k)   where 0 ≤ π_k ≤ 1  and  Σ π_k = 1">
          Because cluster assignments <code className="font-mono">Z_i</code> are unobserved latent variables, maximum likelihood cannot be solved in closed form. We use the <strong>Expectation-Maximization (EM)</strong> algorithm:
        </Callout>

        <TheoryBlock
          title="Expectation-Maximization (EM) Cycle"
          cards={[
            {
              icon: '1️⃣',
              title: 'E-Step (Expectation)',
              body: 'Compute the soft posterior probability (responsibility γ_ik) that component k generated point x_i using Bayes\' rule.',
              mono: 'γ_ik = (π_k N(x_i|μ_k, Σ_k)) / p(x_i)'
            },
            {
              icon: '2️⃣',
              title: 'Effective Population N_k',
              body: 'Sum the soft weights assigned to component k across the entire dataset: N_k = Σ_{i=1}^N γ_ik.',
              mono: 'N_k = sum(γ_ik)'
            },
            {
              icon: '3️⃣',
              title: 'M-Step: Update Mean & Cov',
              body: 'Re-estimate μ_k as the responsibility-weighted average of data points, and recompute covariance matrix Σ_k.',
              mono: 'μ_k = (1/N_k) Σ γ_ik x_i'
            },
            {
              icon: '4️⃣',
              title: 'M-Step: Update Mixing Prior',
              body: 'Update component prior weight π_k as the fraction of the total dataset population explained by cluster k.',
              mono: 'π_k = N_k / N'
            }
          ]}
        />

        <p className={`${BODY} mb-4 mt-6`}>
          Interactive Comparison: Compare K-Means (rigid spherical Voronoi partitions) vs GMM (flexible elliptical covariance contours and soft responsibilities):
        </p>

        <GmmVsKMeansCanvas theme={theme} />

        <DeepDive title="Real-World Clustering Applications">
          <ul className={`list-disc list-inside space-y-2 text-xs ${BODY}`}>
            <li>
              <strong>Customer Segmentation (RFM):</strong> Grouping e-commerce users by Recency, Frequency, and Monetary value into VIPs, churn risks, and bargain hunters.
            </li>
            <li>
              <strong>Anomaly Detection:</strong> Computing <code className="font-mono">log p(x)</code> with GMM. Samples falling below a low-density threshold (e.g. 1st percentile) reveal credit card fraud or industrial sensor failures.
            </li>
            <li>
              <strong>Image Color Quantization:</strong> Compressing RGB color spaces by clustering millions of pixels into <code className="font-mono">k=16</code> palette centroids.
            </li>
          </ul>
        </DeepDive>
      </div>

      {/* Python Code Section */}
      <div className={S}>
        <p className={LBL}>Python Scikit-Learn Implementations</p>
        <h2 className={H2}>Production workflows with scikit-learn</h2>
        <p className={`${BODY} mb-3`}>
          1. <strong>K-Means++ &amp; Mini-Batch K-Means with Silhouette Evaluation:</strong>
        </p>
        <CodeBlock code={PYTHON_CODE_KMEANS} />

        <p className={`${BODY} mt-6 mb-3`}>
          2. <strong>GMM with Full Covariance, Soft Probabilities &amp; Outlier Detection:</strong>
        </p>
        <CodeBlock code={PYTHON_CODE_GMM_EM} />
      </div>

      {/* Quiz Section */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${
          theme === 'dark' ? 'bg-purple-500/10 border-purple-500/20' : 'bg-purple-50 border-purple-200'
        }`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
              10 questions covering K-Means, K-Means++, Elbow Method, Silhouette score, GMM, and EM • +100 XP
            </p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="clustering" />
      </div>
    </motion.div>
  )
}
