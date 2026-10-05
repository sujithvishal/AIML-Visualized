import { useState, useRef, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

// ── Python Code Examples ────────────────────────────────────────────────────────
const PYTHON_CODE_HARD_SOFT = `from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np

# Generate linearly separable 2D data
X, y = make_blobs(n_samples=60, centers=2, random_state=42, cluster_std=1.2)
# Convert labels from {0, 1} to {-1, +1}
y = np.where(y == 0, -1, 1)

# 1. Hard-Margin SVM (large C acts as strict margin constraint)
hard_svm = SVC(kernel='linear', C=1e5)
hard_svm.fit(X, y)
print(f"Hard-margin support vectors count: {len(hard_svm.support_)}")
print(f"Weight vector w: {hard_svm.coef_[0]}")
print(f"Bias b: {hard_svm.intercept_[0]:.4f}")
print(f"Margin width (2 / ||w||): {2 / np.linalg.norm(hard_svm.coef_[0]):.4f}")

# 2. Soft-Margin SVM (moderate C tolerates outliers & broadens margin)
soft_svm = SVC(kernel='linear', C=1.0)
soft_svm.fit(X, y)
print(f"Soft-margin support vectors count: {len(soft_svm.support_)}")
print(f"Soft-margin margin width: {2 / np.linalg.norm(soft_svm.coef_[0]):.4f}")`

const PYTHON_CODE_KERNELS = `import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# Concentric non-linear circles
X, y = make_circles(n_samples=300, noise=0.08, factor=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Compare Linear vs Polynomial vs RBF kernels
models = {
    'Linear': SVC(kernel='linear', C=1.0),
    'Polynomial (deg 3)': SVC(kernel='poly', degree=3, C=1.0),
    'RBF (Gaussian)': SVC(kernel='rbf', C=1.0, gamma='scale'),
    'Sigmoid': SVC(kernel='sigmoid', C=1.0)
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"{name:20s} Test Accuracy: {acc:.2%}")

# Hyperparameter GridSearch for optimal C and gamma in RBF SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10]
}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"\\nBest RBF Parameters: {grid.best_params_}")
print(f"Best CV Accuracy: {grid.best_score_:.2%}")`

const PYTHON_CODE_TEXT_IMAGE = `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Text Classification: TF-IDF + Linear SVM (Gold standard for sparse high-d text)
corpus = [
    ("Congratulations! You won a $1,000 cash prize. Click here now.", 1),
    ("Claim your free gift card and lottery winnings immediately.", 1),
    ("Exclusive loan offer approved with 0% interest rate today.", 1),
    ("Hey team, attached is the revised agenda for tomorrow's sprint demo.", 0),
    ("Can we reschedule our lunch meeting to Thursday afternoon?", 0),
    ("Quarterly financial report is ready for executive review.", 0),
]
texts, labels = zip(*corpus)

# LinearSVC optimizes coordinate descent directly on primal — blazing fast
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('svm', LinearSVC(C=1.0, dual=False, random_state=42))
])

text_pipeline.fit(texts, labels)
sample_email = ["Urgent: Claim your free lottery prize money now!"]
pred = text_pipeline.predict(sample_email)[0]
print("Spam Detector Prediction:", "SPAM" if pred == 1 else "HAM")

# Inspect top informative features (highest positive SVM weights -> spam indicators)
vectorizer = text_pipeline.named_steps['tfidf']
svm_model = text_pipeline.named_steps['svm']
feature_names = np.array(vectorizer.get_feature_names_out())
top_spam_words = feature_names[np.argsort(svm_model.coef_[0])[-3:]]
print("Top spam keywords by SVM weight:", top_spam_words)`

const QUIZ_QUESTIONS = [
  {
    question: 'What is the primary objective of a hard-margin SVM classifier?',
    options: [
      'Minimize training iterations',
      'Maximize the perpendicular distance (margin) between the separating hyperplane and closest points',
      'Maximize the number of support vectors',
      'Minimize cross-entropy probability loss'
    ],
    correct: 1,
    explanation: 'Hard-margin SVM finds the unique hyperplane that separates two classes while maximizing the geometric margin (2 / ||w||), ensuring optimal generalization.'
  },
  {
    question: 'The total margin width of an SVM in terms of the weight vector w is:',
    options: [
      '||w|| / 2',
      '2 / ||w||',
      '1 / ||w||²',
      '2 · ||w||'
    ],
    correct: 1,
    explanation: 'The perpendicular distance from the decision hyperplane (wᵀx + b = 0) to either margin boundary (wᵀx + b = ±1) is 1/||w||, so the total separation gap is 2/||w||.'
  },
  {
    question: 'What are Support Vectors in an SVM?',
    options: [
      'All data points in the training dataset',
      'Only the points that lie precisely on or violate the margin boundaries',
      'The cluster centroids calculated using k-means',
      'The normal vector perpendicular to the decision boundary'
    ],
    correct: 1,
    explanation: 'Support vectors are the critical subset of training points that constrain the boundary (where αᵢ > 0). Removing all other training points leaves the decision boundary completely unchanged!'
  },
  {
    question: 'Why is the Dual formulation of SVM superior to the Primal formulation when applying kernels?',
    options: [
      'The dual avoids matrix multiplication',
      'The dual depends solely on pairwise dot products (xᵢ · xⱼ) between samples',
      'The dual eliminates the bias parameter b',
      'The dual problem has zero constraints'
    ],
    correct: 1,
    explanation: 'In the Wolfe dual, features only appear as inner products (xᵢ · xⱼ). This enables the Kernel Trick: replacing xᵢ · xⱼ with K(xᵢ, xⱼ) without ever computing high-dimensional coordinates explicitly.'
  },
  {
    question: 'According to KKT complementary slackness, for any point xᵢ with Lagrange multiplier αᵢ:',
    options: [
      'αᵢ must equal 1 for all points',
      'αᵢ · [yᵢ(wᵀxᵢ + b) − 1] = 0',
      'αᵢ is always negative',
      'w · xᵢ must equal 0'
    ],
    correct: 1,
    explanation: 'Complementary slackness enforces αᵢ · [yᵢ(wᵀxᵢ + b) − 1] = 0. For points strictly outside the margin (yᵢ(wᵀxᵢ + b) > 1), αᵢ must be 0. Thus, only support vectors have αᵢ > 0.'
  },
  {
    question: 'What is the role of the slack variable ξᵢ in a soft-margin SVM?',
    options: [
      'It scales the learning rate during gradient descent',
      'It measures the degree of margin violation or misclassification for point i',
      'It acts as the kernel bandwidth',
      'It randomly drops features to prevent co-adaptation'
    ],
    correct: 1,
    explanation: 'Slack variable ξᵢ ≥ 0 quantifies margin violations: ξᵢ = 0 means on or outside the margin; 0 < ξᵢ ≤ 1 means inside the margin but correctly classified; ξᵢ > 1 means misclassified.'
  },
  {
    question: 'How does increasing the regularization parameter C affect a soft-margin SVM?',
    options: [
      'It widens the margin and tolerates more violations (more regularization)',
      'It heavily penalizes slack violations, creating a narrower margin that fits training data more strictly (risk of overfitting)',
      'It transforms the linear kernel into an RBF kernel',
      'It has no effect on the decision boundary'
    ],
    correct: 1,
    explanation: 'C is the penalty coefficient for misclassifications and margin violations. Large C demands fewer violations (narrow margin, lower bias, higher variance); small C tolerates violations (wider margin, higher bias, lower variance).'
  },
  {
    question: 'What constitutes the "Kernel Trick"?',
    options: [
      'Using deep neural layers to generate synthetic training data',
      'Computing inner products in high-dimensional feature spaces implicitly via a kernel function K(x, z) without explicit transformation',
      'Replacing quadratic programming with random forest trees',
      'Normalizing all input vectors to unit Euclidean norm'
    ],
    correct: 1,
    explanation: 'The kernel trick evaluates K(x, z) = ⟨ϕ(x), ϕ(z)⟩ directly in input space, bypassing the astronomical (or infinite) computational cost of computing the feature map ϕ(x).'
  },
  {
    question: 'Mercer\'s Theorem states that a function K(x, z) is a valid kernel if and only if:',
    options: [
      'The function output is always between 0 and 1',
      'The corresponding Gram kernel matrix is symmetric and positive semi-definite (PSD)',
      'The kernel function is monotonically decreasing',
      'The gradient of the kernel function is linear everywhere'
    ],
    correct: 1,
    explanation: 'Mercer\'s condition guarantees that for any dataset, the Gram matrix K with entries Kᵢⱼ = K(xᵢ, xⱼ) is symmetric and positive semi-definite (all eigenvalues ≥ 0), ensuring a valid Hilbert inner-product space.'
  },
  {
    question: 'Why is a Linear SVM often preferred over an RBF kernel for high-dimensional text classification (e.g. TF-IDF)?',
    options: [
      'RBF kernels cannot handle negative numbers',
      'High-dimensional text spaces (d > 20,000) are almost always already linearly separable, and linear SVMs are much faster and avoid overfitting',
      'Linear SVMs do not require tokenization',
      'Text data violates Mercer\'s theorem'
    ],
    correct: 1,
    explanation: 'In bag-of-words / TF-IDF representations, the feature dimension d is typically huge (10⁴ - 10⁵) and sparse. The data is usually already linearly separable, so linear SVM trains in O(n·d) time without the risk of over-fitting associated with RBF.'
  }
]

// ── Interactive Component 1: Hard-Margin Maximum Margin Visualizer ───────────
function HardMarginCanvas({ theme }) {
  const canvasRef = useRef(null)
  const [activePreset, setActivePreset] = useState('standard')
  const [hideNonSVs, setHideNonSVs] = useState(false)
  const W = 460
  const H = 300

  // Presets of linearly separable 2D data
  const datasets = useMemo(() => ({
    standard: [
      { id: 1, x: 90,  y: 80,  label: 1 },
      { id: 2, x: 130, y: 55,  label: 1 },
      { id: 3, x: 75,  y: 130, label: 1 },
      { id: 4, x: 150, y: 110, label: 1 }, // Support vector
      { id: 5, x: 110, y: 160, label: 1 }, // Support vector
      { id: 6, x: 340, y: 230, label: -1 },
      { id: 7, x: 380, y: 190, label: -1 },
      { id: 8, x: 300, y: 250, label: -1 },
      { id: 9, x: 255, y: 180, label: -1 }, // Support vector
      { id: 10,x: 290, y: 140, label: -1 }
    ],
    narrow: [
      { id: 1, x: 100, y: 70,  label: 1 },
      { id: 2, x: 140, y: 100, label: 1 },
      { id: 3, x: 175, y: 140, label: 1 }, // Support vector
      { id: 4, x: 80,  y: 160, label: 1 },
      { id: 5, x: 215, y: 160, label: -1 }, // Support vector
      { id: 6, x: 260, y: 200, label: -1 },
      { id: 7, x: 310, y: 180, label: -1 },
      { id: 8, x: 350, y: 240, label: -1 }
    ],
    wide: [
      { id: 1, x: 70,  y: 80,  label: 1 },
      { id: 2, x: 100, y: 50,  label: 1 },
      { id: 3, x: 120, y: 120, label: 1 }, // SV
      { id: 4, x: 320, y: 190, label: -1 }, // SV
      { id: 5, x: 370, y: 220, label: -1 },
      { id: 6, x: 390, y: 170, label: -1 }
    ]
  }), [])

  const points = datasets[activePreset] || datasets.standard

  // Simple, accurate linear SVM solver for 2D points using iterative projected subgradient
  const model = useMemo(() => {
    // Normalize coordinates to [-2, 2]
    const normPts = points.map(p => ({
      x: (p.x - W / 2) / 100,
      y: (p.y - H / 2) / 100,
      label: p.label,
      orig: p
    }))

    // Projected gradient descent on dual or primal
    let w1 = 0.5, w2 = -0.5, b = 0
    const lr = 0.01
    const epochs = 1200
    const C = 500 // simulate hard margin

    for (let ep = 0; ep < epochs; ep++) {
      let gradW1 = w1
      let gradW2 = w2
      let gradB = 0

      for (const p of normPts) {
        const score = p.label * (w1 * p.x + w2 * p.y + b)
        if (score < 1) {
          gradW1 -= C * p.label * p.x
          gradW2 -= C * p.label * p.y
          gradB -= C * p.label
        }
      }

      w1 -= lr * (gradW1 / normPts.length)
      w2 -= lr * (gradW2 / normPts.length)
      b -= lr * (gradB / normPts.length)
    }

    const normW = Math.sqrt(w1 * w1 + w2 * w2) || 1
    const marginWidthPx = (2 / normW) * 100

    // Identify support vectors: points where y(w^T x + b) ≈ 1
    const svThreshold = 0.35
    const supportVectors = normPts.filter(p => {
      const score = p.label * (w1 * p.x + w2 * p.y + b)
      return Math.abs(score - 1) < svThreshold || score < 1
    }).map(p => p.orig.id)

    return { w1, w2, b, normW, marginWidthPx, supportVectors }
  }, [points, W, H])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'
    const { w1, w2, b } = model

    // Helper to evaluate raw functional margin at canvas pixel (cx, cy)
    const evalAt = (cx, cy) => {
      const nx = (cx - W / 2) / 100
      const ny = (cy - H / 2) / 100
      return w1 * nx + w2 * ny + b
    }

    // Draw Margin Strip Region (where -1 <= score <= 1)
    const step = 4
    for (let x = 0; x < W; x += step) {
      for (let y = 0; y < H; y += step) {
        const val = evalAt(x, y)
        if (Math.abs(val) <= 1.0) {
          ctx.fillStyle = dark
            ? `rgba(99, 102, 241, ${0.15 * (1 - Math.abs(val))})`
            : `rgba(99, 102, 241, ${0.12 * (1 - Math.abs(val))})`
          ctx.fillRect(x, y, step, step)
        }
      }
    }

    // Helper to draw contour line where evalAt(x, y) == target
    const drawContour = (target, strokeStyle, lineWidth, dashed = false) => {
      ctx.beginPath()
      ctx.strokeStyle = strokeStyle
      ctx.lineWidth = lineWidth
      if (dashed) ctx.setLineDash([5, 4])
      else ctx.setLineDash([])

      // Boundary line: w1 * ((x - W/2)/100) + w2 * ((y - H/2)/100) + b = target
      // y_norm = (target - b - w1 * x_norm) / w2
      if (Math.abs(w2) > 0.001) {
        const yAtX0 = ((target - b - w1 * (-W / 2 / 100)) / w2) * 100 + H / 2
        const yAtXW = ((target - b - w1 * ((W - W / 2) / 100)) / w2) * 100 + H / 2
        ctx.moveTo(0, yAtX0)
        ctx.lineTo(W, yAtXW)
      } else {
        const xAtTarget = ((target - b) / w1) * 100 + W / 2
        ctx.moveTo(xAtTarget, 0)
        ctx.lineTo(xAtTarget, H)
      }
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Positive margin boundary: wᵀx + b = +1
    drawContour(1, dark ? 'rgba(99, 102, 241, 0.7)' : 'rgba(79, 70, 229, 0.8)', 2, true)
    // Negative margin boundary: wᵀx + b = -1
    drawContour(-1, dark ? 'rgba(16, 185, 129, 0.7)' : 'rgba(5, 150, 105, 0.8)', 2, true)
    // Decision hyperplane: wᵀx + b = 0
    drawContour(0, dark ? '#F59E0B' : '#D97706', 2.5, false)

    // Draw normal vector w arrow starting from center of boundary
    const cx = W / 2
    const cy = H / 2
    const centerVal = evalAt(cx, cy)
    const normLen = Math.sqrt(w1 * w1 + w2 * w2) || 1
    const unitW1 = w1 / normLen
    const unitW2 = w2 / normLen
    // Anchor on line
    const anchorX = cx - unitW1 * (centerVal / normLen) * 100
    const anchorY = cy - unitW2 * (centerVal / normLen) * 100
    const arrowLen = 38

    ctx.beginPath()
    ctx.moveTo(anchorX, anchorY)
    ctx.lineTo(anchorX + unitW1 * arrowLen, anchorY + unitW2 * arrowLen)
    ctx.strokeStyle = '#F59E0B'
    ctx.lineWidth = 2
    ctx.stroke()

    // Arrowhead
    const angle = Math.atan2(unitW2, unitW1)
    ctx.beginPath()
    ctx.moveTo(anchorX + unitW1 * arrowLen, anchorY + unitW2 * arrowLen)
    ctx.lineTo(
      anchorX + unitW1 * arrowLen - 7 * Math.cos(angle - Math.PI / 6),
      anchorY + unitW2 * arrowLen - 7 * Math.sin(angle - Math.PI / 6)
    )
    ctx.lineTo(
      anchorX + unitW1 * arrowLen - 7 * Math.cos(angle + Math.PI / 6),
      anchorY + unitW2 * arrowLen - 7 * Math.sin(angle + Math.PI / 6)
    )
    ctx.fillStyle = '#F59E0B'
    ctx.fill()

    ctx.font = 'bold 11px sans-serif'
    ctx.fillStyle = '#F59E0B'
    ctx.fillText('w (normal)', anchorX + unitW1 * (arrowLen + 10), anchorY + unitW2 * (arrowLen + 10))

    // Draw Training Points
    points.forEach(p => {
      const isSV = model.supportVectors.includes(p.id)
      if (hideNonSVs && !isSV) {
        // Render ghostly transparent
        ctx.beginPath()
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2)
        ctx.fillStyle = dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)'
        ctx.fill()
        return
      }

      // Draw Support Vector halo if it is an SV
      if (isSV) {
        ctx.beginPath()
        ctx.arc(p.x, p.y, 14, 0, Math.PI * 2)
        ctx.strokeStyle = '#F59E0B'
        ctx.lineWidth = 2.5
        ctx.stroke()

        // Outer pulsing ring
        ctx.beginPath()
        ctx.arc(p.x, p.y, 18, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)'
        ctx.lineWidth = 1
        ctx.setLineDash([2, 3])
        ctx.stroke()
        ctx.setLineDash([])
      }

      // Main Point
      ctx.beginPath()
      ctx.arc(p.x, p.y, isSV ? 8 : 6.5, 0, Math.PI * 2)
      ctx.fillStyle = p.label === 1 ? '#6366F1' : '#10B981'
      ctx.fill()
      ctx.strokeStyle = dark ? '#0F172A' : '#FFFFFF'
      ctx.lineWidth = 2
      ctx.stroke()

      // Class label inside point
      ctx.font = 'bold 9px monospace'
      ctx.fillStyle = '#FFFFFF'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(p.label === 1 ? '+' : '−', p.x, p.y)
    })

    // Labels for boundary lines
    ctx.font = '10px monospace'
    ctx.fillStyle = dark ? '#A5B4FC' : '#4338CA'
    ctx.textAlign = 'left'
    ctx.fillText('wᵀx + b = +1 (Margin +)', 12, 22)

    ctx.fillStyle = '#F59E0B'
    ctx.fillText('wᵀx + b = 0 (Decision Hyperplane)', 12, 38)

    ctx.fillStyle = dark ? '#6EE7B7' : '#047857'
    ctx.fillText('wᵀx + b = −1 (Margin −)', 12, 54)
  }, [points, model, hideNonSVs, theme, W, H])

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Dataset:</span>
          {['standard', 'narrow', 'wide'].map(p => (
            <button
              key={p}
              onClick={() => setActivePreset(p)}
              className={`px-3 py-1 rounded-lg text-xs font-medium capitalize transition-all ${
                activePreset === p
                  ? 'bg-indigo-500 text-white shadow-sm'
                  : theme === 'dark'
                  ? 'bg-slate-800 text-gray-400 hover:text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {p} Margin
            </button>
          ))}
        </div>

        <button
          onClick={() => setHideNonSVs(v => !v)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${
            hideNonSVs
              ? 'bg-amber-500/20 border-amber-500/40 text-amber-300'
              : theme === 'dark'
              ? 'bg-slate-800/80 border-white/10 text-gray-300 hover:border-amber-400/40'
              : 'bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100'
          }`}
        >
          <span>{hideNonSVs ? '👁️ Restore All Points' : '🔬 Test: Remove Non-Support Vectors'}</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div className={`col-span-2 rounded-2xl border overflow-hidden relative ${
          theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
        }`}>
          <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
          {hideNonSVs && (
            <div className="absolute bottom-3 left-3 right-3 bg-amber-500/15 border border-amber-500/30 rounded-xl p-2.5 backdrop-blur-md">
              <p className="text-xs text-amber-300 font-medium">
                💡 <strong>The Invariance Proof:</strong> Notice the boundary and margin did not change! Only the highlighted Support Vectors with rings determine the optimal hyperplane.
              </p>
            </div>
          )}
        </div>

        {/* Real-time Math Metrics Panel */}
        <div className={`rounded-2xl border p-4 flex flex-col justify-between ${
          theme === 'dark' ? 'bg-slate-900/90 border-white/10' : 'bg-white border-gray-200'
        }`}>
          <div>
            <h4 className={`text-xs font-bold uppercase tracking-wider mb-3 ${
              theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'
            }`}>
              Geometry & Metrics
            </h4>
            <div className="space-y-3">
              <div className={`p-2.5 rounded-xl border ${
                theme === 'dark' ? 'bg-slate-800/50 border-white/5' : 'bg-gray-50 border-gray-200'
              }`}>
                <div className="flex justify-between items-center text-xs mb-1">
                  <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Margin Width (2/||w||)</span>
                  <span className="font-mono font-bold text-amber-400">
                    {(model.marginWidthPx / 40).toFixed(2)} units
                  </span>
                </div>
                <div className={`h-2 rounded-full overflow-hidden ${theme === 'dark' ? 'bg-slate-700' : 'bg-gray-200'}`}>
                  <motion.div
                    className="h-full bg-amber-400 rounded-full"
                    animate={{ width: `${Math.min(100, (model.marginWidthPx / 150) * 100)}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              </div>

              <div className="flex justify-between items-center text-xs py-1 border-b border-inherit">
                <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Support Vectors Count</span>
                <span className="font-mono font-bold text-indigo-400">{model.supportVectors.length} of {points.length}</span>
              </div>

              <div className="flex justify-between items-center text-xs py-1 border-b border-inherit">
                <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Weight Norm ||w||</span>
                <span className="font-mono text-xs text-purple-400">{model.normW.toFixed(3)}</span>
              </div>

              <div className="flex justify-between items-center text-xs py-1">
                <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Status</span>
                <span className="inline-flex items-center gap-1 text-emerald-400 font-medium">
                  <span>✓</span> Strictly Separable
                </span>
              </div>
            </div>
          </div>

          <div className={`mt-4 p-2.5 rounded-xl text-xs font-mono ${
            theme === 'dark' ? 'bg-indigo-500/10 text-indigo-300 border border-indigo-500/20' : 'bg-indigo-50 text-indigo-700 border border-indigo-200'
          }`}>
            wᵀx + b = 0<br />
            γ = 2 / ||w||
          </div>
        </div>
      </div>
      <p className={`text-xs text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        Interactive Hard-Margin SVM • Points touching the dashed lines are active Support Vectors (αᵢ &gt; 0).
      </p>
    </div>
  )
}

// ── Interactive Component 2: Soft-Margin & Slack Variable (C-Parameter) ───────
function SoftMarginCanvas({ theme }) {
  const canvasRef = useRef(null)
  const [logC, setLogC] = useState(0) // log10(C) from -1 (C=0.1) to 2 (C=100)
  const W = 460
  const H = 280

  const C = Math.pow(10, logC)

  // Dataset with 2 overlapping noisy points
  const rawPoints = useMemo(() => [
    // Class +1
    { id: 1, x: 80,  y: 70,  label: 1 },
    { id: 2, x: 120, y: 60,  label: 1 },
    { id: 3, x: 100, y: 130, label: 1 },
    { id: 4, x: 160, y: 110, label: 1 },
    { id: 5, x: 140, y: 170, label: 1 },
    // Outlier in negative territory!
    { id: 6, x: 235, y: 155, label: 1, isNoise: true },

    // Class -1
    { id: 7, x: 340, y: 220, label: -1 },
    { id: 8, x: 380, y: 180, label: -1 },
    { id: 9, x: 300, y: 250, label: -1 },
    { id: 10,x: 270, y: 195, label: -1 },
    { id: 11,x: 310, y: 140, label: -1 },
    // Outlier in positive territory!
    { id: 12,x: 180, y: 115, label: -1, isNoise: true }
  ], [])

  // Analytical/numerical soft margin model based on C
  const model = useMemo(() => {
    const normPts = rawPoints.map(p => ({
      x: (p.x - W / 2) / 100,
      y: (p.y - H / 2) / 100,
      label: p.label,
      orig: p
    }))

    let w1 = 0.4, w2 = -0.4, b = 0
    const lr = 0.012
    const epochs = 1000

    for (let ep = 0; ep < epochs; ep++) {
      let gradW1 = w1
      let gradW2 = w2
      let gradB = 0

      for (const p of normPts) {
        const score = p.label * (w1 * p.x + w2 * p.y + b)
        if (score < 1) {
          // Hinge loss gradient
          gradW1 -= C * p.label * p.x
          gradW2 -= C * p.label * p.y
          gradB -= C * p.label
        }
      }

      w1 -= lr * (gradW1 / normPts.length)
      w2 -= lr * (gradW2 / normPts.length)
      b -= lr * (gradB / normPts.length)
    }

    const normW = Math.sqrt(w1 * w1 + w2 * w2) || 1

    // Calculate slack variables: xi_i = max(0, 1 - y_i(w^T x + b))
    let totalSlack = 0
    let marginViolations = 0
    const pointStats = normPts.map(p => {
      const functionalMargin = p.label * (w1 * p.x + w2 * p.y + b)
      const slack = Math.max(0, 1 - functionalMargin)
      if (slack > 0) {
        totalSlack += slack
        marginViolations++
      }
      return {
        ...p.orig,
        functionalMargin,
        slack,
        isMisclassified: functionalMargin < 0
      }
    })

    return { w1, w2, b, normW, totalSlack, marginViolations, pointStats }
  }, [rawPoints, C, W, H])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'
    const { w1, w2, b, pointStats } = model

    const evalAt = (cx, cy) => {
      const nx = (cx - W / 2) / 100
      const ny = (cy - H / 2) / 100
      return w1 * nx + w2 * ny + b
    }

    // Draw Margin Strip Region
    const step = 4
    for (let x = 0; x < W; x += step) {
      for (let y = 0; y < H; y += step) {
        const val = evalAt(x, y)
        if (Math.abs(val) <= 1.0) {
          ctx.fillStyle = dark
            ? `rgba(139, 92, 246, ${0.14 * (1 - Math.abs(val))})`
            : `rgba(139, 92, 246, ${0.12 * (1 - Math.abs(val))})`
          ctx.fillRect(x, y, step, step)
        }
      }
    }

    const drawContour = (target, strokeStyle, lineWidth, dashed = false) => {
      ctx.beginPath()
      ctx.strokeStyle = strokeStyle
      ctx.lineWidth = lineWidth
      if (dashed) ctx.setLineDash([5, 4])
      else ctx.setLineDash([])

      if (Math.abs(w2) > 0.001) {
        const yAtX0 = ((target - b - w1 * (-W / 2 / 100)) / w2) * 100 + H / 2
        const yAtXW = ((target - b - w1 * ((W - W / 2) / 100)) / w2) * 100 + H / 2
        ctx.moveTo(0, yAtX0)
        ctx.lineTo(W, yAtXW)
      } else {
        const xAtTarget = ((target - b) / w1) * 100 + W / 2
        ctx.moveTo(xAtTarget, 0)
        ctx.lineTo(xAtTarget, H)
      }
      ctx.stroke()
      ctx.setLineDash([])
    }

    drawContour(1, dark ? 'rgba(99, 102, 241, 0.7)' : 'rgba(79, 70, 229, 0.8)', 2, true)
    drawContour(-1, dark ? 'rgba(16, 185, 129, 0.7)' : 'rgba(5, 150, 105, 0.8)', 2, true)
    drawContour(0, dark ? '#8B5CF6' : '#6D28D9', 2.5, false)

    // Draw Slack Vectors (arrows from violating points to target margin boundary)
    const normLen = Math.sqrt(w1 * w1 + w2 * w2) || 1
    const unitW1 = w1 / normLen
    const unitW2 = w2 / normLen

    pointStats.forEach(p => {
      if (p.slack > 0.05) {
        // Target boundary value is p.label (+1 or -1)
        const targetVal = p.label
        const currentVal = evalAt(p.x, p.y)
        const distPixels = ((targetVal - currentVal) / normLen) * 100

        const destX = p.x + unitW1 * distPixels
        const destY = p.y + unitW2 * distPixels

        ctx.beginPath()
        ctx.moveTo(p.x, p.y)
        ctx.lineTo(destX, destY)
        ctx.strokeStyle = p.isMisclassified ? '#EF4444' : '#F59E0B'
        ctx.lineWidth = 1.5
        ctx.setLineDash([2, 2])
        ctx.stroke()
        ctx.setLineDash([])

        // Small indicator circle at target margin projection
        ctx.beginPath()
        ctx.arc(destX, destY, 3, 0, Math.PI * 2)
        ctx.fillStyle = p.isMisclassified ? '#EF4444' : '#F59E0B'
        ctx.fill()
      }

      // Draw Point
      ctx.beginPath()
      ctx.arc(p.x, p.y, p.slack > 0 ? 8 : 6.5, 0, Math.PI * 2)
      ctx.fillStyle = p.label === 1 ? '#6366F1' : '#10B981'
      ctx.fill()
      ctx.strokeStyle = p.isMisclassified ? '#EF4444' : p.slack > 0 ? '#F59E0B' : (dark ? '#0F172A' : '#FFFFFF')
      ctx.lineWidth = p.slack > 0 ? 2.5 : 1.5
      ctx.stroke()

      // Show indicator
      if (p.isNoise) {
        ctx.font = 'bold 9px sans-serif'
        ctx.fillStyle = p.isMisclassified ? '#EF4444' : '#F59E0B'
        ctx.fillText(`ξ=${p.slack.toFixed(1)}`, p.x + 10, p.y - 6)
      }
    })

    // Legend
    ctx.font = '10px monospace'
    ctx.fillStyle = '#8B5CF6'
    ctx.fillText('Decision boundary (wᵀx + b = 0)', 12, 20)
    ctx.fillStyle = '#EF4444'
    ctx.fillText('Red dash: Slack penalty distance ξᵢ (violation)', 12, 34)
  }, [model, theme, W, H])

  return (
    <div>
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1.5">
          <label className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Slack Penalty Parameter: <span className="font-mono text-purple-400 font-bold text-sm">C = {C < 1 ? C.toFixed(2) : C.toFixed(0)}</span>
          </label>
          <span className={`text-xs font-mono ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>
            {C > 20 ? 'High C: Strict fit, narrow margin (low bias, high variance)' : C < 0.5 ? 'Low C: Wide margin, high tolerance (high bias, low variance)' : 'Balanced C'}
          </span>
        </div>
        <input
          type="range"
          min="-1"
          max="2"
          step="0.05"
          value={logC}
          onChange={e => setLogC(parseFloat(e.target.value))}
          className="w-full accent-purple-500 cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-gray-500 font-mono mt-1">
          <span>C = 0.1 (Max tolerance, wide margin)</span>
          <span>C = 1.0</span>
          <span>C = 10</span>
          <span>C = 100 (Hard margin limit)</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div className={`col-span-2 rounded-2xl border overflow-hidden relative ${
          theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
        }`}>
          <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
        </div>

        <div className={`rounded-2xl border p-4 flex flex-col justify-between ${
          theme === 'dark' ? 'bg-slate-900/90 border-white/10' : 'bg-white border-gray-200'
        }`}>
          <div>
            <h4 className={`text-xs font-bold uppercase tracking-wider mb-3 ${
              theme === 'dark' ? 'text-purple-400' : 'text-purple-600'
            }`}>
              Soft Margin Status
            </h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center text-xs py-1 border-b border-inherit">
                <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Violations Count (ξ &gt; 0)</span>
                <span className="font-mono font-bold text-amber-400">{model.marginViolations} points</span>
              </div>
              <div className="flex justify-between items-center text-xs py-1 border-b border-inherit">
                <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Total Slack Penalty Σ ξᵢ</span>
                <span className="font-mono font-bold text-red-400">{model.totalSlack.toFixed(2)}</span>
              </div>
              <div className="flex justify-between items-center text-xs py-1 border-b border-inherit">
                <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Margin Width (2/||w||)</span>
                <span className="font-mono text-purple-400">{(2 / model.normW).toFixed(2)}</span>
              </div>
              <div className="flex justify-between items-center text-xs py-1">
                <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Total Objective</span>
                <span className="font-mono text-xs text-indigo-400">
                  {(0.5 * model.normW * model.normW + C * model.totalSlack).toFixed(1)}
                </span>
              </div>
            </div>
          </div>

          <div className={`mt-4 p-2.5 rounded-xl text-xs ${
            theme === 'dark' ? 'bg-purple-500/10 text-purple-300 border border-purple-500/20' : 'bg-purple-50 text-purple-700 border border-purple-200'
          }`}>
            <p className="font-mono font-semibold">min ½||w||² + C Σ ξᵢ</p>
            <p className="mt-1 text-[11px] leading-relaxed">
              When C is small, the solver prioritizes a larger margin even if some points cross the line.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Deterministic generator for visualization points (pure, zero Math.random during render)
function makePRNG(seed = 42) {
  let s = seed
  return () => {
    s = (s * 16807 + 11) % 2147483647
    return (s - 1) / 2147483646
  }
}

const LIFT_POINTS = (() => {
  const rand = makePRNG(12345)
  const pts = []
  // Inner cluster (radius ~ 35px)
  for (let i = 0; i < 14; i++) {
    const r = 15 + rand() * 25
    const theta = (i / 14) * Math.PI * 2 + (rand() - 0.5) * 0.3
    pts.push({ x: r * Math.cos(theta), y: r * Math.sin(theta), label: 1 })
  }
  // Outer ring (radius ~ 80px)
  for (let i = 0; i < 20; i++) {
    const r = 70 + rand() * 25
    const theta = (i / 20) * Math.PI * 2 + (rand() - 0.5) * 0.2
    pts.push({ x: r * Math.cos(theta), y: r * Math.sin(theta), label: -1 })
  }
  return pts
})()

// ── Interactive Component 3: 2D-to-3D Kernel Lifting Visualization ───────────
function KernelLiftCanvas({ theme }) {
  const canvasRef = useRef(null)
  const [elevation, setElevation] = useState(0.85) // 0 (flat 2D) to 1 (full 3D paraboloid)
  const [rotAngle, setRotAngle] = useState(0.55)
  const W = 460
  const H = 280

  const data = LIFT_POINTS

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'
    const cx = W / 2
    const cy = H / 2 + 20

    // 3D Isometric projection parameters
    const pitch = 0.52 // tilt down
    const yaw = rotAngle

    const project = (x, y, z) => {
      // Rotate around Z axis (yaw)
      const rx = x * Math.cos(yaw) - y * Math.sin(yaw)
      const ry = x * Math.sin(yaw) + y * Math.cos(yaw)
      // Tilt around X axis (pitch)
      const px = rx
      const py = ry * Math.cos(pitch) - z * Math.sin(pitch)
      return { px: cx + px * 1.8, py: cy + py * 1.8 }
    }

    // Draw base plane grid (z = 0)
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
    ctx.lineWidth = 1
    const gridSize = 110
    const step = 22
    for (let g = -gridSize; g <= gridSize; g += step) {
      const p1 = project(g, -gridSize, 0)
      const p2 = project(g, gridSize, 0)
      ctx.beginPath()
      ctx.moveTo(p1.px, p1.py)
      ctx.lineTo(p2.px, p2.py)
      ctx.stroke()

      const p3 = project(-gridSize, g, 0)
      const p4 = project(gridSize, g, 0)
      ctx.beginPath()
      ctx.moveTo(p3.px, p3.py)
      ctx.lineTo(p4.px, p4.py)
      ctx.stroke()
    }

    // Draw 3D Cutting Plane if elevation > 0.3
    const planeZ = 28 * elevation
    if (elevation > 0.25) {
      const corner1 = project(-gridSize, -gridSize, planeZ)
      const corner2 = project(gridSize, -gridSize, planeZ)
      const corner3 = project(gridSize, gridSize, planeZ)
      const corner4 = project(-gridSize, gridSize, planeZ)

      ctx.beginPath()
      ctx.moveTo(corner1.px, corner1.py)
      ctx.lineTo(corner2.px, corner2.py)
      ctx.lineTo(corner3.px, corner3.py)
      ctx.lineTo(corner4.px, corner4.py)
      ctx.closePath()
      ctx.fillStyle = dark ? 'rgba(245, 158, 11, 0.18)' : 'rgba(217, 119, 6, 0.16)'
      ctx.fill()
      ctx.strokeStyle = '#F59E0B'
      ctx.lineWidth = 1.5
      ctx.setLineDash([4, 3])
      ctx.stroke()
      ctx.setLineDash([])

      ctx.fillStyle = '#F59E0B'
      ctx.font = 'bold 10px monospace'
      ctx.fillText('Hyperplane: z = z₀', corner2.px - 60, corner2.py - 6)
    }

    // Project and sort points by 3D depth for correct occlusion
    const projectedPoints = data.map(p => {
      // Paraboloid feature map: z = (x^2 + y^2) / 80
      const r2 = (p.x * p.x + p.y * p.y) / 100
      const z = r2 * 0.9 * elevation
      const pt3D = project(p.x, p.y, z)
      const ptBase = project(p.x, p.y, 0)
      return { ...p, z, px: pt3D.px, py: pt3D.py, basePx: ptBase.px, basePy: ptBase.py }
    })

    projectedPoints.sort((a, b) => a.py - b.py)

    // Draw stems connecting base (x, y) to lifted (x, y, z)
    if (elevation > 0.1) {
      projectedPoints.forEach(p => {
        ctx.beginPath()
        ctx.moveTo(p.basePx, p.basePy)
        ctx.lineTo(p.px, p.py)
        ctx.strokeStyle = p.label === 1 ? 'rgba(99, 102, 241, 0.4)' : 'rgba(16, 185, 129, 0.4)'
        ctx.lineWidth = 1.2
        ctx.setLineDash([2, 2])
        ctx.stroke()
        ctx.setLineDash([])
      })
    }

    // Draw 3D points
    projectedPoints.forEach(p => {
      ctx.beginPath()
      ctx.arc(p.px, p.py, 6, 0, Math.PI * 2)
      ctx.fillStyle = p.label === 1 ? '#6366F1' : '#10B981'
      ctx.fill()
      ctx.strokeStyle = dark ? '#FFFFFF' : '#0F172A'
      ctx.lineWidth = 1.5
      ctx.stroke()
    })

    // Label coordinates
    ctx.font = '10px monospace'
    ctx.fillStyle = dark ? '#94A3B8' : '#64748B'
    ctx.fillText('Original 2D: [x₁, x₂] → Feature Space: [x₁, x₂, z = x₁² + x₂²]', 12, 20)
  }, [elevation, rotAngle, data, theme, W, H])

  return (
    <div>
      <div className="grid grid-cols-2 gap-4 mb-3">
        <div>
          <label className={`text-xs font-semibold block mb-1 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Feature Lift (z = λ·(x₁² + x₂²)): <span className="text-cyan-400 font-bold font-mono">{(elevation * 100).toFixed(0)}%</span>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.02"
            value={elevation}
            onChange={e => setElevation(parseFloat(e.target.value))}
            className="w-full accent-cyan-500 cursor-pointer"
          />
        </div>
        <div>
          <label className={`text-xs font-semibold block mb-1 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Orbit Camera Angle: <span className="text-purple-400 font-bold font-mono">{((rotAngle / Math.PI) * 180).toFixed(0)}°</span>
          </label>
          <input
            type="range"
            min="0"
            max={Math.PI * 2}
            step="0.05"
            value={rotAngle}
            onChange={e => setRotAngle(parseFloat(e.target.value))}
            className="w-full accent-purple-500 cursor-pointer"
          />
        </div>
      </div>

      <div className={`rounded-2xl border overflow-hidden relative mb-2 ${
        theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
      }`}>
        <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
      </div>
      <p className={`text-xs text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        Slide the elevation up to see how the circular boundary in 2D becomes a completely flat cutting hyperplane in 3D!
      </p>
    </div>
  )
}

// Precomputed deterministic benchmark datasets for Kernel Zoo
const ZOO_DATASETS = (() => {
  const rand = makePRNG(9876)
  const W = 460
  const H = 280

  const circles = []
  for (let i = 0; i < 18; i++) {
    const r = 20 + rand() * 25
    const a = (i / 18) * Math.PI * 2
    circles.push({ x: W / 2 + r * Math.cos(a), y: H / 2 + r * Math.sin(a), label: 1 })
  }
  for (let i = 0; i < 26; i++) {
    const r = 85 + rand() * 25
    const a = (i / 26) * Math.PI * 2
    circles.push({ x: W / 2 + r * Math.cos(a), y: H / 2 + r * Math.sin(a), label: -1 })
  }

  const moons = []
  for (let i = 0; i < 22; i++) {
    const a = (i / 22) * Math.PI
    const r = 60
    moons.push({
      x: W / 2 - 40 + r * Math.cos(a) + (rand() - 0.5) * 16,
      y: H / 2 - 20 - r * Math.sin(a) + (rand() - 0.5) * 16,
      label: 1
    })
  }
  for (let i = 0; i < 22; i++) {
    const a = (i / 22) * Math.PI
    const r = 60
    moons.push({
      x: W / 2 + 30 - r * Math.cos(a) + (rand() - 0.5) * 16,
      y: H / 2 + 10 + r * Math.sin(a) + (rand() - 0.5) * 16,
      label: -1
    })
  }

  const xor = []
  for (let i = 0; i < 12; i++) {
    xor.push({ x: W / 2 - 80 + rand() * 50, y: H / 2 - 70 + rand() * 45, label: 1 })
    xor.push({ x: W / 2 + 30 + rand() * 50, y: H / 2 + 25 + rand() * 45, label: 1 })
    xor.push({ x: W / 2 + 30 + rand() * 50, y: H / 2 - 70 + rand() * 45, label: -1 })
    xor.push({ x: W / 2 - 80 + rand() * 50, y: H / 2 + 25 + rand() * 45, label: -1 })
  }

  return { circles, moons, xor }
})()

// Pure top-level kernel evaluation function
function evaluateKernel(x1, y1, x2, y2, kernelType, gamma, polyDegree) {
  const dot = x1 * x2 + y1 * y2
  if (kernelType === 'linear') return dot
  if (kernelType === 'poly') return Math.pow(dot + 1, polyDegree)
  if (kernelType === 'rbf') return Math.exp(-gamma * ((x1 - x2) ** 2 + (y1 - y2) ** 2))
  if (kernelType === 'sigmoid') return Math.tanh(0.8 * dot - 0.2)
  return dot
}

// ── Interactive Component 4: The Kernel Zoo Classifier Playground ─────────────
function KernelZooCanvas({ theme }) {
  const canvasRef = useRef(null)
  const [datasetType, setDatasetType] = useState('circles')
  const [kernelType, setKernelType] = useState('rbf')
  const [gamma, setGamma] = useState(1.5)
  const [polyDegree, setPolyDegree] = useState(3)
  const [C, setC] = useState(2.0)
  const W = 460
  const H = 280

  const data = ZOO_DATASETS[datasetType] || ZOO_DATASETS.circles

  // Dual solver using Kernel Matrix K
  const model = useMemo(() => {
    const normPts = data.map(p => ({
      nx: (p.x - W / 2) / 60,
      ny: (p.y - H / 2) / 60,
      label: p.label,
      orig: p
    }))

    const n = normPts.length
    // Dual variables alpha_i >= 0 initialized
    const alphas = new Array(n).fill(0.1)
    let b = 0

    // Coordinate descent / sequential SMO-like update steps
    const epochs = 180
    const lr = 0.015

    for (let ep = 0; ep < epochs; ep++) {
      for (let i = 0; i < n; i++) {
        // Compute current decision score f(x_i)
        let score = b
        for (let j = 0; j < n; j++) {
          score += alphas[j] * normPts[j].label * evaluateKernel(normPts[j].nx, normPts[j].ny, normPts[i].nx, normPts[i].ny, kernelType, gamma, polyDegree)
        }
        const error = normPts[i].label * score - 1
        // Gradient update capped between [0, C]
        alphas[i] = Math.max(0, Math.min(C, alphas[i] - lr * error))
      }
    }

    // Support vectors are points where alpha > 0.02
    const svs = []
    alphas.forEach((a, idx) => {
      if (a > 0.02) svs.push({ ...normPts[idx], alpha: a })
    })

    // Compute bias b from support vectors
    if (svs.length > 0) {
      let bSum = 0
      svs.slice(0, 8).forEach(sv => {
        let f = 0
        for (let j = 0; j < n; j++) {
          f += alphas[j] * normPts[j].label * evaluateKernel(normPts[j].nx, normPts[j].ny, sv.nx, sv.ny, kernelType, gamma, polyDegree)
        }
        bSum += sv.label - f
      })
      b = bSum / Math.min(svs.length, 8)
    }

    return { normPts, alphas, b, svs }
  }, [data, kernelType, gamma, polyDegree, C, W, H])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'
    const { normPts, alphas, b } = model

    // Rasterize decision space on grid
    const step = 8
    for (let x = 0; x < W; x += step) {
      for (let y = 0; y < H; y += step) {
        const nx = (x - W / 2) / 60
        const ny = (y - H / 2) / 60

        // Compute kernel decision function: f(x) = sum(alpha_i * y_i * K(x_i, x)) + b
        let f = b
        for (let i = 0; i < normPts.length; i++) {
          if (alphas[i] > 0.001) {
            f += alphas[i] * normPts[i].label * evaluateKernel(normPts[i].nx, normPts[i].ny, nx, ny, kernelType, gamma, polyDegree)
          }
        }

        // Color based on class prediction and distance to boundary
        if (f >= 0) {
          ctx.fillStyle = dark
            ? `rgba(99, 102, 241, ${Math.min(0.24, 0.06 + Math.abs(f) * 0.05)})`
            : `rgba(99, 102, 241, ${Math.min(0.22, 0.05 + Math.abs(f) * 0.04)})`
        } else {
          ctx.fillStyle = dark
            ? `rgba(16, 185, 129, ${Math.min(0.24, 0.06 + Math.abs(f) * 0.05)})`
            : `rgba(16, 185, 129, ${Math.min(0.22, 0.05 + Math.abs(f) * 0.04)})`
        }
        ctx.fillRect(x, y, step, step)
      }
    }

    // Draw training data points
    data.forEach(p => {
      const isSV = model.svs.some(sv => sv.orig === p)
      if (isSV) {
        ctx.beginPath()
        ctx.arc(p.x, p.y, 11, 0, Math.PI * 2)
        ctx.strokeStyle = '#F59E0B'
        ctx.lineWidth = 2
        ctx.stroke()
      }

      ctx.beginPath()
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2)
      ctx.fillStyle = p.label === 1 ? '#6366F1' : '#10B981'
      ctx.fill()
      ctx.strokeStyle = dark ? '#0F172A' : '#FFFFFF'
      ctx.lineWidth = 1.5
      ctx.stroke()
    })

    // Legend
    ctx.font = '10px monospace'
    ctx.fillStyle = '#6366F1'
    ctx.fillText('Class +1 (Indigo)', 12, 20)
    ctx.fillStyle = '#10B981'
    ctx.fillText('Class −1 (Emerald)', 12, 34)
    ctx.fillStyle = '#F59E0B'
    ctx.fillText('Yellow Rings = Support Vectors', 12, 48)
  }, [model, data, theme, W, H, kernelType, gamma, polyDegree])

  return (
    <div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <div>
          <label className={`text-xs font-semibold block mb-1 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Pattern:
          </label>
          <select
            value={datasetType}
            onChange={e => setDatasetType(e.target.value)}
            className={`w-full px-2.5 py-1.5 rounded-lg text-xs font-medium border outline-none ${
              theme === 'dark' ? 'bg-slate-800 border-white/10 text-white' : 'bg-white border-gray-200 text-gray-800'
            }`}
          >
            <option value="circles">Concentric Rings</option>
            <option value="moons">Two Moons</option>
            <option value="xor">XOR Quadrants</option>
          </select>
        </div>

        <div>
          <label className={`text-xs font-semibold block mb-1 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Kernel Function:
          </label>
          <select
            value={kernelType}
            onChange={e => setKernelType(e.target.value)}
            className={`w-full px-2.5 py-1.5 rounded-lg text-xs font-medium border outline-none ${
              theme === 'dark' ? 'bg-slate-800 border-white/10 text-white' : 'bg-white border-gray-200 text-gray-800'
            }`}
          >
            <option value="rbf">RBF (Gaussian)</option>
            <option value="poly">Polynomial</option>
            <option value="linear">Linear</option>
            <option value="sigmoid">Sigmoid</option>
          </select>
        </div>

        {kernelType === 'rbf' && (
          <div>
            <label className={`text-xs font-semibold block mb-1 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              Bandwidth (γ): <span className="text-cyan-400 font-bold">{gamma.toFixed(1)}</span>
            </label>
            <input
              type="range"
              min="0.2"
              max="5"
              step="0.2"
              value={gamma}
              onChange={e => setGamma(parseFloat(e.target.value))}
              className="w-full accent-cyan-500 cursor-pointer"
            />
          </div>
        )}

        {kernelType === 'poly' && (
          <div>
            <label className={`text-xs font-semibold block mb-1 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              Degree (d): <span className="text-cyan-400 font-bold">{polyDegree}</span>
            </label>
            <input
              type="range"
              min="2"
              max="5"
              step="1"
              value={polyDegree}
              onChange={e => setPolyDegree(parseInt(e.target.value, 10))}
              className="w-full accent-cyan-500 cursor-pointer"
            />
          </div>
        )}

        <div>
          <label className={`text-xs font-semibold block mb-1 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Penalty (C): <span className="text-purple-400 font-bold">{C.toFixed(1)}</span>
          </label>
          <input
            type="range"
            min="0.5"
            max="10"
            step="0.5"
            value={C}
            onChange={e => setC(parseFloat(e.target.value))}
            className="w-full accent-purple-500 cursor-pointer"
          />
        </div>
      </div>

      <div className={`rounded-2xl border overflow-hidden relative mb-2 ${
        theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
      }`}>
        <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
      </div>

      <p className={`text-xs text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        Try switching to Linear Kernel on Concentric Rings to observe complete failure, then switch to RBF to watch it seamlessly carve the non-linear boundary!
      </p>
    </div>
  )
}

// ── Main SVM Page Component ──────────────────────────────────────────────────
export default function SVM() {
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
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/20 border border-indigo-500/30 mb-4"
        >
          <span className="text-xs text-indigo-400 font-medium">Machine Learning • Maximum Margin &amp; Kernel Methods</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Support Vector <span className="gradient-text">Machines (SVM)</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          One of the mathematically richest algorithms in machine learning. Instead of just drawing <em>any</em> line that separates classes, an SVM computes the unique <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>Maximum Margin Hyperplane</strong>, governed entirely by the critical <strong className="text-amber-400">Support Vectors</strong> that touch the boundary gutters.
        </p>

        <Callout type="analogy" title="Analogy: The demilitarized highway between two kingdoms">
          Imagine two rival kingdoms separated by a wide highway. A reckless general might build a fence right against enemy huts. But a wise diplomat builds the highway so that the gap between the closest guard posts of each kingdom is as wide as humanly possible. If a guard post is moved back, it doesn’t affect the border; only the guard posts right on the perimeter fence (the <strong>Support Vectors</strong>) determine where the boundary lies!
        </Callout>
      </div>

      {/* Core Concepts Strip */}
      <TheoryBlock
        title="Core Architectural Pillars"
        cards={[
          {
            icon: '📏',
            title: 'Maximum Margin',
            body: 'Separates classes by maximizing the geometric margin (2/||w||). Maximizing the margin minimizes the VC dimension, conferring formal mathematical resistance to overfitting.',
            mono: 'margin width = 2 / ||w||'
          },
          {
            icon: '🛡️',
            title: 'Support Vectors',
            body: 'Only the critical training points lying exactly on or inside the margin gutters determine the boundary. All non-support vectors can be deleted with zero change to the classifier.',
            mono: 'αᵢ > 0  (KKT condition)'
          },
          {
            icon: '⚖️',
            title: 'Slack Variables (C)',
            body: 'Real-world data has noise and overlap. Soft-margin SVM introduces slack variables ξᵢ with penalty parameter C to trade off margin width against training errors.',
            mono: 'min ½||w||² + C Σ ξᵢ'
          },
          {
            icon: '✨',
            title: 'The Kernel Trick',
            body: 'Maps non-linear data into high or infinite-dimensional Hilbert spaces. Evaluates dot products directly via K(x, z) without ever computing the explicit coordinate mapping.',
            mono: 'K(x, z) = ⟨ϕ(x), ϕ(z)⟩'
          },
          {
            icon: '📜',
            title: 'Mercer\'s Theorem',
            body: 'Guarantees that any symmetric, positive semi-definite (PSD) Gram matrix represents an authentic inner product in an implicit reproducing kernel Hilbert space (RKHS).',
            mono: 'cᵀ K c ≥ 0  for all c'
          },
          {
            icon: '⚡',
            title: 'Convex Optimization',
            body: 'Unlike neural networks with chaotic non-convex loss surfaces and local minima, the SVM dual is a strictly quadratic convex problem — guaranteed global minimum!',
            mono: 'Unique global optimum'
          }
        ]}
      />

      {/* ========================================================================= */}
      {/* SECTION 1: LINEARLY SEPARABLE DATA */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>
          Part 1 — Linearly Separable Data
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Hard-Margin SVM &amp; Geometric Foundations
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          The math and geometry of the maximum margin hyperplane, support vectors, and KKT conditions.
        </p>
      </div>

      {/* 1.1 Maximum margin classifier */}
      <div className={S}>
        <p className={LBL}>1.1 — The Maximum Margin Classifier</p>
        <h2 className={H2}>Why any separating line is not good enough</h2>
        <p className={`${BODY} mb-4`}>
          When data points belonging to two classes can be separated cleanly by a straight line (or flat hyperplane in higher dimensions), there are infinitely many possible separating hyperplanes. Perceptrons simply stop at the first line they encounter, which often skims perilously close to training points. If test points have even slight noise, they will be misclassified!
        </p>

        <Callout type="formula" mono="wᵀx + b = 0   (Decision boundary)  |  yᵢ ∈ {-1, +1}">
          The Support Vector Machine selects the <strong>unique optimal hyperplane</strong> that leaves the maximum perpendicular clearance (margin) to both classes.
        </Callout>

        <p className={`${BODY} mb-4`}>
          Try the interactive hard-margin visualizer below. Observe how the dashed margin lines hug the closest points (the Support Vectors), and use the <strong>Test Button</strong> to see how deleting all other points leaves the boundary unchanged:
        </p>

        <HardMarginCanvas theme={theme} />
      </div>

      {/* 1.2 Support vectors, margin width and geometric derivation */}
      <div className={S}>
        <p className={LBL}>1.2 — Geometric Derivation of Margin Width</p>
        <h2 className={H2}>Deriving γ = 2 / ||w|| from first principles</h2>
        <p className={`${BODY} mb-4`}>
          Let the separating hyperplane be defined by the equation <code className="font-mono font-semibold">wᵀx + b = 0</code>, where <code className="font-mono">w</code> is the normal vector perpendicular to the hyperplane. By scaling <code className="font-mono">(w, b)</code>, we can scale the functional margin so that the closest points satisfy:
        </p>

        <TheoryBlock
          title="Margin Derivation Steps"
          cards={[
            {
              icon: '1️⃣',
              title: 'Canonical Scaling',
              body: 'For the closest positive and negative points, we set the functional margins to exactly +1 and -1. All training samples satisfy yᵢ(wᵀxᵢ + b) ≥ 1.',
              mono: 'wᵀx⁺ + b = +1  and  wᵀx⁻ + b = -1'
            },
            {
              icon: '2️⃣',
              title: 'Projection along Normal Vector',
              body: 'Subtracting the two equations gives wᵀ(x⁺ - x⁻) = 2. Divide both sides by the Euclidean norm ||w|| to obtain the geometric distance projected onto unit vector w/||w||.',
              mono: '(w / ||w||)ᵀ (x⁺ - x⁻) = 2 / ||w||'
            },
            {
              icon: '3️⃣',
              title: 'Total Margin Width γ',
              body: 'The total perpendicular distance between the positive margin gutter and the negative margin gutter is exactly 2 / ||w||. Maximizing 2/||w|| is equivalent to minimizing ½ ||w||²!',
              mono: 'max (2 / ||w||) ⟺ min ½ ||w||²'
            }
          ]}
        />

        <Callout type="info" title="Why minimize ½ ||w||² instead of 1/||w||?">
          Maximizing <code className="font-mono">2 / ||w||</code> is non-linear and awkward to optimize. Inverting it to <code className="font-mono">min ½ ||w||²</code> turns the problem into a standard, strictly convex <strong>Quadratic Programming (QP)</strong> problem with a unique global minimum and zero local traps.
        </Callout>
      </div>

      {/* 1.3 Primal and Dual formulation */}
      <div className={S}>
        <p className={LBL}>1.3 — Primal vs Dual Formulation</p>
        <h2 className={H2}>The Lagrangian transformation that unlocks kernels</h2>
        <p className={`${BODY} mb-4`}>
          The primal optimization problem seeks the weight vector <code className="font-mono">w</code> and bias <code className="font-mono">b</code> directly:
        </p>

        <Callout type="formula" mono="Primal: min_{w, b} ½ ||w||²   subject to   yᵢ(wᵀxᵢ + b) ≥ 1,  ∀i=1..n">
          The primal problem has <code className="font-mono">d</code> variables (the dimension of the feature space) and <code className="font-mono">n</code> inequality constraints.
        </Callout>

        <p className={`${BODY} mb-4`}>
          We construct the generalized Lagrangian by introducing non-negative Lagrange multipliers <code className="font-mono">αᵢ ≥ 0</code> for every constraint:
        </p>

        <div className={`p-4 rounded-xl font-mono text-xs mb-4 overflow-x-auto ${
          theme === 'dark' ? 'bg-slate-950 text-indigo-300 border border-white/10' : 'bg-slate-100 text-indigo-800 border border-slate-300'
        }`}>
          L(w, b, α) = ½ ||w||² − Σᵢ₌₁ⁿ αᵢ [yᵢ(wᵀxᵢ + b) − 1]
        </div>

        <p className={`${BODY} mb-3`}>
          Setting the partial derivatives with respect to the primal variables to zero:
        </p>
        <ul className={`list-disc list-inside space-y-1.5 text-xs font-mono mb-4 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
          <li>∇_w L = w − Σᵢ αᵢ yᵢ xᵢ = 0 ⟹ <strong>w = Σᵢ₌₁ⁿ αᵢ yᵢ xᵢ</strong></li>
          <li>∂L/∂b = − Σᵢ αᵢ yᵢ = 0 ⟹ <strong>Σᵢ₌₁ⁿ αᵢ yᵢ = 0</strong></li>
        </ul>

        <p className={`${BODY} mb-4`}>
          Substituting <code className="font-mono">w = Σ αᵢ yᵢ xᵢ</code> back into the Lagrangian yields the celebrated <strong>Wolfe Dual</strong>:
        </p>

        <Callout type="success" title="The Wolfe Dual Problem" mono="max_α Σᵢ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢ · xⱼ)   s.t.  αᵢ ≥ 0,  Σᵢ αᵢ yᵢ = 0">
          <strong>The Monumental Revelation:</strong> Notice how the training points appear <em>strictly</em> as pairwise dot products <code className="font-mono">xᵢ · xⱼ</code>! The feature dimension <code className="font-mono">d</code> has completely vanished from the objective!
        </Callout>

        <DeepDive title="Why is the Dual formulation so revolutionary?">
          <p className={`text-sm ${BODY} mb-2`}>
            In the primal formulation, if we map data into a 1,000,000-dimensional space, <code className="font-mono">w</code> has 1,000,000 coordinates, making optimization impossible. But in the dual formulation, we only have <code className="font-mono">n</code> variables (<code className="font-mono">α₁ ... αₙ</code>), regardless of whether the feature dimension is 10, 10,000, or infinite!
          </p>
          <p className={`text-sm ${BODY}`}>
            Furthermore, once we know how to compute <code className="font-mono">K(xᵢ, xⱼ)</code>, we can substitute it directly into the dot product slot, enabling infinite-dimensional classification without ever storing a single high-dimensional coordinate.
          </p>
        </DeepDive>
      </div>

      {/* 1.4 KKT Conditions */}
      <div className={S}>
        <p className={LBL}>1.4 — Karush-Kuhn-Tucker (KKT) Conditions</p>
        <h2 className={H2}>The mathematical sieve for Support Vectors</h2>
        <p className={`${BODY} mb-4`}>
          Because the objective function is convex and the constraints are affine, the Karush-Kuhn-Tucker (KKT) conditions are both necessary and sufficient for global optimality:
        </p>

        <TheoryBlock
          title="KKT Conditions for SVM"
          cards={[
            {
              icon: '1️⃣',
              title: 'Stationarity',
              body: 'The gradient of the Lagrangian with respect to primal variables w and b must vanish at the optimum.',
              mono: 'w = Σ αᵢ yᵢ xᵢ   and   Σ αᵢ yᵢ = 0'
            },
            {
              icon: '2️⃣',
              title: 'Primal Feasibility',
              body: 'Every training point must satisfy the margin constraint (correct side or on the margin).',
              mono: 'yᵢ(wᵀxᵢ + b) − 1 ≥ 0,  ∀i'
            },
            {
              icon: '3️⃣',
              title: 'Dual Feasibility',
              body: 'All Lagrange multipliers must be non-negative.',
              mono: 'αᵢ ≥ 0,  ∀i'
            },
            {
              icon: '🎯',
              title: 'Complementary Slackness',
              body: 'The product of each multiplier and its constraint must be exactly zero! This is the most crucial condition.',
              mono: 'αᵢ · [yᵢ(wᵀxᵢ + b) − 1] = 0'
            }
          ]}
        />

        <Callout type="analogy" title="The Sparsity Miracle">
          Look closely at complementary slackness: <code className="font-mono">αᵢ [yᵢ(wᵀxᵢ + b) − 1] = 0</code>.
          <br /><br />
          • Case 1: Point is strictly outside the margin (<code className="font-mono">yᵢ(wᵀxᵢ + b) &gt; 1</code>). For the product to equal 0, <strong>αᵢ MUST BE 0</strong>! The point has zero weight in <code className="font-mono">w = Σ αᵢ yᵢ xᵢ</code>.
          <br />
          • Case 2: Point has <code className="font-mono">αᵢ &gt; 0</code>. For the product to equal 0, <strong>yᵢ(wᵀxᵢ + b) MUST BE EXACTLY 1</strong>! The point lies right on the boundary gutter. It is a <strong>Support Vector</strong>!
        </Callout>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 2: NON-LINEARLY SEPARABLE DATA & KERNEL TRICK */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-purple-500/5 border-purple-500/20' : 'bg-purple-50 border-purple-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-purple-400' : 'text-purple-600'}`}>
          Part 2 — Non-Linearly Separable Data &amp; Kernel Trick
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Soft Margins, High-Dimensional Embeddings &amp; Kernels
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Handling real-world noise with slack variables and breaking linear barriers via Mercer kernels.
        </p>
      </div>

      {/* 2.1 Soft-margin SVM & C parameter */}
      <div className={S}>
        <p className={LBL}>2.1 — Soft-Margin SVM &amp; Slack Variables (ξ)</p>
        <h2 className={H2}>Tolerating noise and balancing the margin vs errors</h2>
        <p className={`${BODY} mb-4`}>
          In real applications, data is rarely strictly linearly separable. Even a single outlier would cause a hard-margin SVM to either fail completely (no feasible solution) or produce a ridiculously narrow, overfitted margin. Corinna Cortes and Vladimir Vapnik (1995) introduced <strong>slack variables</strong> <code className="font-mono">ξᵢ ≥ 0</code> to allow controlled margin violations:
        </p>

        <Callout type="formula" mono="yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ,   where ξᵢ ≥ 0">
          • <code className="font-mono">ξᵢ = 0</code>: Point is safely on or outside the margin (no violation).<br />
          • <code className="font-mono">0 &lt; ξᵢ ≤ 1</code>: Point is inside the margin gap, but still on the correct side of the decision boundary.<br />
          • <code className="font-mono">ξᵢ &gt; 1</code>: Point has crossed the decision boundary and is misclassified!
        </Callout>

        <p className={`${BODY} mb-4`}>
          We modify the objective function to minimize both the inverse margin and the total sum of violations:
        </p>

        <Callout type="formula" mono="Soft-Margin Primal: min_{w, b, ξ}  ½ ||w||² + C Σᵢ₌₁ⁿ ξᵢ">
          The hyperparameter <strong className="text-purple-400">C</strong> acts as a trading knob between margin width and classification tolerance.
        </Callout>

        <p className={`${BODY} mb-4`}>
          Drag the slider below to see how tuning <code className="font-mono">C</code> adapts the boundary and calculates the slack distances <code className="font-mono">ξᵢ</code> for outlier points:
        </p>

        <SoftMarginCanvas theme={theme} />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className="font-semibold text-xs text-amber-400 mb-1">Small C (e.g. C = 0.1)</p>
            <p className={`text-xs ${BODY}`}>
              Tolerates many margin violations. Prioritizes a wide, smooth margin. Less sensitive to individual outliers. Lower variance, higher bias (guards against overfitting).
            </p>
          </div>
          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className="font-semibold text-xs text-purple-400 mb-1">Large C (e.g. C = 100)</p>
            <p className={`text-xs ${BODY}`}>
              Heavily penalizes every violation. Forces a narrow margin that strives to classify every training point correctly. Lower bias, higher variance (prone to overfitting).
            </p>
          </div>
        </div>

        <DeepDive title="The Box Constraint in the Soft-Margin Dual">
          <p className={`text-sm ${BODY} mb-2`}>
            In the soft-margin dual problem, the slack variables <code className="font-mono">ξᵢ</code> translate into an upper bound on the Lagrange multipliers:
          </p>
          <div className={`p-3 rounded-lg font-mono text-xs mb-2 ${theme === 'dark' ? 'bg-slate-950 text-purple-300' : 'bg-slate-100 text-purple-800'}`}>
            0 ≤ αᵢ ≤ C,   ∀i
          </div>
          <p className={`text-sm ${BODY}`}>
            This is known as the <strong>box constraint</strong>. In hard-margin, <code className="font-mono">αᵢ</code> could grow to infinity for an outlier point, dragging the entire boundary with it. In soft-margin, no single point can exert an influence greater than <code className="font-mono">C</code>!
          </p>
        </DeepDive>
      </div>

      {/* 2.2 Kernel Trick — mapping to higher dimensions */}
      <div className={S}>
        <p className={LBL}>2.2 — The Kernel Trick</p>
        <h2 className={H2}>Lifting points into higher dimensions without paying the price</h2>
        <p className={`${BODY} mb-4`}>
          When data points form concentric circles, intertwining moons, or an XOR pattern, no linear hyperplane in the input space can separate them. The classic mathematical remedy is to map the input vector <code className="font-mono">x ∈ ℝᵈ</code> into a higher-dimensional feature space <code className="font-mono">ϕ(x) ∈ ℝᴰ</code> where <code className="font-mono">D ≫ d</code>.
        </p>

        <p className={`${BODY} mb-4`}>
          Interactive demonstration: Watch how concentric 2D rings become linearly separable when elevated into a 3D paraboloid <code className="font-mono">z = x₁² + x₂²</code>:
        </p>

        <KernelLiftCanvas theme={theme} />

        <Callout type="analogy" title="The Kernel Trick Breakthrough">
          If <code className="font-mono">D</code> is very large (e.g. 10¹² or infinite), explicitly calculating <code className="font-mono">ϕ(x)</code> for every data point and taking their dot product would freeze any computer.
          <br /><br />
          The <strong>Kernel Trick</strong> observes: We don't need <code className="font-mono">ϕ(x)</code> itself! We only ever need the scalar value of the dot product <code className="font-mono">⟨ϕ(x), ϕ(z)⟩</code> in the Wolfe dual. If a function <code className="font-mono">K(x, z)</code> computes that dot product directly using operations in the low-dimensional input space, we get the infinite-dimensional separation <strong>completely for free</strong>!
        </Callout>
      </div>

      {/* 2.3 Mercer's Theorem */}
      <div className={S}>
        <p className={LBL}>2.3 — Mercer's Theorem &amp; Valid Kernels</p>
        <h2 className={H2}>How do we know if K(x, z) is a valid inner product?</h2>
        <p className={`${BODY} mb-4`}>
          Can we pick any mathematical formula for <code className="font-mono">K(x, z)</code>? No. For <code className="font-mono">K(x, z)</code> to correspond to an authentic dot product in some Reproducing Kernel Hilbert Space (RKHS), it must satisfy <strong>Mercer's Condition</strong>:
        </p>

        <TheoryBlock
          title="Mercer's Theorem Requirements"
          cards={[
            {
              icon: '🪞',
              title: 'Symmetry',
              body: 'The kernel function must be symmetric under argument exchange: K(x, z) = K(z, x) for all pairs.',
              mono: 'K(x, z) = K(z, x)'
            },
            {
              icon: '📐',
              title: 'Positive Semi-Definite Gram Matrix',
              body: 'For any arbitrary set of points {x₁, ..., xₙ}, the Gram matrix K with entries Kᵢⱼ = K(xᵢ, xⱼ) must have all non-negative eigenvalues (cᵀ K c ≥ 0 for all c ∈ ℝⁿ).',
              mono: 'Σᵢ Σⱼ cᵢ cⱼ K(xᵢ, xⱼ) ≥ 0'
            },
            {
              icon: '🧩',
              title: 'Kernel Algebra (Closure)',
              body: 'Valid kernels can be combined! The sum of two kernels is a kernel; the product of two kernels is a kernel; multiplying by a positive constant yields a valid kernel.',
              mono: 'K_new = a·K₁ + K₂ · K₃'
            }
          ]}
        />

        <Callout type="info" title="Why is Positive Semi-Definiteness non-negotiable?">
          If the Gram matrix were not PSD, the quadratic optimization problem would become non-convex, possessing multiple local minima and saddle points, destroying the guaranteed global optimality of SVMs!
        </Callout>
      </div>

      {/* 2.4 Common Kernels */}
      <div className={S}>
        <p className={LBL}>2.4 — Common Kernel Functions in Practice</p>
        <h2 className={H2}>The Kernel Zoo: Linear, Polynomial, RBF &amp; Sigmoid</h2>
        <p className={`${BODY} mb-4`}>
          Different kernels define different geometries of similarity. Below are the four standard kernels used in machine learning:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
          {[
            {
              name: 'Linear Kernel',
              formula: 'K(x, z) = xᵀz + c',
              desc: 'No mapping (standard Euclidean dot product). Fastest to train. Best when the number of features d is already very large (e.g. text classification, gene expression).',
              badge: 'O(n·d) Fast',
              color: 'text-indigo-400'
            },
            {
              name: 'Polynomial Kernel',
              formula: 'K(x, z) = (xᵀz + c)ᵈ',
              desc: 'Models all degree-d interactions between features. A quadratic kernel (d=2) in 2D creates an implicit 5-dimensional feature space containing x₁², x₂², x₁x₂, etc.',
              badge: 'Curved Boundaries',
              color: 'text-purple-400'
            },
            {
              name: 'RBF (Gaussian) Kernel',
              formula: 'K(x, z) = exp(−γ ||x − z||²)',
              desc: 'The most popular kernel in ML! Measures Euclidean closeness via a bell curve. Maps points into an INFINITE-dimensional Hilbert space via Taylor series expansion of e^u!',
              badge: 'Infinite Dimensions',
              color: 'text-cyan-400'
            },
            {
              name: 'Sigmoid Kernel',
              formula: 'K(x, z) = tanh(α xᵀz + c)',
              desc: 'Connects SVMs directly to 2-layer Neural Networks (multilayer perceptrons). Note: Only satisfies Mercer\'s condition for certain values of α and c.',
              badge: 'Neural Network Link',
              color: 'text-amber-400'
            }
          ].map(k => (
            <div key={k.name} className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/50 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
              <div className="flex items-center justify-between mb-1.5">
                <span className={`font-semibold text-sm ${k.color}`}>{k.name}</span>
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                  theme === 'dark' ? 'bg-white/10 text-gray-300' : 'bg-gray-200 text-gray-700'
                }`}>{k.badge}</span>
              </div>
              <p className="font-mono text-xs text-gray-400 mb-2">{k.formula}</p>
              <p className={`text-xs ${BODY}`}>{k.desc}</p>
            </div>
          ))}
        </div>

        <p className={`${BODY} mb-4`}>
          Test all four kernels in real-time on challenging non-linear datasets below. Adjust <code className="font-mono">γ</code> and degree <code className="font-mono">d</code> to explore boundary flexibility:
        </p>

        <KernelZooCanvas theme={theme} />

        <DeepDive title="Why does the RBF Kernel map to INFINITE dimensions?">
          <p className={`text-sm ${BODY} mb-2`}>
            Consider the 1D Gaussian kernel <code className="font-mono">K(x, z) = exp(−(x − z)²)</code>. Expanding the squared difference:
          </p>
          <div className={`p-3 rounded-lg font-mono text-xs mb-2 ${theme === 'dark' ? 'bg-slate-950 text-cyan-300' : 'bg-slate-100 text-cyan-800'}`}>
            {'K(x, z) = exp(-x²) · exp(-z²) · exp(2xz)'}
          </div>
          <p className={`text-sm ${BODY} mb-2`}>
            Using the infinite Taylor series expansion <code className="font-mono">{'exp(u) = Σ (uᵏ / k!)'}</code>:
          </p>
          <div className={`p-3 rounded-lg font-mono text-xs mb-2 ${theme === 'dark' ? 'bg-slate-950 text-cyan-300' : 'bg-slate-100 text-cyan-800'}`}>
            {'exp(2xz) = 1 + 2xz + (4x²z²)/2! + (8x³z³)/3! + ... + (2ᵏ xᵏ zᵏ)/k! + ...'}
          </div>
          <p className={`text-sm ${BODY}`}>
            The feature map <code className="font-mono">ϕ(x)</code> contains every power of x from degree 0 to infinity: <code className="font-mono">{'ϕ(x) = exp(-x²) · [1, √(2)x, √(2/2!)x², ..., √(2ᵏ/k!)xᵏ, ...]'}</code>! A single scalar evaluation of <code className="font-mono">K(x, z)</code> implicitly computes the inner product between two vectors of <strong>infinite length</strong>!
          </p>
        </DeepDive>
      </div>

      {/* 2.5 Real-world Applications: Structured and Unstructured */}
      <div className={S}>
        <p className={LBL}>2.5 — Real-World Applications</p>
        <h2 className={H2}>Where SVMs dominate in industry</h2>
        <p className={`${BODY} mb-4`}>
          While deep neural networks receive immense attention, Support Vector Machines remain a vital, high-performance tool in production pipelines, particularly when data is structured, high-dimensional, or limited in sample size.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">📄</span>
              <h3 className={`font-semibold text-sm ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                Structured Data: Text &amp; NLP
              </h3>
            </div>
            <ul className={`space-y-2 text-xs ${BODY}`}>
              <li>
                <strong>Spam &amp; Fraud Detection:</strong> In TF-IDF or Bag-of-Words representations, vocabulary sizes easily exceed 50,000 features. Text data is inherently sparse and already linearly separable. A <em>Linear SVM (LinearSVC)</em> trains in seconds and avoids overfitting.
              </li>
              <li>
                <strong>Document Categorization:</strong> Legal discovery, medical record indexing, and news sentiment classification frequently utilize Linear SVMs due to their deterministic reproducibility and resistance to high dimensionality.
              </li>
              <li>
                <strong>Bioinformatics &amp; Genomics:</strong> String kernels and mismatch kernels applied to DNA sequences classify gene expressions and identify protein fold families directly from raw nucleotide strings.
              </li>
            </ul>
          </div>

          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">🖼️</span>
              <h3 className={`font-semibold text-sm ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                Unstructured Data: Vision &amp; Audio
              </h3>
            </div>
            <ul className={`space-y-2 text-xs ${BODY}`}>
              <li>
                <strong>Pedestrian Detection (HOG + SVM):</strong> Dalal &amp; Triggs (2005) revolutionized computer vision by coupling Histogram of Oriented Gradients (HOG) with a linear SVM. This formed the backbone of automotive collision avoidance systems for over a decade.
              </li>
              <li>
                <strong>Face Recognition (Eigenfaces + SVM):</strong> Projecting facial images onto Principal Component vectors (PCA) followed by an RBF SVM classifier achieves rapid, highly accurate verification on small labeled datasets.
              </li>
              <li>
                <strong>Audio Event Detection:</strong> Mel-frequency cepstral coefficients (MFCCs) extracted from audio spectrograms fed into an RBF SVM classify acoustic scenes (glass breaking, vehicle sirens, speech presence).
              </li>
            </ul>
          </div>
        </div>

        <Callout type="success" title="SVM vs Deep Learning: When to use which?">
          • <strong>Use SVM when:</strong> Training samples are scarce (hundreds to thousands), features are high-dimensional (text, genomics), full mathematical guarantees are required, or edge devices require low-latency CPU inference.<br />
          • <strong>Use Deep Learning when:</strong> Data is massive (hundreds of thousands+ samples), end-to-end feature learning from raw pixels or waveforms is needed, and GPU compute is readily available.
        </Callout>
      </div>

      {/* Python Code Section */}
      <div className={S}>
        <p className={LBL}>Python Scikit-Learn Implementations</p>
        <h2 className={H2}>Production workflows with scikit-learn</h2>
        <p className={`${BODY} mb-3`}>
          1. <strong>Hard vs Soft Margin:</strong> Comparing margin widths and support vector counts:
        </p>
        <CodeBlock code={PYTHON_CODE_HARD_SOFT} />

        <p className={`${BODY} mt-6 mb-3`}>
          2. <strong>Kernel Comparison &amp; GridSearchCV:</strong> Tuning C and gamma for RBF kernels:
        </p>
        <CodeBlock code={PYTHON_CODE_KERNELS} />

        <p className={`${BODY} mt-6 mb-3`}>
          3. <strong>Text Classification Pipeline:</strong> TF-IDF with LinearSVC for fast spam detection:
        </p>
        <CodeBlock code={PYTHON_CODE_TEXT_IMAGE} />
      </div>

      {/* Quiz Section */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${
          theme === 'dark' ? 'bg-indigo-500/10 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'
        }`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
              10 comprehensive questions covering hard-margin, dual formulation, KKT conditions, slack variables, and kernels • +100 XP
            </p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="svm" />
      </div>
    </motion.div>
  )
}
