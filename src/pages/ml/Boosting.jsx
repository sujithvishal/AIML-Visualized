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
const PYTHON_CODE_ADABOOST = `from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=600, n_features=10, n_informative=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# AdaBoost using Decision Stumps (depth-1 trees) as weak learners
# Evaluates exponential loss: L(y, f) = exp(-y * f)
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME',
    random_state=42
)
adaboost.fit(X_train, y_train)

print(f"AdaBoost Test Accuracy: {adaboost.score(X_test, y_test):.2%}")
print(f"Number of estimators: {len(adaboost.estimators_)}")
# Individual weak learner weights alpha_t
print(f"Top 3 stump weights (alpha): {adaboost.estimator_weights_[:3].round(3)}")`

const PYTHON_CODE_GBM_XGBOOST = `from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Scikit-learn Gradient Boosting Machine (GBM)
# Minimizes deviance (log-loss) via functional gradient descent on pseudo-residuals
gbm = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,    # Shrinkage parameter eta
    max_depth=3,           # Shallow trees (weak learners)
    subsample=0.8,         # Stochastic gradient boosting (row subsampling)
    random_state=42
)
gbm.fit(X_train, y_train)
y_pred_gbm = gbm.predict_proba(X_test)[:, 1]
print(f"GBM Test ROC-AUC: {roc_auc_score(y_test, y_pred_gbm):.4f}")

# 2. XGBoost (Extreme Gradient Boosting)
# Regularized loss with 2nd-order Taylor expansion (gradients g_i & hessians h_i)
# Column subsampling, sparsity awareness, and cache-aware parallel split finding
try:
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        reg_lambda=1.0,     # L2 leaf weight regularization
        gamma=0.1,          # Minimum loss reduction required for split
        colsample_bytree=0.8,# Column subsampling
        eval_metric='logloss',
        random_state=42
    )
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict_proba(X_test)[:, 1]
    print(f"XGBoost Test ROC-AUC: {roc_auc_score(y_test, y_pred_xgb):.4f}")
except ImportError:
    print("XGBoost package optional in environment.")`

const QUIZ_QUESTIONS = [
  {
    question: 'How does AdaBoost adapt its focus from one round to the next?',
    options: [
      'It randomly drops 50% of the features at every iteration',
      'It increases the weights of misclassified training samples and decreases the weights of correctly classified samples, forcing the next weak learner to target hard cases',
      'It modifies the depth of the tree dynamically',
      'It switches from classification to regression'
    ],
    correct: 1,
    explanation: 'AdaBoost multiplies the weight of each misclassified sample by exp(α_t) and normalizes, ensuring the subsequent weak learner prioritizes previously difficult instances.'
  },
  {
    question: 'In AdaBoost, the importance weight α_t assigned to weak learner h_t with error ε_t is given by:',
    options: [
      'α_t = 1 - ε_t',
      'α_t = ½ ln((1 - ε_t) / ε_t)',
      'α_t = 1 / ε_t²',
      'α_t = exp(-ε_t)'
    ],
    correct: 1,
    explanation: 'When error ε_t is low (e.g. 0.1), α_t is large and positive, granting the learner high authority. When ε_t = 0.5 (random guess), α_t = 0 (zero weight).'
  },
  {
    question: 'What is the surrogate loss function that AdaBoost implicitly minimizes stage by stage?',
    options: [
      'Mean Squared Error (MSE)',
      'Hinge Loss',
      'Exponential Loss: L(y, f) = exp(-y · f(x))',
      'Cross-Entropy Loss'
    ],
    correct: 2,
    explanation: 'AdaBoost is equivalent to forward stage-wise additive modeling minimizing the exponential loss L(y, f) = exp(-y · f(x)).'
  },
  {
    question: 'What is the primary vulnerability of AdaBoost compared to other ensemble methods?',
    options: [
      'It cannot handle binary classification',
      'It is highly sensitive to noisy data and outliers because exponential weighting exponentially inflates the importance of corrupted labels',
      'It requires deep 20-level decision trees',
      'It cannot run on multi-core CPUs'
    ],
    correct: 1,
    explanation: 'Because misclassified points are boosted exponentially, outliers and mislabeled samples quickly dominate the sample weights, derailing subsequent weak learners.'
  },
  {
    question: 'What are the targets that each new tree in a Gradient Boosting Machine (GBM) is trained to fit?',
    options: [
      'The original raw target labels y',
      'The negative gradients (pseudo-residuals) of the loss function with respect to current model predictions',
      'Random bootstrap subsamples of features',
      'The principal components of the design matrix'
    ],
    correct: 1,
    explanation: 'Gradient boosting treats function optimization as gradient descent in function space. Each new base learner fits the negative gradient (pseudo-residual) r_im = -[∂L/∂f]. Under squared loss, this equals the simple residual y - f(x).'
  },
  {
    question: 'What is the purpose of Shrinkage (Learning Rate η) in Gradient Boosting?',
    options: [
      'To prune trees that exceed memory limits',
      'To scale down the contribution of each newly added tree (f_m = f_{m-1} + η · h_m), slowing down learning and leaving room for future trees to improve generalization',
      'To convert multi-class predictions into binary scores',
      'To normalize feature ranges between 0 and 1'
    ],
    correct: 1,
    explanation: 'Shrinkage acts as a form of regularization. Setting a small learning rate (e.g. η = 0.05 or 0.1) requires more trees but drastically reduces overfitting and improves test set accuracy.'
  },
  {
    question: 'How does XGBoost optimize the loss function compared to traditional Gradient Boosting (GBM)?',
    options: [
      'XGBoost uses a 2nd-order Taylor expansion utilizing both 1st-order gradients (g_i) and 2nd-order hessians (h_i), plus explicit tree complexity regularization',
      'XGBoost only supports linear models',
      'XGBoost computes exact matrix inversions',
      'XGBoost replaces decision trees with neural networks'
    ],
    correct: 0,
    explanation: 'While standard GBM relies only on first-order gradients, XGBoost approximates the objective using a 2nd-order Taylor polynomial, incorporating hessians h_i = ∂²L/∂f² for much faster, more accurate convergence.'
  },
  {
    question: 'In the XGBoost optimal leaf weight formula w_j* = -Σ g_i / (Σ h_i + λ), what role does λ play?',
    options: [
      'The maximum number of leaves in the tree',
      'L2 regularization penalty on leaf weights, preventing extreme predictions and shrinking weights toward zero',
      'The learning rate multiplier',
      'The fraction of features dropped during column subsampling'
    ],
    correct: 1,
    explanation: 'λ is the L2 regularization parameter on leaf scores. A larger λ penalizes large leaf weights, stabilizing trees and protecting against overfitting on sparse leaf nodes.'
  },
  {
    question: 'What is the &ldquo;Sparsity-Aware Split Finding&rdquo; feature in XGBoost?',
    options: [
      'It discards all columns that contain null values',
      'It assigns a learned default split direction (left or right) for missing/zero values during training, handling missing data with zero imputation overhead',
      'It enforces that 90% of model weights are strictly zero',
      'It only trains on dense floating-point matrices'
    ],
    correct: 1,
    explanation: 'XGBoost visits only non-missing entries during split finding and automatically assigns missing values to whichever branch yields the highest gain, enabling seamless handling of missing data.'
  },
  {
    question: 'How does Column (Feature) Subsampling in XGBoost benefit model performance?',
    options: [
      'It speeds up tree construction and prevents dominant features from overpowering trees, mirroring Random Forest de-correlation',
      'It eliminates the need for computing gradients',
      'It guarantees 100% training accuracy',
      'It converts categorical variables to one-hot encoding'
    ],
    correct: 0,
    explanation: 'Column subsampling (both per-tree and per-split) speeds up training and provides additional variance reduction by preventing a few highly correlated features from dictating every tree split.'
  }
]

// ── Interactive Component 1: Step-by-Step AdaBoost Sample Reweighting ─────────
function AdaBoostStepperCanvas({ theme }) {
  const [round, setRound] = useState(1) // rounds 1..5
  const W = 460
  const H = 260

  // 10 toy samples in 2D with known labels (+1 or -1)
  const initialData = useMemo(() => [
    { id: 1, x: 70,  y: 60,  label: 1 },
    { id: 2, x: 110, y: 100, label: 1 },
    { id: 3, x: 80,  y: 160, label: 1 },
    { id: 4, x: 140, y: 70,  label: 1 },
    { id: 5, x: 150, y: 170, label: 1 },
    // Hard positive point in negative territory
    { id: 6, x: 280, y: 70,  label: 1 },

    { id: 7, x: 330, y: 190, label: -1 },
    { id: 8, x: 370, y: 120, label: -1 },
    { id: 9, x: 260, y: 210, label: -1 },
    { id: 10,x: 390, y: 220, label: -1 },
    { id: 11,x: 310, y: 150, label: -1 },
    // Hard negative point in positive territory
    { id: 12,x: 180, y: 130, label: -1 }
  ], [])

  // 5 predetermined decision stumps:
  // stump: { axis: 'x'|'y', threshold, sign: 1|-1 }
  const stumpSequence = useMemo(() => [
    { axis: 'x', threshold: 210, sign: 1 },  // Round 1: Vertical line at x=210
    { axis: 'y', threshold: 105, sign: -1 }, // Round 2: Horizontal line at y=105
    { axis: 'x', threshold: 160, sign: 1 },  // Round 3: Vertical line at x=160
    { axis: 'y', threshold: 150, sign: -1 }, // Round 4: Horizontal line at y=150
    { axis: 'x', threshold: 300, sign: 1 },  // Round 5: Vertical line at x=300
  ], [])

  // Calculate weights evolution up to current round
  const state = useMemo(() => {
    let weights = new Array(initialData.length).fill(1 / initialData.length)
    const history = []

    for (let r = 0; r < round; r++) {
      const stump = stumpSequence[r]
      // Predictions of stump: +1 if (val > thresh * sign), else -1
      const preds = initialData.map(p => {
        const val = stump.axis === 'x' ? p.x : p.y
        if (stump.sign === 1) return val < stump.threshold ? 1 : -1
        return val < stump.threshold ? -1 : 1
      })

      // Weighted error epsilon
      let eps = 0
      initialData.forEach((p, idx) => {
        if (preds[idx] !== p.label) eps += weights[idx]
      })
      eps = Math.max(0.01, Math.min(0.49, eps))

      // Stump importance alpha
      const alpha = 0.5 * Math.log((1 - eps) / eps)

      // Update sample weights
      let zSum = 0
      const newWeights = weights.map((w, idx) => {
        const isWrong = preds[idx] !== initialData[idx].label
        const nw = w * Math.exp(isWrong ? alpha : -alpha)
        zSum += nw
        return nw
      })

      // Normalize
      weights = newWeights.map(nw => nw / zSum)

      history.push({ stump, eps, alpha, preds, weights })
    }

    return {
      currentWeights: history[round - 1].weights,
      currentStump: history[round - 1].stump,
      currentEps: history[round - 1].eps,
      currentAlpha: history[round - 1].alpha,
      history
    }
  }, [initialData, stumpSequence, round])

  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'
    const { currentWeights, currentStump, history } = state

    // Rasterize cumulative ensemble prediction: H(x) = sign(sum alpha_t * h_t(x))
    const step = 8
    for (let x = 0; x < W; x += step) {
      for (let y = 0; y < H; y += step) {
        let ensembleScore = 0
        history.forEach(h => {
          const val = h.stump.axis === 'x' ? x : y
          const pred = h.stump.sign === 1
            ? (val < h.stump.threshold ? 1 : -1)
            : (val < h.stump.threshold ? -1 : 1)
          ensembleScore += h.alpha * pred
        })

        if (ensembleScore >= 0) {
          ctx.fillStyle = dark
            ? `rgba(99, 102, 241, ${Math.min(0.24, 0.06 + Math.abs(ensembleScore) * 0.05)})`
            : `rgba(99, 102, 241, ${Math.min(0.2, 0.05 + Math.abs(ensembleScore) * 0.04)})`
        } else {
          ctx.fillStyle = dark
            ? `rgba(16, 185, 129, ${Math.min(0.24, 0.06 + Math.abs(ensembleScore) * 0.05)})`
            : `rgba(16, 185, 129, ${Math.min(0.2, 0.05 + Math.abs(ensembleScore) * 0.04)})`
        }
        ctx.fillRect(x, y, step, step)
      }
    }

    // Draw current weak learner decision line (stump boundary)
    ctx.beginPath()
    ctx.strokeStyle = '#F59E0B'
    ctx.lineWidth = 2.5
    ctx.setLineDash([5, 4])
    if (currentStump.axis === 'x') {
      ctx.moveTo(currentStump.threshold, 0)
      ctx.lineTo(currentStump.threshold, H)
    } else {
      ctx.moveTo(0, currentStump.threshold)
      ctx.lineTo(W, currentStump.threshold)
    }
    ctx.stroke()
    ctx.setLineDash([])

    // Render Data Points sized by sample weight w_i!
    initialData.forEach((p, idx) => {
      const w = currentWeights[idx]
      // Radius scaled by weight (base 4px, up to 16px)
      const radius = Math.max(5, Math.min(18, 4 + w * 70))

      ctx.beginPath()
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2)
      ctx.fillStyle = p.label === 1 ? '#6366F1' : '#10B981'
      ctx.fill()
      ctx.strokeStyle = dark ? '#0F172A' : '#FFFFFF'
      ctx.lineWidth = 2
      ctx.stroke()

      // Small weight label if boosted
      if (w > 0.12) {
        ctx.font = 'bold 9px monospace'
        ctx.fillStyle = '#F59E0B'
        ctx.fillText(`w=${(w * 100).toFixed(0)}%`, p.x + radius + 3, p.y - 4)
      }
    })

    // Legend
    ctx.font = '10px monospace'
    ctx.fillStyle = '#F59E0B'
    ctx.fillText(`Round ${round} Stump Boundary (α = ${state.currentAlpha.toFixed(2)})`, 12, 20)
    ctx.fillStyle = dark ? '#94A3B8' : '#64748B'
    ctx.fillText('Circle radius = Sample weight wᵢ (Misclassified points grow!)', 12, 34)
  }, [state, initialData, round, theme, W, H])

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Iteration Round:
          </span>
          {[1, 2, 3, 4, 5].map(r => (
            <button
              key={r}
              onClick={() => setRound(r)}
              className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                round === r
                  ? 'bg-amber-500 text-white shadow-sm'
                  : theme === 'dark' ? 'bg-slate-800 text-gray-400 hover:text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              t = {r}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3 text-xs">
          <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>
            Error ε: <strong className="font-mono text-red-400">{state.currentEps.toFixed(2)}</strong>
          </span>
          <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>
            Stump Weight α: <strong className="font-mono text-amber-400">{state.currentAlpha.toFixed(2)}</strong>
          </span>
        </div>
      </div>

      <div className={`rounded-2xl border overflow-hidden relative mb-2 ${
        theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
      }`}>
        <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
      </div>

      <p className={`text-xs text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        Step through rounds 1 to 5. Notice how hard misclassified samples balloon in size, compelling the next stump to position its split directly to correct them!
      </p>
    </div>
  )
}

// ── Interactive Component 2: Gradient Boosting Residual Fitting Simulator ─────
function GradientBoostingResidualDemo({ theme }) {
  const [numTrees, setNumTrees] = useState(4) // 1..15
  const [eta, setEta] = useState(0.3) // learning rate shrinkage
  const W = 460
  const H = 220

  // 1D non-linear ground truth function: y = sin(1.8 x) + 0.4*cos(3 x)
  const trueFn = (x) => Math.sin(1.8 * x) + 0.35 * Math.cos(3 * x)

  // Synthetic sample points
  const points = useMemo(() => {
    const rand = makePRNG(7890)
    const pts = []
    const N = 24
    for (let i = 0; i < N; i++) {
      const x = -2.8 + (i / (N - 1)) * 5.6
      const noise = (rand() - 0.5) * 0.25
      pts.push({ x, y: trueFn(x) + noise })
    }
    return pts
  }, [])

  // Fit sequential gradient boosting trees to residuals
  const model = useMemo(() => {
    // Initial constant prediction f_0 = mean(y)
    const yMean = points.reduce((acc, p) => acc + p.y, 0) / points.length

    // Trees are simple 1-split decision stumps fitting residuals
    const trees = []
    let currentPreds = points.map(() => yMean)

    for (let m = 0; m < numTrees; m++) {
      // 1. Calculate pseudo-residuals: r_i = y_i - f_{m-1}(x_i)
      const residuals = points.map((p, idx) => p.y - currentPreds[idx])

      // 2. Best single threshold split on x to fit residuals (least squares)
      let bestSplitX = 0
      let bestGammaL = 0
      let bestGammaR = 0
      let minSSE = Infinity

      for (let s = 1; s < points.length - 1; s++) {
        const splitX = (points[s].x + points[s + 1].x) / 2
        const leftRes = []
        const rightRes = []
        points.forEach((p, idx) => {
          if (p.x <= splitX) leftRes.push(residuals[idx])
          else rightRes.push(residuals[idx])
        })

        if (leftRes.length > 0 && rightRes.length > 0) {
          const gL = leftRes.reduce((a, b) => a + b, 0) / leftRes.length
          const gR = rightRes.reduce((a, b) => a + b, 0) / rightRes.length
          let sse = 0
          leftRes.forEach(r => (sse += (r - gL) ** 2))
          rightRes.forEach(r => (sse += (r - gR) ** 2))

          if (sse < minSSE) {
            minSSE = sse
            bestSplitX = splitX
            bestGammaL = gL
            bestGammaR = gR
          }
        }
      }

      trees.push({ splitX: bestSplitX, gammaL: bestGammaL, gammaR: bestGammaR })

      // 3. Update current predictions with shrinkage eta
      currentPreds = points.map((p, idx) => {
        const leafVal = p.x <= bestSplitX ? bestGammaL : bestGammaR
        return currentPreds[idx] + eta * leafVal
      })
    }

    // Function to evaluate model prediction at any continuous x
    const predict = (xVal) => {
      let val = yMean
      trees.forEach(t => {
        const leaf = xVal <= t.splitX ? t.gammaL : t.gammaR
        val += eta * leaf
      })
      return val
    }

    return { yMean, trees, predict, finalPreds: currentPreds }
  }, [points, numTrees, eta])

  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'
    const xMin = -3.2, xMax = 3.2
    const yMin = -1.8, yMax = 1.8

    const toCx = (x) => ((x - xMin) / (xMax - xMin)) * W
    const toCy = (y) => H - ((y - yMin) / (yMax - yMin)) * H

    // Draw grid
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
    ctx.lineWidth = 1
    for (let x = -3; x <= 3; x += 1) {
      ctx.beginPath()
      ctx.moveTo(toCx(x), 0)
      ctx.lineTo(toCx(x), H)
      ctx.stroke()
    }
    for (let y = -1; y <= 1; y += 1) {
      ctx.beginPath()
      ctx.moveTo(0, toCy(y))
      ctx.lineTo(W, toCy(y))
      ctx.stroke()
    }

    // Draw Smooth Underlying Truth (dashed)
    ctx.beginPath()
    ctx.strokeStyle = dark ? 'rgba(148, 163, 184, 0.4)' : 'rgba(100, 116, 139, 0.5)'
    ctx.lineWidth = 1.5
    ctx.setLineDash([4, 4])
    const steps = 100
    for (let i = 0; i <= steps; i++) {
      const x = xMin + (i / steps) * (xMax - xMin)
      const y = trueFn(x)
      if (i === 0) ctx.moveTo(toCx(x), toCy(y))
      else ctx.lineTo(toCx(x), toCy(y))
    }
    ctx.stroke()
    ctx.setLineDash([])

    // Draw Gradient Boosted Step Function Curve f_M(x)
    ctx.beginPath()
    ctx.strokeStyle = '#6366F1'
    ctx.lineWidth = 2.5
    for (let i = 0; i <= steps * 2; i++) {
      const x = xMin + (i / (steps * 2)) * (xMax - xMin)
      const y = model.predict(x)
      if (i === 0) ctx.moveTo(toCx(x), toCy(y))
      else ctx.lineTo(toCx(x), toCy(y))
    }
    ctx.stroke()

    // Draw Residual connecting lines from points to model prediction
    points.forEach((p, idx) => {
      const predY = model.finalPreds[idx]
      ctx.beginPath()
      ctx.moveTo(toCx(p.x), toCy(p.y))
      ctx.lineTo(toCx(p.x), toCy(predY))
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.45)'
      ctx.lineWidth = 1.2
      ctx.stroke()
    })

    // Draw Data Points
    points.forEach(p => {
      ctx.beginPath()
      ctx.arc(toCx(p.x), toCy(p.y), 4.5, 0, Math.PI * 2)
      ctx.fillStyle = '#10B981'
      ctx.fill()
      ctx.strokeStyle = dark ? '#0F172A' : '#FFFFFF'
      ctx.lineWidth = 1.5
      ctx.stroke()
    })

    // Legend
    ctx.font = '10px monospace'
    ctx.fillStyle = '#6366F1'
    ctx.fillText(`— GBM Prediction (M=${numTrees} trees, η=${eta})`, 12, 18)
    ctx.fillStyle = dark ? '#94A3B8' : '#64748B'
    ctx.fillText('--- Ground Truth Curve', 12, 32)
    ctx.fillStyle = '#EF4444'
    ctx.fillText('| Red Bars: Remaining Residuals (y - f(x))', 12, 46)
  }, [points, model, numTrees, eta, theme, W, H])

  return (
    <div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              Number of Boosted Trees (M): <span className="font-mono font-bold text-indigo-400">{numTrees}</span>
            </span>
          </div>
          <input
            type="range"
            min="1"
            max="15"
            step="1"
            value={numTrees}
            onChange={e => setNumTrees(parseInt(e.target.value, 10))}
            className="w-full accent-indigo-500 cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              Shrinkage / Learning Rate (η): <span className="font-mono font-bold text-cyan-400">{eta.toFixed(2)}</span>
            </span>
          </div>
          <input
            type="range"
            min="0.05"
            max="0.8"
            step="0.05"
            value={eta}
            onChange={e => setEta(parseFloat(e.target.value))}
            className="w-full accent-cyan-500 cursor-pointer"
          />
        </div>
      </div>

      <div className={`rounded-2xl border overflow-hidden relative mb-2 ${
        theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
      }`}>
        <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
      </div>

      <p className={`text-xs text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        Increase M to watch each sequential tree shave down the red residual spikes until the step curve hugs the true non-linear pattern!
      </p>
    </div>
  )
}

// ── Interactive Component 3: XGBoost 2nd-Order Gain & Regularization Inspector ─
function XgBoostSplitGainInspector({ theme }) {
  const [lambdaReg, setLambdaReg] = useState(1.0) // L2 regularization parameter
  const [gammaCost, setGammaCost] = useState(2.0) // tree split threshold cost gamma

  // Sample gradients and hessians on left and right candidate split:
  // Left partition: G_L = -14.2, H_L = 8.5
  // Right partition: G_R = +16.0, H_R = 9.0
  const GL = -14.2, HL = 8.5
  const GR = 16.0,  HR = 9.0

  // XGBoost Split Gain formula:
  // Gain = 1/2 * [ (GL^2 / (HL + lambda)) + (GR^2 / (HR + lambda)) - ((GL + GR)^2 / (HL + HR + lambda)) ] - gamma
  const scoreLeft = (GL * GL) / (HL + lambdaReg)
  const scoreRight = (GR * GR) / (HR + lambdaReg)
  const scoreParent = ((GL + GR) * (GL + GR)) / (HL + HR + lambdaReg)
  const rawGain = 0.5 * (scoreLeft + scoreRight - scoreParent)
  const netGain = rawGain - gammaCost

  // Optimal leaf weights w* = -G / (H + lambda)
  const optWeightL = -GL / (HL + lambdaReg)
  const optWeightR = -GR / (HR + lambdaReg)

  return (
    <div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              L2 Leaf Regularization (λ): <span className="font-mono text-purple-400 font-bold">{lambdaReg.toFixed(1)}</span>
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="10"
            step="0.5"
            value={lambdaReg}
            onChange={e => setLambdaReg(parseFloat(e.target.value))}
            className="w-full accent-purple-500 cursor-pointer"
          />
          <span className={`text-[11px] block mt-1 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
            Shrinks leaf weights w* to prevent overfitting on small leaf groups.
          </span>
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              Minimum Split Gain Penalty (γ): <span className="font-mono text-amber-400 font-bold">{gammaCost.toFixed(1)}</span>
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="15"
            step="0.5"
            value={gammaCost}
            onChange={e => setGammaCost(parseFloat(e.target.value))}
            className="w-full accent-amber-500 cursor-pointer"
          />
          <span className={`text-[11px] block mt-1 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
            Split is pruned away if net gain does not exceed γ.
          </span>
        </div>
      </div>

      <div className={`p-5 rounded-2xl border mb-3 ${
        theme === 'dark' ? 'bg-slate-900/90 border-white/10' : 'bg-white border-gray-200'
      }`}>
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-3 border-b border-inherit">
          <div>
            <span className="text-xs uppercase tracking-wider font-semibold text-indigo-400">Candidate Tree Split</span>
            <h4 className={`text-base font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
              2nd-Order Gain Evaluation
            </h4>
          </div>

          <div className={`px-3 py-1.5 rounded-xl border text-xs font-bold flex items-center gap-2 ${
            netGain > 0
              ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300'
              : 'bg-red-500/15 border-red-500/30 text-red-300'
          }`}>
            <span>{netGain > 0 ? '✓ Split Approved' : '✂️ Pruned (Gain < γ)'}</span>
            <span className="font-mono">Net: {netGain.toFixed(2)}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div className={`p-3 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/50 border-white/5' : 'bg-gray-50 border-gray-200'}`}>
            <span className="text-indigo-400 font-semibold block mb-1">Left Leaf Node</span>
            <p className="font-mono text-gray-400 text-[11px]">G_L = {GL}, H_L = {HL}</p>
            <p className="font-mono font-bold text-white mt-1">w_L* = {optWeightL.toFixed(3)}</p>
          </div>

          <div className={`p-3 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/50 border-white/5' : 'bg-gray-50 border-gray-200'}`}>
            <span className="text-emerald-400 font-semibold block mb-1">Right Leaf Node</span>
            <p className="font-mono text-gray-400 text-[11px]">G_R = +{GR}, H_R = {HR}</p>
            <p className="font-mono font-bold text-white mt-1">w_R* = {optWeightR.toFixed(3)}</p>
          </div>

          <div className={`p-3 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/50 border-white/5' : 'bg-gray-50 border-gray-200'}`}>
            <span className="text-amber-400 font-semibold block mb-1">Split Math</span>
            <p className="font-mono text-[11px] text-gray-400">Raw Gain = {rawGain.toFixed(2)}</p>
            <p className="font-mono text-[11px] text-gray-400">Cost Penalty γ = {gammaCost.toFixed(2)}</p>
          </div>
        </div>
      </div>

      <Callout type="formula" mono="Gain = ½ [ (G_L² / (H_L + λ)) + (G_R² / (H_R + λ)) - ((G_L + G_R)² / (H_L + H_R + λ)) ] - γ">
        XGBoost uses this closed-form formula to rapidly evaluate millions of potential splits across parallel threads, completely bypassing gradient descent at the leaf level!
      </Callout>
    </div>
  )
}

// ── Main Page Component ───────────────────────────────────────────────────────
export default function Boosting() {
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
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-500/20 border border-amber-500/30 mb-4"
        >
          <span className="text-xs text-amber-400 font-medium">Machine Learning • Sequential Ensembles &amp; Residual Optimization</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Ensemble Learning – II <span className="gradient-text">(Boosting)</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          While Bagging builds independent models in parallel to slash variance, <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>Boosting trains learners sequentially</strong>. Each new model acts as a specialist specifically tasked with correcting the mistakes of its predecessors, transforming collections of barely-better-than-chance weak learners into state-of-the-art predictive machines.
        </p>

        <Callout type="analogy" title="Analogy: The Math Tutoring Session">
          Imagine studying for an exam. In round 1, you take a practice test and get 70%. In round 2, you don&apos;t reread the whole book; you focus exclusively on the 30% of questions you got wrong! In round 3, you drill the sub-topics you still struggled with. That is <strong>Boosting</strong>: focusing compute power precisely where the current model is weak.
        </Callout>
      </div>

      {/* Core Concepts */}
      <TheoryBlock
        title="Boosting Architecture Pillars"
        cards={[
          {
            icon: '🔄',
            title: 'Sequential Learning',
            body: 'Unlike parallel Bagging, trees are added one by one: f_m(x) = f_{m-1}(x) + η·h_m(x). Each step reduces empirical loss along the steepest descent path.',
            mono: 'Sequential (Bias Reduction)'
          },
          {
            icon: '🎯',
            title: 'AdaBoost (Reweighting)',
            body: 'Focuses on mistakes by boosting sample weights exp(α_t) on misclassified instances. Solves forward stage-wise additive modeling on exponential loss.',
            mono: 'w_{t+1} ∝ w_t · exp(-α_t y h_t)'
          },
          {
            icon: '📉',
            title: 'Gradient Boosting (GBM)',
            body: 'Jerome Friedman\'s insight: fitting residuals is equivalent to performing gradient descent directly in function space against any differentiable loss function.',
            mono: 'r_im = -[∂L / ∂f]'
          },
          {
            icon: '⚡',
            title: 'XGBoost (2nd Order)',
            body: 'Tianqi Chen\'s system breakthrough: leverages 2nd-order Taylor expansions with gradients g_i and hessians h_i, coupled with L2 leaf regularization and hardware optimizations.',
            mono: 'w* = -Σ g_i / (Σ h_i + λ)'
          },
          {
            icon: '🧊',
            title: 'Shrinkage (η)',
            body: 'Multiplies every tree\'s output by learning rate 0 < η ≤ 1. Slows down learning to prevent early trees from monopolizing residual variance, yielding vastly better generalization.',
            mono: 'f_m = f_{m-1} + η · h_m'
          },
          {
            icon: '📊',
            title: 'Sparsity Awareness',
            body: 'XGBoost automatically learns a default path for missing values and zero counts, achieving zero imputation overhead in production tabular pipelines.',
            mono: 'Default split routing'
          }
        ]}
      />

      {/* ========================================================================= */}
      {/* SECTION 1: ADABOOST */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-amber-500/5 border-amber-500/20' : 'bg-amber-50 border-amber-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-amber-400' : 'text-amber-600'}`}>
          Part 1 — Adaptive Boosting
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          AdaBoost &amp; Exponential Loss
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Yoav Freund and Robert Schapire&apos;s breakthrough algorithm that solved the weak-learning problem.
        </p>
      </div>

      <div className={S}>
        <p className={LBL}>1.1 — The AdaBoost Algorithm Walkthrough</p>
        <h2 className={H2}>Sample reweighting and weak learner combination</h2>
        <p className={`${BODY} mb-4`}>
          AdaBoost begins with uniform sample weights <code className="font-mono">w_i = 1/N</code>. At each iteration <code className="font-mono">t</code>:
        </p>

        <TheoryBlock
          title="AdaBoost Iteration Steps"
          cards={[
            {
              icon: '1️⃣',
              title: 'Train Weak Learner',
              body: 'Fit a base classifier h_t(x) (typically a depth-1 Decision Stump) to the weighted dataset.',
              mono: 'h_t = argmin_h ε_t'
            },
            {
              icon: '2️⃣',
              title: 'Compute Weighted Error',
              body: 'Calculate the total weight of misclassified samples: ε_t = Σ_{i: h_t(x_i) ≠ y_i} w_i.',
              mono: 'ε_t = sum(w_i * I(misclassified))'
            },
            {
              icon: '3️⃣',
              title: 'Calculate Learner Authority α_t',
              body: 'Stump voting weight: α_t = ½ ln((1 - ε_t) / ε_t). A stump with 90% accuracy gets high weight; 50% gets 0.',
              mono: 'α_t = ½ ln((1 - ε_t) / ε_t)'
            },
            {
              icon: '4️⃣',
              title: 'Update Sample Weights',
              body: 'Multiply weights of misclassified points by exp(α_t) and correctly classified by exp(-α_t). Normalize by Z_t.',
              mono: 'w_{i, t+1} = (w_{i, t} · exp(-α_t y_i h_t(x_i))) / Z_t'
            }
          ]}
        />

        <p className={`${BODY} mb-4`}>
          Interactive Stepper: Advance round-by-round from <code className="font-mono">t = 1</code> to <code className="font-mono">5</code> to see points resize dynamically as weights shift:
        </p>

        <AdaBoostStepperCanvas theme={theme} />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <h3 className="font-semibold text-xs text-emerald-400 mb-1">AdaBoost Strengths</h3>
            <ul className={`list-disc list-inside space-y-1 text-xs ${BODY}`}>
              <li>Requires minimal hyperparameter tuning.</li>
              <li>Weak learners are fast to compute (depth-1 decision stumps).</li>
              <li>Surprisingly resistant to overfitting when data is clean.</li>
            </ul>
          </div>

          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <h3 className="font-semibold text-xs text-red-400 mb-1">AdaBoost Weaknesses</h3>
            <ul className={`list-disc list-inside space-y-1 text-xs ${BODY}`}>
              <li>Extremely sensitive to noise and outliers!</li>
              <li>Exponential loss inflates misclassified noise samples to astronomical weights.</li>
              <li>Cannot easily optimize non-exponential loss functions.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 2: GRADIENT BOOSTING & XGBOOST */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-cyan-500/5 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}`}>
          Part 2 — Gradient Boosting &amp; Extreme Gradient Boosting
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          GBM &amp; XGBoost Architecture
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Gradient descent in function space, 2nd-order Taylor expansions, and production scalability.
        </p>
      </div>

      <div className={S}>
        <p className={LBL}>2.1 — Gradient Boosting (GBM) &amp; Residual Fitting</p>
        <h2 className={H2}>Gradient descent in function space</h2>
        <p className={`${BODY} mb-4`}>
          In 1999, Jerome Friedman revolutionized boosting by showing that instead of reweighting samples, one can fit each new tree <code className="font-mono">h_m(x)</code> to the <strong>negative gradient of the loss function</strong> (the pseudo-residuals):
        </p>

        <Callout type="formula" mono="r_im = - [ ∂L(y_i, f(x_i)) / ∂f(x_i) ]_{f = f_{m-1}}">
          For Mean Squared Error <code className="font-mono">L = ½ (y - f)²</code>, the negative gradient is simply the classic residual <code className="font-mono">r_i = y_i - f(x_i)</code>!
        </Callout>

        <p className={`${BODY} mb-4`}>
          Interactive Simulator: Adjust the number of trees <code className="font-mono">M</code> and learning rate <code className="font-mono">η</code> to watch sequential trees shave away residual errors:
        </p>

        <GradientBoostingResidualDemo theme={theme} />

        <DeepDive title="Why Shrinkage (Learning Rate η) is Non-Negotiable">
          <p className={`text-sm ${BODY} mb-2`}>
            If you set <code className="font-mono">η = 1.0</code>, the first few trees greedily devour the majority of variance, often creating overfitted, brittle decision regions.
          </p>
          <p className={`text-sm ${BODY}`}>
            By introducing shrinkage <code className="font-mono">{'f_m(x) = f_{m-1}(x) + η · h_m(x)'}</code> with <code className="font-mono">0.05 ≤ η ≤ 0.1</code>, we force the ensemble to take tiny, measured steps along the gradient path. This leaves room for subsequent trees to discover orthogonal feature relationships.
          </p>
        </DeepDive>
      </div>

      <div className={S}>
        <p className={LBL}>2.2 — XGBoost (Extreme Gradient Boosting)</p>
        <h2 className={H2}>2nd-Order Taylor expansion &amp; regularized split gain</h2>
        <p className={`${BODY} mb-4`}>
          Developed by Tianqi Chen in 2016, <strong>XGBoost</strong> became the dominant algorithm on Kaggle and in enterprise tabular pipelines. Instead of only using 1st-order gradients, XGBoost takes a <strong>2nd-order Taylor approximation</strong> of the loss function:
        </p>

        <Callout type="formula" mono="Obj ≈ Σᵢ [ g_i · f_t(x_i) + ½ h_i · f_t²(x_i) ] + γ T + ½ λ Σⱼ w_j²">
          where <code className="font-mono">g_i = ∂_f L</code> (gradient), <code className="font-mono">h_i = ∂²_f L</code> (hessian), <code className="font-mono">T</code> is number of leaves, and <code className="font-mono">λ</code> is L2 regularization on leaf weights <code className="font-mono">w_j</code>.
        </Callout>

        <p className={`${BODY} mb-4`}>
          Interactive Split Inspector: Test how tuning <code className="font-mono">λ</code> and <code className="font-mono">γ</code> determines whether candidate tree splits are approved or pruned:
        </p>

        <XgBoostSplitGainInspector theme={theme} />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">
          {[
            {
              name: 'Column Subsampling',
              desc: 'Selects a random subset of features per tree or per split, curbing correlation and speeding up training.',
              badge: 'Tree De-correlation',
              color: 'text-indigo-400'
            },
            {
              name: 'Sparsity Awareness',
              desc: 'Learns a default branch direction for missing values and zero counts, eliminating data imputation steps.',
              badge: 'Zero Imputation',
              color: 'text-purple-400'
            },
            {
              name: 'Cache-Aware & Parallel',
              desc: 'Organizes data into pre-sorted in-memory columnar blocks, maximizing CPU L1/L2 cache hit rates.',
              badge: 'Hardware Optimized',
              color: 'text-cyan-400'
            }
          ].map(f => (
            <div key={f.name} className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
              <div className="flex items-center justify-between mb-1.5">
                <span className={`font-semibold text-xs ${f.color}`}>{f.name}</span>
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                  theme === 'dark' ? 'bg-white/10 text-gray-300' : 'bg-gray-200 text-gray-700'
                }`}>{f.badge}</span>
              </div>
              <p className={`text-xs ${BODY}`}>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Python Code Section */}
      <div className={S}>
        <p className={LBL}>Python Scikit-Learn &amp; XGBoost Implementations</p>
        <h2 className={H2}>Production workflows in Python</h2>
        <p className={`${BODY} mb-3`}>
          1. <strong>AdaBoost with Decision Stumps:</strong>
        </p>
        <CodeBlock code={PYTHON_CODE_ADABOOST} />

        <p className={`${BODY} mt-6 mb-3`}>
          2. <strong>Gradient Boosting (GBM) vs XGBoost with ROC-AUC evaluation:</strong>
        </p>
        <CodeBlock code={PYTHON_CODE_GBM_XGBOOST} />
      </div>

      {/* Quiz Section */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${
          theme === 'dark' ? 'bg-amber-500/10 border-amber-500/20' : 'bg-amber-50 border-amber-200'
        }`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
              10 comprehensive questions covering AdaBoost, exponential loss, GBM pseudo-residuals, shrinkage, and XGBoost • +100 XP
            </p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="boosting" />
      </div>
    </motion.div>
  )
}
