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
const PYTHON_CODE_VOTING = `from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Three diverse base learners
clf1 = LogisticRegression(random_state=42)
clf2 = DecisionTreeClassifier(max_depth=4, random_state=42)
clf3 = SVC(probability=True, random_state=42)

# Soft Voting: averages predicted class probabilities (often outperforms hard voting)
voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)],
    voting='soft'
)

for name, clf in [('Logistic Regression', clf1), ('Decision Tree', clf2),
                  ('SVM (RBF)', clf3), ('Voting Ensemble', voting_clf)]:
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"{name:22s} Test Accuracy: {acc:.2%}")`

const PYTHON_CODE_BAGGING_OOB = `from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Bagging with Out-of-Bag (OOB) evaluation
# With replacement sampling, ~36.8% of samples are never picked per tree.
# These act as an automatic, free validation set!
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train, y_train)
print(f"Bagging OOB Score:   {bagging.oob_score_:.2%}")
print(f"Bagging Test Score:  {bagging.score(X_test, y_test):.2%}")

# 2. Random Forest: Bagging + Random Feature Subsets (max_features='sqrt')
# Subsampling features de-correlates the individual trees!
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"Random Forest OOB:   {rf.oob_score_:.2%}")
print(f"Random Forest Test:  {rf.score(X_test, y_test):.2%}")`

const PYTHON_CODE_FEATURE_IMPORTANCE = `import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 1. Mean Decrease Impurity (MDI / Gini Importance)
# Fast, computed during tree building. Can be biased toward high-cardinality features.
print("--- Gini Impurity Importance (MDI) ---")
for name, imp in zip(feature_names, rf.feature_importances_):
    print(f"{name:20s}: {imp:.4f}")

# 2. Permutation Feature Importance (MDA)
# Model-agnostic and robust: shuffles one feature at a time and measures accuracy drop.
perm_imp = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
print("\\n--- Permutation Feature Importance (MDA) ---")
for name, imp in zip(feature_names, perm_imp.importances_mean):
    print(f"{name:20s}: {imp:.4f}")`

const QUIZ_QUESTIONS = [
  {
    question: 'According to Condorcet\'s Jury Theorem, what condition MUST be met for an ensemble of classifiers to outperform any single classifier?',
    options: [
      'The classifiers must all use the same learning algorithm',
      'Each base classifier must perform strictly better than random guessing (p > 0.5) and their errors must be independent or uncorrelated',
      'The number of classifiers must be an even number',
      'The ensemble must be trained on a single continuous GPU thread'
    ],
    correct: 1,
    explanation: 'If each individual classifier has an accuracy p > 0.5 and their errors are independent, the probability that the majority vote is correct approaches 1.0 as the number of classifiers M increases.'
  },
  {
    question: 'What is the theoretical probability that a specific training sample is NEVER selected in a bootstrap sample of size N drawn with replacement?',
    options: [
      '1 / N',
      '50%',
      '(1 - 1/N)^N ≈ 1/e ≈ 36.8%',
      '0% (all samples are always picked)'
    ],
    correct: 2,
    explanation: 'The probability of picking a sample on one draw is 1/N, so the probability of not picking it is (1 - 1/N). Across N independent draws, the probability is (1 - 1/N)^N, which converges to 1/e ≈ 36.8% as N grows.'
  },
  {
    question: 'What is Out-Of-Bag (OOB) error estimation in Bagging and Random Forests?',
    options: [
      'An error metric calculated exclusively on test samples outside the training domain',
      'Evaluating each base estimator on the ~36.8% of training samples that were left out of its bootstrap sample, providing free validation without a separate holdout set',
      'The error incurred when trees exceed the maximum allowed depth',
      'A penalty term added to the loss function to prevent deep trees'
    ],
    correct: 1,
    explanation: 'For every training sample, we aggregate predictions only from the trees that did not include it in their bootstrap draw. The resulting OOB score mirrors cross-validation accuracy without requiring separate CV folds.'
  },
  {
    question: 'How does a Random Forest de-correlate individual decision trees compared to standard Bagging?',
    options: [
      'By training trees on completely disjoint subsets of the training data',
      'By restricting each split to consider only a random subset of features (typically √d for classification), preventing strong features from dominating all tree roots',
      'By converting all continuous features into binary flags',
      'By using gradient descent instead of greedy splitting'
    ],
    correct: 1,
    explanation: 'Standard bagging trees often pick the same dominant feature for top splits, producing highly correlated trees. Random Forest forces trees to consider only m ≈ √d features at each split, creating genuinely diverse perspectives.'
  },
  {
    question: 'What is the primary difference between Soft Voting and Hard Voting in an ensemble classifier?',
    options: [
      'Hard voting uses deep trees; soft voting uses shallow trees',
      'Hard voting tallies discrete class label predictions (simple majority), while soft voting averages predicted class probabilities',
      'Soft voting only works for regression problems',
      'Hard voting requires GPU memory allocation'
    ],
    correct: 1,
    explanation: 'Hard voting counts class votes (mode). Soft voting computes the weighted average of predicted class probability distributions, giving confident predictions more weight and typically yielding superior accuracy.'
  },
  {
    question: 'How does Stacking (Stacked Generalization) differ from simple Voting?',
    options: [
      'Stacking combines models by training a meta-learner (e.g. Logistic Regression) on the out-of-fold predictions of base models, rather than using a static voting rule',
      'Stacking only uses decision trees',
      'Stacking averages feature values rather than predictions',
      'Stacking eliminates the need for training labels'
    ],
    correct: 0,
    explanation: 'Stacking uses base learners to make predictions, and feeds those predictions as input features into a secondary meta-classifier that learns the optimal way to blend them together.'
  },
  {
    question: 'What is a known pitfall of Mean Decrease Impurity (MDI / Gini Importance) in Random Forests?',
    options: [
      'It cannot be computed for classification problems',
      'It is artificially biased toward high-cardinality numerical or categorical features that provide many split opportunities, even if they lack true predictive signal',
      'It requires cross-validation to compute',
      'It is always negative'
    ],
    correct: 1,
    explanation: 'MDI counts impurity reduction every time a feature is used in a split. Features with many unique values (e.g. IDs, high-cardinality categories) have more opportunities to split and inflate their MDI score artificially.'
  },
  {
    question: 'In terms of the Bias-Variance tradeoff, what is the primary benefit of Bagging (Bootstrap Aggregation)?',
    options: [
      'It dramatically reduces the bias of low-variance models',
      'It significantly reduces the variance of high-variance, low-bias models (like unpruned decision trees) without noticeably increasing bias',
      'It eliminates both bias and variance simultaneously down to zero',
      'It converts non-linear classifiers into linear models'
    ],
    correct: 1,
    explanation: 'A single deep decision tree has low bias but high variance (overfits). Averaging M independent identically distributed trees reduces variance by a factor of 1/M while leaving the expectation (bias) virtually unchanged.'
  },
  {
    question: 'How does Boosting fundamentally differ from Bagging?',
    options: [
      'Boosting trains trees in parallel; Bagging trains them sequentially',
      'Bagging trains diverse trees in parallel to reduce variance; Boosting trains shallow trees sequentially, where each new tree focuses on the errors of previous trees to reduce bias',
      'Boosting does not use decision trees',
      'Bagging only works on text data'
    ],
    correct: 1,
    explanation: 'Bagging reduces variance by averaging parallel, independent, deep trees. Boosting reduces bias by sequentially chaining weak learners that reweight or compute residuals on previous mistakes.'
  },
  {
    question: 'For classification with d features, what is the standard rule of thumb for the number of features considered at each split in a Random Forest?',
    options: [
      'd / 2',
      '√d (square root of d)',
      'd²',
      'log₂(d)'
    ],
    correct: 1,
    explanation: 'For classification, Breiman recommended m = ⌊√d⌋ features randomly sampled per split. For regression, the typical default is m = ⌊d / 3⌋.'
  }
]

// ── Interactive Component 1: Wisdom of the Crowd & Condorcet Simulation ───────
function CondorcetJuryDemo({ theme }) {
  const [baseAcc, setBaseAcc] = useState(0.65) // individual accuracy p
  const [numClassifiers, setNumClassifiers] = useState(15) // M odd: 1..41
  const W = 460
  const H = 220

  // Calculate Condorcet majority probability:
  // P_maj = sum_{k = ceil(M/2)}^M C(M, k) * p^k * (1 - p)^(M - k)
  const ensembleAcc = useMemo(() => {
    const M = numClassifiers
    const p = baseAcc
    const q = 1 - p
    const majority = Math.floor(M / 2) + 1

    // Helper: log binomial coefficient
    const logFactorial = (n) => {
      let r = 0
      for (let i = 2; i <= n; i++) r += Math.log(i)
      return r
    }

    let sumProb = 0
    for (let k = majority; k <= M; k++) {
      const logComb = logFactorial(M) - logFactorial(k) - logFactorial(M - k)
      const logP = logComb + k * Math.log(p) + (M - k) * Math.log(q)
      sumProb += Math.exp(logP)
    }
    return Math.min(1.0, Math.max(0.0, sumProb))
  }, [baseAcc, numClassifiers])

  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'

    // Plot Condorcet Curve: Ensemble Accuracy as a function of Individual Accuracy p in [0, 1]
    const padL = 40, padR = 20, padT = 20, padB = 30
    const plotW = W - padL - padR
    const plotH = H - padT - padB

    const toX = (p) => padL + p * plotW
    const toY = (val) => padT + (1 - val) * plotH

    // Grid & Diagonal Reference
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
    ctx.lineWidth = 1
    for (let p = 0; p <= 1; p += 0.25) {
      ctx.beginPath()
      ctx.moveTo(toX(p), padT)
      ctx.lineTo(toX(p), padT + plotH)
      ctx.stroke()

      ctx.beginPath()
      ctx.moveTo(padL, toY(p))
      ctx.lineTo(padL + plotW, toY(p))
      ctx.stroke()
    }

    // Baseline 45-degree diagonal (Single Classifier)
    ctx.beginPath()
    ctx.moveTo(toX(0), toY(0))
    ctx.lineTo(toX(1), toY(1))
    ctx.strokeStyle = dark ? 'rgba(148, 163, 184, 0.4)' : 'rgba(100, 116, 139, 0.5)'
    ctx.lineWidth = 1.5
    ctx.setLineDash([4, 4])
    ctx.stroke()
    ctx.setLineDash([])

    // Plot S-Curve for chosen M
    const logFactorial = (n) => {
      let r = 0
      for (let i = 2; i <= n; i++) r += Math.log(i)
      return r
    }
    const calcEnsemble = (p, M) => {
      if (p <= 0) return 0
      if (p >= 1) return 1
      const q = 1 - p
      const majority = Math.floor(M / 2) + 1
      let sumProb = 0
      for (let k = majority; k <= M; k++) {
        const logComb = logFactorial(M) - logFactorial(k) - logFactorial(M - k)
        const logP = logComb + k * Math.log(p) + (M - k) * Math.log(q)
        sumProb += Math.exp(logP)
      }
      return Math.min(1.0, Math.max(0.0, sumProb))
    }

    ctx.beginPath()
    ctx.strokeStyle = '#6366F1'
    ctx.lineWidth = 2.5
    const steps = 60
    for (let i = 0; i <= steps; i++) {
      const p = i / steps
      const ens = calcEnsemble(p, numClassifiers)
      const cx = toX(p)
      const cy = toY(ens)
      if (i === 0) ctx.moveTo(cx, cy)
      else ctx.lineTo(cx, cy)
    }
    ctx.stroke()

    // Current operating point dot
    const curX = toX(baseAcc)
    const curY = toY(ensembleAcc)
    ctx.beginPath()
    ctx.arc(curX, curY, 6, 0, Math.PI * 2)
    ctx.fillStyle = '#10B981'
    ctx.fill()
    ctx.strokeStyle = dark ? '#0F172A' : '#FFFFFF'
    ctx.lineWidth = 2
    ctx.stroke()

    // Threshold indicator line at p = 0.5
    const midX = toX(0.5)
    ctx.beginPath()
    ctx.moveTo(midX, padT)
    ctx.lineTo(midX, padT + plotH)
    ctx.strokeStyle = dark ? 'rgba(239, 68, 68, 0.4)' : 'rgba(239, 68, 68, 0.5)'
    ctx.lineWidth = 1
    ctx.setLineDash([2, 3])
    ctx.stroke()
    ctx.setLineDash([])

    // Axis labels
    ctx.font = '10px monospace'
    ctx.fillStyle = dark ? '#94A3B8' : '#64748B'
    ctx.textAlign = 'center'
    ctx.fillText('0.0', toX(0), padT + plotH + 15)
    ctx.fillText('0.5', toX(0.5), padT + plotH + 15)
    ctx.fillText('1.0', toX(1.0), padT + plotH + 15)
    ctx.fillText('Individual Accuracy p', W / 2, H - 4)

    ctx.textAlign = 'right'
    ctx.fillText('1.0', padL - 6, toY(1) + 4)
    ctx.fillText('0.5', padL - 6, toY(0.5) + 4)
    ctx.fillText('0.0', padL - 6, toY(0) + 4)
  }, [baseAcc, numClassifiers, ensembleAcc, theme, W, H])

  return (
    <div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              Individual Classifier Accuracy (p): <span className="font-mono font-bold text-emerald-400">{(baseAcc * 100).toFixed(0)}%</span>
            </span>
          </div>
          <input
            type="range"
            min="0.3"
            max="0.95"
            step="0.02"
            value={baseAcc}
            onChange={e => setBaseAcc(parseFloat(e.target.value))}
            className="w-full accent-emerald-500 cursor-pointer"
          />
          <span className={`text-[11px] block mt-1 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
            {baseAcc > 0.5
              ? '✓ Better than random coin toss (Ensemble boosts performance)'
              : '⚠️ Worse than random guessing (Ensemble magnifies errors to 0%!)'}
          </span>
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              Ensemble Size (M Classifiers): <span className="font-mono font-bold text-indigo-400">{numClassifiers}</span>
            </span>
          </div>
          <input
            type="range"
            min="1"
            max="45"
            step="2"
            value={numClassifiers}
            onChange={e => setNumClassifiers(parseInt(e.target.value, 10))}
            className="w-full accent-indigo-500 cursor-pointer"
          />
          <span className={`text-[11px] block mt-1 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
            Always use an odd number of voters to avoid tie-breakers.
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
        <div className={`col-span-2 rounded-2xl border overflow-hidden relative ${
          theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
        }`}>
          <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
        </div>

        <div className={`rounded-2xl border p-4 flex flex-col justify-between ${
          theme === 'dark' ? 'bg-slate-900/90 border-white/10' : 'bg-white border-gray-200'
        }`}>
          <div>
            <h4 className={`text-xs font-bold uppercase tracking-wider mb-2 ${
              theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'
            }`}>
              Condorcet Outcome
            </h4>

            <div className="space-y-3 mt-3">
              <div className="flex justify-between text-xs py-1 border-b border-inherit">
                <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Solo Model</span>
                <span className="font-mono text-gray-400">{(baseAcc * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between text-xs py-1 border-b border-inherit">
                <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Ensemble Majority</span>
                <span className="font-mono font-bold text-lg text-emerald-400">{(ensembleAcc * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between text-xs py-1">
                <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Ensemble Gain</span>
                <span className={`font-mono font-bold ${ensembleAcc >= baseAcc ? 'text-emerald-400' : 'text-red-400'}`}>
                  {ensembleAcc >= baseAcc ? '+' : ''}{((ensembleAcc - baseAcc) * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          <div className={`p-2.5 rounded-lg text-[11px] font-mono mt-3 ${
            theme === 'dark' ? 'bg-indigo-500/10 text-indigo-300 border border-indigo-500/20' : 'bg-indigo-50 text-indigo-800 border border-indigo-200'
          }`}>
            P(Majority) = Σ C(M, k) pᵏ (1−p)ᴹ⁻ᵏ
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Interactive Component 2: Bootstrap Aggregation (Bagging) & OOB Explorer ───
function BaggingOobDemo({ theme }) {
  const [seed, setSeed] = useState(1)
  const N = 16 // dataset of 16 items

  // Draw bootstrap sample of size N with replacement
  const { sampleCounts, oobIndices } = useMemo(() => {
    const rand = makePRNG(seed * 777 + 123)
    const counts = new Array(N).fill(0)

    for (let i = 0; i < N; i++) {
      const idx = Math.floor(rand() * N)
      counts[idx]++
    }

    const oob = []
    counts.forEach((c, idx) => {
      if (c === 0) oob.push(idx)
    })

    return { sampleCounts: counts, oobIndices: oob }
  }, [seed, N])

  const oobFraction = (oobIndices.length / N) * 100

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <h4 className={`text-xs font-bold uppercase tracking-wider ${theme === 'dark' ? 'text-purple-400' : 'text-purple-600'}`}>
            Bootstrap Draw Simulation (N = {N})
          </h4>
          <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            Click below to draw a fresh bootstrap sample with replacement:
          </p>
        </div>

        <button
          onClick={() => setSeed(s => s + 1)}
          className="px-4 py-2 rounded-xl bg-purple-500 hover:bg-purple-600 text-white text-xs font-semibold shadow-md transition-colors"
        >
          🎲 Draw New Bootstrap Sample
        </button>
      </div>

      <div className={`p-5 rounded-2xl border mb-4 ${
        theme === 'dark' ? 'bg-slate-900/90 border-white/10' : 'bg-white border-gray-200'
      }`}>
        <p className={`text-xs font-semibold mb-3 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
          Original Dataset Population (Sample Frequency in current draw):
        </p>

        <div className="grid grid-cols-4 sm:grid-cols-8 gap-2 mb-4">
          {sampleCounts.map((count, idx) => {
            const isOob = count === 0
            return (
              <div
                key={idx}
                className={`p-2.5 rounded-xl border flex flex-col items-center justify-center transition-all ${
                  isOob
                    ? 'border-amber-500/40 bg-amber-500/10 text-amber-300'
                    : count === 1
                    ? 'border-indigo-500/40 bg-indigo-500/10 text-indigo-300'
                    : 'border-emerald-500/40 bg-emerald-500/15 text-emerald-300 font-bold'
                }`}
              >
                <span className="text-xs font-mono">D{idx + 1}</span>
                <span className="text-[10px] mt-1">
                  {isOob ? 'OOB (0×)' : `${count}×`}
                </span>
              </div>
            )
          })}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs pt-3 border-t border-inherit">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-amber-400 shrink-0" />
            <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
              Out-of-Bag (OOB): <strong className="text-amber-400 font-mono">{oobIndices.length} / {N} ({oobFraction.toFixed(1)}%)</strong>
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-indigo-400 shrink-0" />
            <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
              Picked once: <strong className="font-mono text-indigo-400">{sampleCounts.filter(c => c === 1).length}</strong>
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-emerald-400 shrink-0" />
            <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
              Duplicate draws: <strong className="font-mono text-emerald-400">{sampleCounts.filter(c => c > 1).length}</strong>
            </span>
          </div>
        </div>
      </div>

      <Callout type="formula" mono="lim_{N → ∞} (1 - 1/N)^N = 1/e ≈ 36.7879%">
        Notice that in almost every sample, roughly <strong>one-third (~36.8%)</strong> of data points are never picked! Bagging harnesses these unselected samples as an automatic out-of-bag validation set, granting free cross-validation on every bootstrap round.
      </Callout>
    </div>
  )
}

// ── Interactive Component 3: Single Tree vs Random Forest Boundary ─────────────
function ForestBoundaryDemo({ theme }) {
  const canvasRef = useRef(null)
  const [modelType, setModelType] = useState('forest') // 'single' or 'forest'
  const [numTrees, setNumTrees] = useState(15)
  const [featureSubsample, setFeatureSubsample] = useState(true) // sqrt(d) vs all
  const W = 460
  const H = 260

  // Two Moons synthetic dataset
  const dataset = useMemo(() => {
    const rand = makePRNG(13579)
    const pts = []
    // Moon 1 (Class +1: Indigo)
    for (let i = 0; i < 28; i++) {
      const a = (i / 28) * Math.PI
      const r = 70
      pts.push({
        x: W / 2 - 45 + r * Math.cos(a) + (rand() - 0.5) * 18,
        y: H / 2 - 20 - r * Math.sin(a) + (rand() - 0.5) * 18,
        label: 1
      })
    }
    // Moon 2 (Class -1: Emerald)
    for (let i = 0; i < 28; i++) {
      const a = (i / 28) * Math.PI
      const r = 70
      pts.push({
        x: W / 2 + 35 - r * Math.cos(a) + (rand() - 0.5) * 18,
        y: H / 2 + 15 + r * Math.sin(a) + (rand() - 0.5) * 18,
        label: -1
      })
    }
    return pts
  }, [W, H])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'

    // Simulate simple orthogonal tree thresholds on normalized coordinates
    const rand = makePRNG(9999)
    const trees = []
    const treeCount = modelType === 'single' ? 1 : numTrees

    for (let t = 0; t < treeCount; t++) {
      // Each tree generates 3-4 decision stump splits
      const splits = []
      for (let s = 0; s < 4; s++) {
        // Random axis-aligned split
        const useX = featureSubsample ? (rand() > 0.5) : (s % 2 === 0)
        const threshold = useX ? W / 2 + (rand() - 0.5) * 120 : H / 2 + (rand() - 0.5) * 100
        const sign = rand() > 0.5 ? 1 : -1
        splits.push({ useX, threshold, sign })
      }
      trees.push(splits)
    }

    // Rasterize decision boundary on grid
    const step = 8
    for (let x = 0; x < W; x += step) {
      for (let y = 0; y < H; y += step) {
        let votes = 0
        trees.forEach(splits => {
          let score = 0
          splits.forEach(sp => {
            const val = sp.useX ? x : y
            if ((val > sp.threshold && sp.sign > 0) || (val <= sp.threshold && sp.sign < 0)) {
              score += 1
            } else {
              score -= 1
            }
          })
          votes += score > 0 ? 1 : -1
        })

        // Background color
        if (votes >= 0) {
          ctx.fillStyle = dark
            ? `rgba(99, 102, 241, ${Math.min(0.25, 0.08 + (votes / treeCount) * 0.15)})`
            : `rgba(99, 102, 241, ${Math.min(0.22, 0.06 + (votes / treeCount) * 0.14)})`
        } else {
          ctx.fillStyle = dark
            ? `rgba(16, 185, 129, ${Math.min(0.25, 0.08 + (-votes / treeCount) * 0.15)})`
            : `rgba(16, 185, 129, ${Math.min(0.22, 0.06 + (-votes / treeCount) * 0.14)})`
        }
        ctx.fillRect(x, y, step, step)
      }
    }

    // Render Data Points
    dataset.forEach(p => {
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
    ctx.fillStyle = dark ? '#94A3B8' : '#64748B'
    ctx.fillText(
      modelType === 'single'
        ? 'Single Tree: Jagged, axis-aligned, high variance'
        : `Random Forest (${numTrees} trees): Smooth, averaged boundary`,
      12,
      48
    )
  }, [dataset, modelType, numTrees, featureSubsample, theme, W, H])

  return (
    <div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        <div>
          <label className={`text-xs font-semibold block mb-1 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Model Architecture:
          </label>
          <div className="flex gap-2">
            <button
              onClick={() => setModelType('single')}
              className={`flex-1 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                modelType === 'single'
                  ? 'bg-amber-500 text-white'
                  : theme === 'dark' ? 'bg-slate-800 text-gray-400' : 'bg-gray-100 text-gray-600'
              }`}
            >
              1 Tree (Single)
            </button>
            <button
              onClick={() => setModelType('forest')}
              className={`flex-1 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                modelType === 'forest'
                  ? 'bg-indigo-500 text-white'
                  : theme === 'dark' ? 'bg-slate-800 text-gray-400' : 'bg-gray-100 text-gray-600'
              }`}
            >
              Random Forest
            </button>
          </div>
        </div>

        {modelType === 'forest' && (
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                Ensemble Trees: <span className="text-indigo-400 font-mono font-bold">{numTrees}</span>
              </span>
            </div>
            <input
              type="range"
              min="3"
              max="35"
              step="2"
              value={numTrees}
              onChange={e => setNumTrees(parseInt(e.target.value, 10))}
              className="w-full accent-indigo-500 cursor-pointer"
            />
          </div>
        )}

        <div className="flex items-center gap-2 pt-4">
          <input
            type="checkbox"
            id="featSub"
            checked={featureSubsample}
            onChange={e => setFeatureSubsample(e.target.checked)}
            className="rounded accent-purple-500 cursor-pointer"
          />
          <label htmlFor="featSub" className={`text-xs font-medium cursor-pointer ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Subsample Features (m = √d)
          </label>
        </div>
      </div>

      <div className={`rounded-2xl border overflow-hidden relative mb-2 ${
        theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
      }`}>
        <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
      </div>

      <p className={`text-xs text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        Compare the rigid blocky staircases of a Single Decision Tree against the smooth, softened transitions of the Random Forest ensemble!
      </p>
    </div>
  )
}

// ── Main Page Component ───────────────────────────────────────────────────────
export default function EnsembleLearning() {
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
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/20 border border-emerald-500/30 mb-4"
        >
          <span className="text-xs text-emerald-400 font-medium">Machine Learning • Ensemble Methods &amp; Bagging</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Ensemble Learning – I <span className="gradient-text">(Bagging)</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          No single model possesses universal omniscience. Ensemble learning combines dozens or hundreds of diverse individual models to produce a collective prediction that is dramatically <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>more accurate, robust, and resistant to variance</strong> than any single constituent.
        </p>

        <Callout type="analogy" title="Analogy: Who Wants to Be a Millionaire?">
          When a contestant gets stuck on a tough question, they can &ldquo;Phone-a-Friend&rdquo; (single expert model) or &ldquo;Ask the Audience&rdquo; (ensemble model). While individual audience members might know little, their collective majority vote is right over <strong>90% of the time</strong>! That is the mathematical magic of Condorcet&apos;s Jury Theorem.
        </Callout>
      </div>

      {/* Core Concepts */}
      <TheoryBlock
        title="Ensemble Architecture Pillars"
        cards={[
          {
            icon: '🗳️',
            title: 'Wisdom of the Crowd',
            body: 'When diverse, independent models have an individual accuracy p > 0.5, aggregating their votes drives the error rate exponentially toward zero.',
            mono: 'Var(Ensemble) = Var(Single) / M'
          },
          {
            icon: '🛍️',
            title: 'Bootstrap Aggregation',
            body: 'Bagging trains M base models on M different bootstrap samples drawn with replacement. Averages predictions to conquer high variance.',
            mono: 'D_bootstrap ~ Uniform(D)'
          },
          {
            icon: '🎯',
            title: 'Out-Of-Bag (OOB) Score',
            body: 'Sampling with replacement leaves approximately 36.8% (1/e) of data unselected per tree. These act as free, unbiased validation samples!',
            mono: '(1 - 1/N)^N → 1/e ≈ 36.8%'
          },
          {
            icon: '🌲',
            title: 'Random Forest',
            body: 'De-correlates bagging trees by randomly subsampling features (m = √d) at every split, preventing one strong feature from hijacking all tree roots.',
            mono: 'max_features = sqrt(d)'
          },
          {
            icon: '📊',
            title: 'Feature Importance',
            body: 'Quantifies feature influence via Mean Decrease Impurity (MDI / Gini) or Permutation Importance (MDA / accuracy loss after shuffling).',
            mono: 'MDI vs Permutation MDA'
          },
          {
            icon: '⚡',
            title: 'Bagging vs Boosting',
            body: 'Bagging trains deep, independent trees in parallel to slash variance. Boosting chains shallow trees sequentially to systematically eliminate bias.',
            mono: 'Parallel (Var) vs Sequential (Bias)'
          }
        ]}
      />

      {/* ========================================================================= */}
      {/* SECTION 1: COMBINING CLASSIFIERS */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>
          Part 1 — Aggregation Theory
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Combining Classifiers &amp; Voting Schemes
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Mathematical foundations of Condorcet&apos;s jury theorem, soft voting, and stacking.
        </p>
      </div>

      <div className={S}>
        <p className={LBL}>1.1 — Condorcet&apos;s Jury Theorem</p>
        <h2 className={H2}>Why groups outperform individuals</h2>
        <p className={`${BODY} mb-4`}>
          Formulated by Marquis de Condorcet in 1785, the theorem states: If an ensemble consists of <code className="font-mono">M</code> independent voters, each having an independent probability <code className="font-mono">p &gt; 0.5</code> of making the correct decision:
        </p>

        <Callout type="formula" mono="P(Majority Correct) = Σ_{k=⌈M/2⌉}^M  (M choose k) · pᵏ (1−p)^{M−k}">
          As the ensemble size <code className="font-mono">M → ∞</code>, the probability of the majority vote being correct approaches <strong>100%</strong>! Conversely, if <code className="font-mono">p &lt; 0.5</code>, the ensemble accuracy cascades to 0%.
        </Callout>

        <p className={`${BODY} mb-4`}>
          Interactive Simulation: Adjust individual accuracy <code className="font-mono">p</code> and ensemble size <code className="font-mono">M</code> to watch the S-curve steepen:
        </p>

        <CondorcetJuryDemo theme={theme} />
      </div>

      <div className={S}>
        <p className={LBL}>1.2 — Voting Schemes &amp; Stacking</p>
        <h2 className={H2}>Hard Voting, Soft Voting &amp; Stacked Generalization</h2>
        <p className={`${BODY} mb-4`}>
          How should multiple diverse predictions be synthesized into a final verdict?
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          {[
            {
              name: 'Hard Voting (Majority)',
              desc: 'Every model casts a discrete vote for a class. The class with the highest vote count (mode) wins. Ignores probability confidence scores.',
              badge: 'Discrete Mode',
              color: 'text-indigo-400'
            },
            {
              name: 'Soft Voting (Probabilistic)',
              desc: 'Computes the weighted average of predicted class probabilities across all models. Gives high-confidence predictions more sway.',
              badge: 'Highest Accuracy',
              color: 'text-purple-400'
            },
            {
              name: 'Stacking (Meta-Learning)',
              desc: 'Trains a secondary meta-classifier (e.g. Logistic Regression) on the out-of-fold predictions of base learners to learn optimal weighting.',
              badge: 'Kaggle Winner',
              color: 'text-emerald-400'
            }
          ].map(m => (
            <div key={m.name} className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
              <div className="flex items-center justify-between mb-1.5">
                <span className={`font-semibold text-xs ${m.color}`}>{m.name}</span>
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                  theme === 'dark' ? 'bg-white/10 text-gray-300' : 'bg-gray-200 text-gray-700'
                }`}>{m.badge}</span>
              </div>
              <p className={`text-xs ${BODY}`}>{m.desc}</p>
            </div>
          ))}
        </div>

        <DeepDive title="How Stacking Prevents Data Leakage (Out-of-Fold Splits)">
          <p className={`text-sm ${BODY} mb-2`}>
            If you train the meta-classifier on the predictions made by base models on their own training data, the meta-model will overfit to whichever base model memorized the training set.
          </p>
          <p className={`text-sm ${BODY}`}>
            <strong>The Solution (K-Fold Blending):</strong> Partition the dataset into K folds. Train base learners on K-1 folds, predict on the held-out fold, and assemble a full dataset of out-of-fold predictions. The meta-learner is then trained strictly on these unseen predictions!
          </p>
        </DeepDive>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 2: BAGGING AND RANDOM FOREST */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-purple-500/5 border-purple-500/20' : 'bg-purple-50 border-purple-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-purple-400' : 'text-purple-600'}`}>
          Part 2 — Bagging &amp; Tree Ensembles
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Bootstrap Aggregation &amp; Random Forest
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Sampling with replacement, Out-of-Bag validation, and tree de-correlation.
        </p>
      </div>

      <div className={S}>
        <p className={LBL}>2.1 — Bootstrap Aggregation &amp; OOB Error</p>
        <h2 className={H2}>Leo Breiman&apos;s Bagging breakthrough (1996)</h2>
        <p className={`${BODY} mb-4`}>
          A single unpruned decision tree is a low-bias, high-variance learner — it memorizes training noise and splits hair-thin rectangles. <strong>Bagging (Bootstrap Aggregation)</strong> resolves this by training <code className="font-mono">M</code> deep trees on independent bootstrap samples drawn with replacement:
        </p>

        <BaggingOobDemo theme={theme} />
      </div>

      <div className={S}>
        <p className={LBL}>2.2 — Random Forest &amp; Feature Subsampling</p>
        <h2 className={H2}>De-correlating trees with random feature subsets</h2>
        <p className={`${BODY} mb-4`}>
          If your dataset has one dominant feature (e.g. tumor radius), all standard bagging trees will choose that feature as their root split. As a consequence, the trees will be strongly correlated with each other!
        </p>

        <Callout type="info" title="The Variance of Correlated Trees">
          The variance of the average of <code className="font-mono">M</code> trees with correlation <code className="font-mono">ρ</code> is:
          <br />
          <code className="font-mono font-semibold">Var(Ensemble) = ρ·σ² + (1 - ρ)/M · σ²</code>
          <br /><br />
          Even if <code className="font-mono">M → ∞</code>, the second term vanishes, but the first term <code className="font-mono">ρ·σ²</code> remains! To slash ensemble variance, we MUST reduce the tree correlation <code className="font-mono">ρ</code>!
        </Callout>

        <p className={`${BODY} mb-4`}>
          <strong>Random Forest</strong> reduces <code className="font-mono">ρ</code> by forcing each tree split to randomly sample only <code className="font-mono">m = ⌊√d⌋</code> candidate features. Observe the smoother boundary below:
        </p>

        <ForestBoundaryDemo theme={theme} />
      </div>

      <div className={S}>
        <p className={LBL}>2.3 — Feature Importance &amp; Boosting Preview</p>
        <h2 className={H2}>Measuring feature influence &amp; transitioning to Boosting</h2>
        <p className={`${BODY} mb-4`}>
          Random Forests offer built-in interpretability by quantifying which features contribute most to reducing impurity:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <h3 className={`font-semibold text-sm mb-1.5 ${theme === 'dark' ? 'text-indigo-300' : 'text-indigo-700'}`}>
              Mean Decrease Impurity (MDI / Gini)
            </h3>
            <p className={`text-xs ${BODY} mb-2`}>
              Aggregates the total weighted reduction in Gini impurity brought by feature <code className="font-mono">j</code> across all tree splits.
            </p>
            <span className="text-[11px] text-amber-400 font-medium">
              ⚠️ Warning: Can artificially inflate scores for high-cardinality features.
            </span>
          </div>

          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <h3 className={`font-semibold text-sm mb-1.5 ${theme === 'dark' ? 'text-emerald-300' : 'text-emerald-700'}`}>
              Permutation Importance (MDA)
            </h3>
            <p className={`text-xs ${BODY} mb-2`}>
              Randomly shuffles values of feature <code className="font-mono">j</code> in the validation set and records the resulting drop in model accuracy.
            </p>
            <span className="text-[11px] text-emerald-400 font-medium">
              ✓ Gold Standard: Model-agnostic, immune to cardinality bias.
            </span>
          </div>
        </div>

        <Callout type="success" title="Preview: Bagging vs Boosting">
          • <strong>Bagging (Random Forest):</strong> Trains trees <em>in parallel</em>. Each tree is deep and unconstrained. Focuses on <strong>variance reduction</strong>.<br />
          • <strong>Boosting (AdaBoost, XGBoost, LightGBM):</strong> Trains trees <em>sequentially</em>. Each tree is a shallow weak learner that corrects the residual mistakes of the previous tree. Focuses on <strong>bias reduction</strong>!
        </Callout>
      </div>

      {/* Python Code Section */}
      <div className={S}>
        <p className={LBL}>Python Scikit-Learn Implementations</p>
        <h2 className={H2}>Production workflows with scikit-learn</h2>
        <p className={`${BODY} mb-3`}>
          1. <strong>Voting Classifier (Soft vs Hard Voting):</strong>
        </p>
        <CodeBlock code={PYTHON_CODE_VOTING} />

        <p className={`${BODY} mt-6 mb-3`}>
          2. <strong>Bagging &amp; Random Forest with Out-of-Bag (OOB) Evaluation:</strong>
        </p>
        <CodeBlock code={PYTHON_CODE_BAGGING_OOB} />

        <p className={`${BODY} mt-6 mb-3`}>
          3. <strong>Feature Importance: MDI Gini vs Permutation Importance:</strong>
        </p>
        <CodeBlock code={PYTHON_CODE_FEATURE_IMPORTANCE} />
      </div>

      {/* Quiz Section */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${
          theme === 'dark' ? 'bg-emerald-500/10 border-emerald-500/20' : 'bg-emerald-50 border-emerald-200'
        }`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
              10 questions covering Condorcet theorem, voting, OOB error, tree de-correlation, and feature importance • +100 XP
            </p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="ensemble-learning" />
      </div>
    </motion.div>
  )
}
