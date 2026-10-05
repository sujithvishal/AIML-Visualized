import { useState, useRef, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

// Deterministic PRNG for canvas animations and point seeding
function makePRNG(seed = 42) {
  let s = seed
  return () => {
    s = (s * 16807 + 11) % 2147483647
    return (s - 1) / 2147483646
  }
}

// ── Python Code Examples ────────────────────────────────────────────────────────
const PYTHON_CODE_MLE_MAP = `import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Coin toss experiment: 8 Heads, 2 Tails
n_heads, n_tails = 8, 2
n_trials = n_heads + n_tails

# 1. Maximum Likelihood Estimation (MLE)
# Likelihood L(theta) = theta^H * (1 - theta)^T
# argmax log L(theta) -> theta_mle = H / (H + T)
theta_mle = n_heads / n_trials
print(f"MLE Estimate: {theta_mle:.3f} (overfits to small sample)")

# 2. Maximum A Posteriori (MAP)
# Prior: Beta(alpha=5, beta=5) reflecting belief the coin is roughly fair
alpha_prior, beta_prior = 5, 5
# Posterior: Beta(alpha + H, beta + T)
# Mode of Beta(a, b) = (a - 1) / (a + b - 2)
alpha_post = alpha_prior + n_heads
beta_post = beta_prior + n_tails
theta_map = (alpha_post - 1) / (alpha_post + beta_post - 2)
print(f"MAP Estimate: {theta_map:.3f} (pulled toward 0.5 by prior)")

# 3. Regularization connection:
# MAP with Gaussian prior on weights == L2 Ridge Regularization!
# log P(w|D) = log P(D|w) - (lambda / 2) * ||w||^2 + const`

const PYTHON_CODE_NAIVE_BAYES = `from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Text Classification with Multinomial Naive Bayes & Laplace Smoothing
corpus = [
    ("Win real money cash prize click now", 1),
    ("Claim free lottery reward and jackpot", 1),
    ("Exclusive luxury prize winner notification", 1),
    ("Weekly team project sync on Thursday morning", 0),
    ("Please review the updated quarterly budget draft", 0),
    ("Can we reschedule the design demo to 3pm?", 0),
]
texts, labels = zip(*corpus)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# alpha=1.0 specifies Laplace Add-One Smoothing
# Solves the zero-frequency problem: P(word|class) = (count + alpha) / (total + alpha * V)
nb_classifier = MultinomialNB(alpha=1.0)
nb_classifier.fit(X, labels)

# Test on a new message with both known and previously unseen words
test_email = ["Review project demo and claim prize"]
X_test = vectorizer.transform(test_email)
pred = nb_classifier.predict(X_test)[0]
probs = nb_classifier.predict_proba(X_test)[0]

print(f"Prediction: {'SPAM' if pred == 1 else 'HAM'}")
print(f"P(HAM) = {probs[0]:.2%}, P(SPAM) = {probs[1]:.2%}")`

const PYTHON_CODE_BAYESIAN_REGRESSION = `import numpy as np
from sklearn.linear_model import BayesianRidge

# Toy dataset: noisy linear trend
np.random.seed(42)
X = np.sort(np.random.uniform(-3, 3, 20)).reshape(-1, 1)
y = 1.8 * X.ravel() + 0.5 + np.random.normal(0, 0.4, 20)

# Bayesian Ridge Regression treats weights as random variables:
# Prior: w ~ N(0, alpha^-1 * I)
# Likelihood: y ~ N(Xw, lambda^-1 * I)
bayes_reg = BayesianRidge(compute_score=True)
bayes_reg.fit(X, y)

# Predict over fine grid and obtain both mean and predictive standard deviation!
X_grid = np.linspace(-4, 4, 100).reshape(-1, 1)
y_mean, y_std = bayes_reg.predict(X_grid, return_std=True)

print(f"Posterior mean slope w: {bayes_reg.coef_[0]:.3f}")
print(f"Estimated noise variance: {1.0 / bayes_reg.alpha_:.3f}")
print(f"Uncertainty at x=0 (dense data):  +-{2*y_std[50]:.3f} (95% CI)")
print(f"Uncertainty at x=4 (extrapolation): +-{2*y_std[-1]:.3f} (blooms wide!)")`

const QUIZ_QUESTIONS = [
  {
    question: 'What is the fundamental difference between Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP)?',
    options: [
      'MLE requires cross-validation, while MAP does not',
      'MAP incorporates a prior probability distribution over parameters, while MLE assumes all parameter values are equally likely a priori',
      'MLE handles non-linear models, while MAP only works for linear models',
      'MAP maximizes the likelihood function directly without calculating probabilities'
    ],
    correct: 1,
    explanation: 'MLE maximizes P(Data|θ), while MAP maximizes the posterior P(θ|Data) ∝ P(Data|θ) · P(θ). MAP incorporates prior knowledge P(θ) to prevent overfitting.'
  },
  {
    question: 'Under a Gaussian prior w ~ N(0, σ²I), the MAP estimate for linear regression is mathematically equivalent to:',
    options: [
      'L1 Lasso Regularization',
      'L2 Ridge Regularization (Weight Decay)',
      'Unregularized Ordinary Least Squares',
      'Logistic Regression'
    ],
    correct: 1,
    explanation: 'Taking the negative logarithm of the Gaussian prior produces -log P(w) = (1 / 2σ²) ||w||² + const. Adding this to the negative log-likelihood recovers the exact L2 Ridge loss penalty!'
  },
  {
    question: 'What is the core assumption of the Naïve Bayes classifier that gives it its "naïve" moniker?',
    options: [
      'All features follow a uniform distribution',
      'Features are conditionally independent given the class label',
      'Training data contains zero measurement errors',
      'Classes are always perfectly balanced'
    ],
    correct: 1,
    explanation: 'Naïve Bayes assumes that given the class y, each feature xᵢ is conditionally independent of any other feature: P(x₁, ..., x_d | y) = ∏ P(xᵢ | y). While often violated in real life, it works surprisingly well in practice.'
  },
  {
    question: 'Why is Laplace (additive) smoothing essential in Naïve Bayes classification?',
    options: [
      'To prevent the learning rate from vanishing',
      'To solve the zero-frequency problem where an unseen feature zeroes out the entire product of class probabilities',
      'To invert non-invertible covariance matrices',
      'To convert continuous features into discrete categories'
    ],
    correct: 1,
    explanation: 'Because Naïve Bayes multiplies feature probabilities together, if a word was never observed in a class during training (count = 0), P(word|class) = 0. Multiplying by 0 wipes out all other evidence! Laplace smoothing adds a pseudocount α > 0.'
  },
  {
    question: 'What is the theoretical significance of the Bayes Error Rate?',
    options: [
      'The error rate of a model trained without validation data',
      'The absolute lowest possible classification error achievable by any classifier on a given data distribution',
      'The difference between training error and test error',
      'The number of misclassified samples in the test set'
    ],
    correct: 1,
    explanation: 'The Bayes Error Rate is the irreducible error resulting from inherently overlapping class-conditional probability distributions. No classifier, no matter how complex or deep, can ever achieve a lower error on that distribution.'
  },
  {
    question: 'In Bayesian Linear Regression, what happens to the predictive uncertainty as you move away from the training data (extrapolation)?',
    options: [
      'Uncertainty drops to zero',
      'Uncertainty stays strictly constant',
      'Predictive variance increases dramatically (flares into a trumpet shape)',
      'The model outputs an error'
    ],
    correct: 2,
    explanation: 'Bayesian regression quantifies epistemic uncertainty. Where training data is dense, the posterior variance of the predictions is tight; in unobserved regions, the uncertainty flares outward, warning that the model is extrapolating.'
  },
  {
    question: 'Which Naïve Bayes variant is best suited for continuous real-valued features (e.g., sensor measurements, heights)?',
    options: [
      'Multinomial Naïve Bayes',
      'Bernoulli Naïve Bayes',
      'Gaussian Naïve Bayes',
      'Complement Naïve Bayes'
    ],
    correct: 2,
    explanation: 'Gaussian Naïve Bayes models continuous feature distributions using a 1D normal distribution N(μ_c, σ_c²) for each feature within each class.'
  },
  {
    question: 'What distinguishes a Generative classifier (like Naïve Bayes or GDA) from a Discriminative classifier (like Logistic Regression or SVM)?',
    options: [
      'Generative models only generate synthetic text',
      'Generative models model the joint distribution P(x, y) = P(x|y)P(y), whereas discriminative models directly model the posterior boundary P(y|x)',
      'Discriminative models cannot predict probabilities',
      'Generative models require GPU acceleration'
    ],
    correct: 1,
    explanation: 'Generative models understand how data is generated by modeling P(x|y) and class priors P(y). Discriminative models focus purely on mapping inputs to outputs by estimating the decision boundary P(y|x).'
  },
  {
    question: 'As the amount of training data N approaches infinity, what happens to the MAP estimate?',
    options: [
      'It diverges to infinity',
      'It becomes completely dominated by the prior',
      'It converges exactly to the MLE estimate as the likelihood swamps the prior',
      'It reduces to random guessing'
    ],
    correct: 2,
    explanation: 'The log-posterior is log P(D|θ) + log P(θ). The log-likelihood grows proportionally to N, while the log-prior is fixed. Thus, as N → ∞, the data overwhelms any reasonable prior, and MAP converges to MLE.'
  },
  {
    question: 'What is the primary practical hurdle of full Bayesian learning for complex deep models?',
    options: [
      'Bayesian models cannot be written in Python',
      'Computing the marginal likelihood evidence integral in the denominator ∫ P(D|θ)P(θ)dθ is computationally intractable in high dimensions',
      'Bayesian models always overfit',
      'Bayes rule only applies to binary variables'
    ],
    correct: 1,
    explanation: 'Integrating over millions of parameters in the denominator P(D) = ∫ P(D|θ)P(θ)dθ has no closed-form solution for complex models, requiring approximations like MCMC, Gibbs sampling, or Variational Inference.'
  }
]

// ── Interactive Component 1: MLE vs MAP Coin Toss & Prior Swamping ───────────
function MleMapCanvas({ theme }) {
  const canvasRef = useRef(null)
  const [heads, setHeads] = useState(7)
  const [tails, setTails] = useState(3)
  const [priorType, setPriorType] = useState('fair') // 'fair', 'flat', 'biased'
  const W = 460
  const H = 260

  const { alphaPrior, betaPrior } = useMemo(() => {
    if (priorType === 'fair') return { alphaPrior: 8, betaPrior: 8 }
    if (priorType === 'flat') return { alphaPrior: 1, betaPrior: 1 }
    return { alphaPrior: 14, betaPrior: 3 } // biased toward heads
  }, [priorType])

  // MLE: H / (H + T)
  const total = heads + tails
  const thetaMle = total > 0 ? heads / total : 0.5

  // MAP with Beta(alpha, beta) prior: (H + alpha - 1) / (H + T + alpha + beta - 2)
  const denom = total + alphaPrior + betaPrior - 2
  const thetaMap = denom > 0 ? (heads + alphaPrior - 1) / denom : 0.5

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'

    // Compute Beta PDF values on grid of theta in [0, 1]
    const numPoints = 120
    const thetas = []
    const priorVals = []
    const postVals = []

    // Helper: log-gamma approximation
    const logGamma = (z) => {
      // Lanczos approximation
      const g = 7
      const C = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
      ]
      if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * Math.exp(logGamma(1 - z)))
      z -= 1
      let x = C[0]
      for (let i = 1; i < g + 2; i++) x += C[i] / (z + i)
      const t = z + g + 0.5
      return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x)
    }

    const betaPdf = (th, a, b) => {
      if (th <= 0 || th >= 1) return 0
      const logB = logGamma(a) + logGamma(b) - logGamma(a + b)
      const logP = (a - 1) * Math.log(th) + (b - 1) * Math.log(1 - th) - logB
      return Math.exp(logP)
    }

    const alphaPost = alphaPrior + heads
    const betaPost = betaPrior + tails

    let maxPdf = 0.001
    for (let i = 0; i <= numPoints; i++) {
      const th = Math.max(0.001, Math.min(0.999, i / numPoints))
      thetas.push(th)
      const pr = betaPdf(th, alphaPrior, betaPrior)
      const po = betaPdf(th, alphaPost, betaPost)
      priorVals.push(pr)
      postVals.push(po)
      if (pr > maxPdf) maxPdf = pr
      if (po > maxPdf) maxPdf = po
    }

    const padX = 40
    const padY = 35
    const plotW = W - padX * 2
    const plotH = H - padY * 2

    const toPx = (th) => padX + th * plotW
    const toPy = (val) => H - padY - (val / maxPdf) * plotH

    // Axes
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(padX, H - padY)
    ctx.lineTo(W - padX, H - padY)
    ctx.stroke()

    // Ticks & Labels
    ctx.font = '10px monospace'
    ctx.fillStyle = dark ? '#94A3B8' : '#64748B'
    ctx.textAlign = 'center'
    for (let t = 0; t <= 1; t += 0.25) {
      const px = toPx(t)
      ctx.fillText(t.toFixed(2), px, H - padY + 16)
      ctx.beginPath()
      ctx.moveTo(px, H - padY)
      ctx.lineTo(px, H - padY + 4)
      ctx.stroke()
    }
    ctx.fillText('Coin Bias θ (Probability of Heads)', W / 2, H - 6)

    // Plot Prior Curve (Dashed Gray/Blue)
    ctx.beginPath()
    ctx.strokeStyle = dark ? 'rgba(148, 163, 184, 0.7)' : 'rgba(100, 116, 139, 0.8)'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 4])
    for (let i = 0; i <= numPoints; i++) {
      const x = toPx(thetas[i])
      const y = toPy(priorVals[i])
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()
    ctx.setLineDash([])

    // Plot Posterior Curve (Solid Gradient / Indigo)
    ctx.beginPath()
    ctx.strokeStyle = '#6366F1'
    ctx.lineWidth = 3
    for (let i = 0; i <= numPoints; i++) {
      const x = toPx(thetas[i])
      const y = toPy(postVals[i])
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Shade area under posterior
    ctx.lineTo(toPx(thetas[numPoints]), H - padY)
    ctx.lineTo(toPx(thetas[0]), H - padY)
    ctx.closePath()
    ctx.fillStyle = dark ? 'rgba(99, 102, 241, 0.15)' : 'rgba(99, 102, 241, 0.12)'
    ctx.fill()

    // Vertical indicator for MLE (Emerald)
    const mlePx = toPx(thetaMle)
    ctx.beginPath()
    ctx.moveTo(mlePx, padY)
    ctx.lineTo(mlePx, H - padY)
    ctx.strokeStyle = '#10B981'
    ctx.lineWidth = 2.5
    ctx.stroke()

    // Vertical indicator for MAP (Purple)
    const mapPx = toPx(thetaMap)
    ctx.beginPath()
    ctx.moveTo(mapPx, padY)
    ctx.lineTo(mapPx, H - padY)
    ctx.strokeStyle = '#8B5CF6'
    ctx.lineWidth = 2.5
    ctx.setLineDash([3, 3])
    ctx.stroke()
    ctx.setLineDash([])

    // Legend
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillStyle = '#10B981'
    ctx.fillText(`MLE = ${thetaMle.toFixed(2)} (Data only)`, 16, 20)

    ctx.fillStyle = '#8B5CF6'
    ctx.fillText(`MAP = ${thetaMap.toFixed(2)} (Posterior mode)`, 16, 35)

    ctx.fillStyle = dark ? '#94A3B8' : '#64748B'
    ctx.fillText(`Prior: Beta(${alphaPrior}, ${betaPrior})`, 16, 50)
  }, [heads, tails, alphaPrior, betaPrior, thetaMle, thetaMap, theme, W, H])

  return (
    <div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        <div>
          <label className={`text-xs font-semibold block mb-1 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Prior Belief:
          </label>
          <select
            value={priorType}
            onChange={e => setPriorType(e.target.value)}
            className={`w-full px-2.5 py-1.5 rounded-lg text-xs font-medium border outline-none ${
              theme === 'dark' ? 'bg-slate-800 border-white/10 text-white' : 'bg-white border-gray-200 text-gray-800'
            }`}
          >
            <option value="fair">Strong Fair Coin Beta(8,8)</option>
            <option value="flat">Uniform Prior Beta(1,1)</option>
            <option value="biased">Biased Heads Prior Beta(14,3)</option>
          </select>
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              Heads Observed (H): <span className="text-emerald-400 font-mono font-bold">{heads}</span>
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="30"
            value={heads}
            onChange={e => setHeads(parseInt(e.target.value, 10))}
            className="w-full accent-emerald-500 cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              Tails Observed (T): <span className="text-amber-400 font-mono font-bold">{tails}</span>
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="30"
            value={tails}
            onChange={e => setTails(parseInt(e.target.value, 10))}
            className="w-full accent-amber-500 cursor-pointer"
          />
        </div>
      </div>

      <div className={`rounded-2xl border overflow-hidden relative mb-2 ${
        theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
      }`}>
        <canvas ref={canvasRef} width={W} height={H} className="w-full h-auto block" />
      </div>

      <div className="flex flex-wrap items-center justify-between text-xs gap-2 py-1 px-2">
        <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>
          Sample size: <strong className="font-mono text-indigo-400">{heads + tails} flips</strong>
        </span>
        <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>
          {heads + tails < 6
            ? '🔍 Small sample: Prior exerts heavy gravitational pull on MAP.'
            : heads + tails > 25
            ? '🚀 Large sample: Likelihood dominates! MLE and MAP converge.'
            : '⚖️ Balanced: MAP smoothly blends prior experience with fresh observations.'}
        </span>
      </div>
    </div>
  )
}

// ── Interactive Component 2: Bayesian Linear Regression & Uncertainty Bands ──
function BayesianRegressionCanvas({ theme }) {
  const canvasRef = useRef(null)
  const W = 460
  const H = 280

  // Starting seed points: y ≈ 0.6x - 0.2 + noise
  const initialPoints = useMemo(() => [
    { x: -1.8, y: -1.3 },
    { x: -1.2, y: -0.9 },
    { x: -0.6, y: -0.5 },
    { x: 0.1,  y: -0.1 },
    { x: 0.7,  y: 0.3 },
    { x: 1.2,  y: 0.65 },
  ], [])

  const [points, setPoints] = useState(initialPoints)
  const [alphaPrior, setAlphaPrior] = useState(1.0) // prior precision on weights w ~ N(0, alpha^-1 I)
  const noiseBeta = 10.0 // noise precision 1/sigma^2 = 10 (sigma ~ 0.316)

  // Analytical Bayesian linear regression calculations:
  // Model: y = w0 + w1 * x
  // Basis phi(x) = [1, x]^T
  const model = useMemo(() => {
    const N = points.length
    if (N < 2) return null

    // Design matrix Phi (N x 2)
    // S_N^-1 = alpha * I + beta * Phi^T * Phi
    let sum1 = 0, sumX = 0, sumX2 = 0
    let sumY = 0, sumXY = 0

    points.forEach(p => {
      sum1 += 1
      sumX += p.x
      sumX2 += p.x * p.x
      sumY += p.y
      sumXY += p.x * p.y
    })

    // S_N_inv = [[alpha + beta*N, beta*sumX], [beta*sumX, alpha + beta*sumX2]]
    const a11 = alphaPrior + noiseBeta * sum1
    const a12 = noiseBeta * sumX
    const a21 = noiseBeta * sumX
    const a22 = alphaPrior + noiseBeta * sumX2

    // 2x2 matrix inversion
    const det = a11 * a22 - a12 * a21
    if (Math.abs(det) < 1e-7) return null

    const s11 = a22 / det
    const s12 = -a12 / det
    const s21 = -a21 / det
    const s22 = a11 / det

    // m_N = beta * S_N * Phi^T * y
    // Phi^T * y = [sumY, sumXY]^T
    const v1 = noiseBeta * sumY
    const v2 = noiseBeta * sumXY

    const m0 = s11 * v1 + s12 * v2 // intercept mean
    const m1 = s21 * v1 + s22 * v2 // slope mean

    // Generate 5 deterministic pseudo-sample lines from posterior
    const rand = makePRNG(54321)
    const sampleWeights = []
    for (let k = 0; k < 6; k++) {
      // standard normal variables via Box-Muller
      const u1 = Math.max(1e-6, rand())
      const u2 = rand()
      const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2)

      // Cholesky factor of S_N: L = [[l11, 0], [l21, l22]]
      const l11 = Math.sqrt(Math.max(1e-6, s11))
      const l21 = s12 / l11
      const l22 = Math.sqrt(Math.max(1e-6, s22 - l21 * l21))

      const sampleW0 = m0 + l11 * z0
      const sampleW1 = m1 + (l21 * z0 + l22 * z1)
      sampleWeights.push({ w0: sampleW0, w1: sampleW1 })
    }

    return { m0, m1, s11, s12, s22, sampleWeights }
  }, [points, alphaPrior, noiseBeta])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const dark = theme === 'dark'
    const xMin = -3.2, xMax = 3.2
    const yMin = -2.5, yMax = 2.5

    const toCanvasX = (x) => ((x - xMin) / (xMax - xMin)) * W
    const toCanvasY = (y) => H - ((y - yMin) / (yMax - yMin)) * H

    // Draw grid lines
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
    ctx.lineWidth = 1
    for (let x = -3; x <= 3; x += 1) {
      ctx.beginPath()
      ctx.moveTo(toCanvasX(x), 0)
      ctx.lineTo(toCanvasX(x), H)
      ctx.stroke()
    }
    for (let y = -2; y <= 2; y += 1) {
      ctx.beginPath()
      ctx.moveTo(0, toCanvasY(y))
      ctx.lineTo(W, toCanvasY(y))
      ctx.stroke()
    }

    if (model) {
      const { m0, m1, s11, s12, s22, sampleWeights } = model
      const steps = 80
      const upperBand = []
      const lowerBand = []

      // Calculate mean and 95% predictive uncertainty envelope:
      // sigma_N^2(x) = 1/beta + phi(x)^T S_N phi(x)
      for (let i = 0; i <= steps; i++) {
        const xVal = xMin + (i / steps) * (xMax - xMin)
        const meanY = m0 + m1 * xVal
        // phi^T S_N phi = s11 + 2*s12*x + s22*x^2
        const epistemicVar = s11 + 2 * s12 * xVal + s22 * xVal * xVal
        const predVar = Math.max(0.001, (1 / noiseBeta) + epistemicVar)
        const std = Math.sqrt(predVar)

        upperBand.push({ cx: toCanvasX(xVal), cy: toCanvasY(meanY + 2 * std) })
        lowerBand.push({ cx: toCanvasX(xVal), cy: toCanvasY(meanY - 2 * std) })
      }

      // Draw Shaded 95% Uncertainty Envelope
      ctx.beginPath()
      ctx.moveTo(upperBand[0].cx, upperBand[0].cy)
      upperBand.forEach(pt => ctx.lineTo(pt.cx, pt.cy))
      for (let i = lowerBand.length - 1; i >= 0; i--) {
        ctx.lineTo(lowerBand[i].cx, lowerBand[i].cy)
      }
      ctx.closePath()
      ctx.fillStyle = dark ? 'rgba(99, 102, 241, 0.16)' : 'rgba(99, 102, 241, 0.14)'
      ctx.fill()

      // Envelope border lines
      ctx.strokeStyle = dark ? 'rgba(99, 102, 241, 0.4)' : 'rgba(79, 70, 229, 0.4)'
      ctx.lineWidth = 1
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      upperBand.forEach((pt, i) => (i === 0 ? ctx.moveTo(pt.cx, pt.cy) : ctx.lineTo(pt.cx, pt.cy)))
      ctx.stroke()
      ctx.beginPath()
      lowerBand.forEach((pt, i) => (i === 0 ? ctx.moveTo(pt.cx, pt.cy) : ctx.lineTo(pt.cx, pt.cy)))
      ctx.stroke()
      ctx.setLineDash([])

      // Draw posterior sample candidate lines (thin translucent purple)
      sampleWeights.forEach(s => {
        ctx.beginPath()
        ctx.moveTo(toCanvasX(xMin), toCanvasY(s.w0 + s.w1 * xMin))
        ctx.lineTo(toCanvasX(xMax), toCanvasY(s.w0 + s.w1 * xMax))
        ctx.strokeStyle = dark ? 'rgba(192, 132, 252, 0.3)' : 'rgba(147, 51, 234, 0.25)'
        ctx.lineWidth = 1.2
        ctx.stroke()
      })

      // Draw Posterior Mean Regression Line (Solid Vibrant Indigo)
      ctx.beginPath()
      ctx.moveTo(toCanvasX(xMin), toCanvasY(m0 + m1 * xMin))
      ctx.lineTo(toCanvasX(xMax), toCanvasY(m0 + m1 * xMax))
      ctx.strokeStyle = '#6366F1'
      ctx.lineWidth = 2.5
      ctx.stroke()
    }

    // Draw Data Points
    points.forEach((p, idx) => {
      const cx = toCanvasX(p.x)
      const cy = toCanvasY(p.y)
      ctx.beginPath()
      ctx.arc(cx, cy, 6, 0, Math.PI * 2)
      ctx.fillStyle = '#10B981'
      ctx.fill()
      ctx.strokeStyle = dark ? '#0F172A' : '#FFFFFF'
      ctx.lineWidth = 2
      ctx.stroke()

      // Small index tag
      ctx.font = '9px monospace'
      ctx.fillStyle = dark ? '#94A3B8' : '#64748B'
      ctx.fillText(`x${idx + 1}`, cx + 7, cy - 5)
    })

    // Legend
    ctx.font = '10px monospace'
    ctx.fillStyle = '#6366F1'
    ctx.fillText('— Posterior Mean Line', 12, 20)
    ctx.fillStyle = '#A855F7'
    ctx.fillText('--- Sampled Hypotheses w ~ P(w|D)', 12, 34)
    ctx.fillStyle = dark ? '#818CF8' : '#4F46E5'
    ctx.fillText('Shaded: 95% Epistemic Uncertainty Envelope (±2σ)', 12, 48)
  }, [points, model, theme, W, H])

  const handleCanvasClick = (e) => {
    const rect = canvasRef.current.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    const scaleX = W / rect.width
    const scaleY = H / rect.height
    const cx = px * scaleX
    const cy = py * scaleY

    const xMin = -3.2, xMax = 3.2
    const yMin = -2.5, yMax = 2.5

    const clickX = xMin + (cx / W) * (xMax - xMin)
    const clickY = yMin + ((H - cy) / H) * (yMax - yMin)

    // Add point
    if (points.length < 14) {
      setPoints(pts => [...pts, { x: parseFloat(clickX.toFixed(2)), y: parseFloat(clickY.toFixed(2)) }])
    }
  }

  const handleResetPoints = () => {
    setPoints(initialPoints)
  }

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-3">
          <label className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Prior Weight Precision (α): <span className="font-mono text-purple-400 font-bold">{alphaPrior.toFixed(1)}</span>
          </label>
          <input
            type="range"
            min="0.1"
            max="5"
            step="0.2"
            value={alphaPrior}
            onChange={e => setAlphaPrior(parseFloat(e.target.value))}
            className="w-32 accent-purple-500 cursor-pointer"
          />
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleResetPoints}
            className={`px-3 py-1 rounded-lg text-xs font-medium border transition-colors ${
              theme === 'dark'
                ? 'bg-slate-800 border-white/10 text-gray-300 hover:text-white'
                : 'bg-gray-100 border-gray-200 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Reset Points
          </button>
          <span className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>
            (Click canvas to add data points)
          </span>
        </div>
      </div>

      <div className={`rounded-2xl border overflow-hidden relative cursor-crosshair mb-2 ${
        theme === 'dark' ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-gray-200'
      }`}>
        <canvas
          ref={canvasRef}
          width={W}
          height={H}
          onClick={handleCanvasClick}
          className="w-full h-auto block"
        />
      </div>

      <p className={`text-xs text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        Notice how the uncertainty envelope pinches tight around data points, but balloons outward in regions with no data. Click in an empty corner to watch the uncertainty shrink instantly!
      </p>
    </div>
  )
}

// ── Interactive Component 3: Naïve Bayes Laplace Smoothing Explorer ───────────
function LaplaceSmoothingDemo({ theme }) {
  const [alpha, setAlpha] = useState(1.0)

  // Toy spam vocabulary frequencies: 100 spam emails, 100 ham emails
  const vocab = useMemo(() => [
    { word: 'prize', spamCount: 45, hamCount: 1 },
    { word: 'free', spamCount: 55, hamCount: 8 },
    { word: 'meeting', spamCount: 2, hamCount: 60 },
    { word: 'project', spamCount: 1, hamCount: 48 },
    { word: 'cryptocurrency', spamCount: 12, hamCount: 0 }, // unseen in HAM!
    { word: 'invoice', spamCount: 15, hamCount: 18 }
  ], [])

  const totalSpamWords = 300
  const totalHamWords = 300
  const vocabSize = vocab.length

  // Probability calculation with additive Laplace smoothing:
  // P(word|class) = (count + alpha) / (total_words + alpha * |V|)
  const testDoc = ['free', 'cryptocurrency', 'meeting']

  let logProbSpam = Math.log(0.5) // P(Spam) prior
  let logProbHam = Math.log(0.5)  // P(Ham) prior
  let zeroHamEncountered = false

  testDoc.forEach(w => {
    const item = vocab.find(v => v.word === w)
    const sCount = item ? item.spamCount : 0
    const hCount = item ? item.hamCount : 0

    const pWordSpam = (sCount + alpha) / (totalSpamWords + alpha * vocabSize)
    const pWordHam = (hCount + alpha) / (totalHamWords + alpha * vocabSize)

    if (pWordHam === 0) zeroHamEncountered = true

    logProbSpam += pWordSpam > 0 ? Math.log(pWordSpam) : -999
    logProbHam += pWordHam > 0 ? Math.log(pWordHam) : -999
  })

  // Normalize posterior probabilities via softmax-like log-sum-exp
  const maxLog = Math.max(logProbSpam, logProbHam)
  const expSpam = Math.exp(logProbSpam - maxLog)
  const expHam = Math.exp(logProbHam - maxLog)
  const pSpamFinal = expSpam / (expSpam + expHam)
  const pHamFinal = expHam / (expSpam + expHam)

  return (
    <div>
      <div className="mb-4">
        <div className="flex justify-between items-center mb-1.5">
          <label className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Laplace Smoothing Parameter: <span className="font-mono text-cyan-400 font-bold text-sm">α = {alpha.toFixed(1)}</span>
          </label>
          <span className={`text-xs font-mono ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>
            {alpha === 0 ? '⚠️ Zero smoothing (Zero-frequency disaster risk)' : '✓ Additive smoothing active'}
          </span>
        </div>
        <input
          type="range"
          min="0"
          max="3"
          step="0.2"
          value={alpha}
          onChange={e => setAlpha(parseFloat(e.target.value))}
          className="w-full accent-cyan-500 cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-gray-500 font-mono mt-1">
          <span>α = 0 (Unsmoothed MLE)</span>
          <span>α = 1 (Standard Laplace)</span>
          <span>α = 2</span>
          <span>α = 3 (Heavily smoothed)</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-900/90 border-white/10' : 'bg-white border-gray-200'}`}>
          <h4 className={`text-xs font-bold uppercase tracking-wider mb-2 ${theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}`}>
            Test Email Message
          </h4>
          <p className={`text-sm font-medium mb-3 ${theme === 'dark' ? 'text-gray-200' : 'text-gray-800'}`}>
            &ldquo;Please review this <span className="text-emerald-400 font-semibold">free</span> <span className="text-amber-400 font-semibold underline decoration-wavy">cryptocurrency</span> <span className="text-indigo-400 font-semibold">meeting</span> draft.&rdquo;
          </p>

          <div className="space-y-2 text-xs">
            {testDoc.map(w => {
              const item = vocab.find(v => v.word === w)
              const hCount = item ? item.hamCount : 0
              const sCount = item ? item.spamCount : 0
              const pHam = (hCount + alpha) / (totalHamWords + alpha * vocabSize)
              const pSpam = (sCount + alpha) / (totalSpamWords + alpha * vocabSize)

              return (
                <div key={w} className={`p-2 rounded-lg flex justify-between items-center ${
                  theme === 'dark' ? 'bg-slate-800/60' : 'bg-gray-50'
                }`}>
                  <span className="font-mono font-semibold">{w}</span>
                  <div className="flex gap-3 text-[11px] font-mono">
                    <span className="text-red-400">P(w|Spam): {pSpam.toFixed(3)}</span>
                    <span className={pHam === 0 ? 'text-red-500 font-bold' : 'text-emerald-400'}>
                      P(w|Ham): {pHam.toFixed(3)}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        <div className={`p-4 rounded-xl border flex flex-col justify-between ${
          theme === 'dark' ? 'bg-slate-900/90 border-white/10' : 'bg-white border-gray-200'
        }`}>
          <div>
            <h4 className={`text-xs font-bold uppercase tracking-wider mb-2 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>
              Posterior Classification Outcome
            </h4>

            {alpha === 0 && zeroHamEncountered ? (
              <div className="p-3 rounded-lg bg-red-500/15 border border-red-500/30 text-red-300 text-xs mb-3">
                <strong>Zero-Frequency Disaster:</strong> &ldquo;cryptocurrency&rdquo; had 0 observations in HAM training data. Without smoothing, P(cryptocurrency|HAM) = 0, wiping P(HAM|doc) to strictly 0% regardless of all other words!
              </div>
            ) : null}

            <div className="space-y-3 mt-2">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>P(Spam | Document)</span>
                  <span className="font-mono font-bold text-red-400">{(pSpamFinal * 100).toFixed(1)}%</span>
                </div>
                <div className={`h-2.5 rounded-full overflow-hidden ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-200'}`}>
                  <motion.div
                    className="h-full bg-red-500 rounded-full"
                    animate={{ width: `${pSpamFinal * 100}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>P(Ham | Document)</span>
                  <span className="font-mono font-bold text-emerald-400">{(pHamFinal * 100).toFixed(1)}%</span>
                </div>
                <div className={`h-2.5 rounded-full overflow-hidden ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-200'}`}>
                  <motion.div
                    className="h-full bg-emerald-500 rounded-full"
                    animate={{ width: `${pHamFinal * 100}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className={`p-2.5 rounded-lg text-[11px] font-mono mt-3 ${
            theme === 'dark' ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/20' : 'bg-cyan-50 text-cyan-800 border border-cyan-200'
          }`}>
            P(w|C) = (count + α) / (total + α·|V|)
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main Page Component ───────────────────────────────────────────────────────
export default function BayesianLearning() {
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
          <span className="text-xs text-indigo-400 font-medium">Machine Learning • Probabilistic Modeling &amp; Uncertainty</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Bayesian <span className="gradient-text">Learning</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Moving beyond single point estimates. Bayesian learning views model parameters not as static unknowns, but as <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>probability distributions</strong> that dynamically update as evidence arrives, providing calibrated uncertainty quantification for mission-critical decisions.
        </p>

        <Callout type="analogy" title="Analogy: The Detective's Chalkboard">
          A rookie detective only looks at the clues right in front of them (<strong>Maximum Likelihood</strong>). If they find 3 red fibers in a victim&apos;s car, they might arrest the nearest baker wearing a red scarf. A veteran detective maintains prior background probabilities (<strong>Bayesian Prior</strong>): &ldquo;How likely is a baker to commit a bank robbery compared to known heist syndicates?&rdquo; By combining new evidence with prior belief, they arrive at the <strong>Posterior</strong> probability!
        </Callout>
      </div>

      {/* Core Concepts */}
      <TheoryBlock
        title="Foundational Pillars"
        cards={[
          {
            icon: '🎯',
            title: 'MLE (Data Only)',
            body: 'Finds parameter θ that maximizes the likelihood of the observed data P(D|θ). Purely data-driven, but prone to catastrophic overfitting on small samples.',
            mono: 'θ_MLE = argmax log P(D|θ)'
          },
          {
            icon: '⚖️',
            title: 'MAP (Data + Prior)',
            body: 'Incorporates prior distribution P(θ) to pull the estimate toward reasonable baseline beliefs. The prior acts as formal mathematical regularization (e.g. L2 weight decay).',
            mono: 'θ_MAP = argmax [log P(D|θ) + log P(θ)]'
          },
          {
            icon: '🎲',
            title: 'Bayes Rule',
            body: 'The engine of rational inference: Posterior ∝ Likelihood × Prior. Every new observation updates our state of knowledge in a principled, calibrated manner.',
            mono: 'P(θ|D) = P(D|θ) P(θ) / P(D)'
          },
          {
            icon: '⚡',
            title: 'Naïve Bayes',
            body: 'Assumes features are conditionally independent given the class. Unlocks blazing fast O(n·d) training and high accuracy in text classification and spam detection.',
            mono: 'P(x|y) = ∏ P(xᵢ|y)'
          },
          {
            icon: '🎺',
            title: 'Uncertainty Bands',
            body: 'Bayesian models know what they do NOT know. In regions devoid of training data, the predictive variance blooms wide, alerting users to extrapolations.',
            mono: 'σ_pred²(x) = σ_noise² + φ(x)ᵀ S_N φ(x)'
          },
          {
            icon: '🛡️',
            title: 'Laplace Smoothing',
            body: 'Adds pseudocounts to prevent unseen words or categorical levels from having zero probability and zeroing out the entire classification product.',
            mono: 'P(w|C) = (count + α) / (total + α|V|)'
          }
        ]}
      />

      {/* ========================================================================= */}
      {/* SECTION 1: MLE AND MAP HYPOTHESES */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>
          Part 1 — Parameter Estimation
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          MLE vs MAP Hypotheses
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          The transition from empirical observation to regularized probabilistic priors.
        </p>
      </div>

      <div className={S}>
        <p className={LBL}>1.1 — Maximum Likelihood Estimation (MLE)</p>
        <h2 className={H2}>Maximizing data likelihood under independent trials</h2>
        <p className={`${BODY} mb-4`}>
          Given an observed dataset <code className="font-mono">D = &#123;x₁, x₂, ..., xₙ&#125;</code> generated by independent and identically distributed (i.i.d.) draws from a distribution parameterized by <code className="font-mono">θ</code>, the <strong>likelihood function</strong> is:
        </p>

        <Callout type="formula" mono="L(θ) = P(D|θ) = ∏ᵢ₌₁ⁿ P(xᵢ|θ)">
          Because multiplying dozens of probabilities between 0 and 1 causes severe floating-point underflow, we take the natural logarithm to obtain the <strong>Log-Likelihood</strong>:
        </Callout>

        <div className={`p-4 rounded-xl font-mono text-xs mb-4 overflow-x-auto ${
          theme === 'dark' ? 'bg-slate-950 text-indigo-300 border border-white/10' : 'bg-slate-100 text-indigo-800 border border-slate-300'
        }`}>
          log L(θ) = Σᵢ₌₁ⁿ log P(xᵢ|θ)
        </div>

        <p className={`${BODY} mb-4`}>
          Because the logarithm is a strictly monotonically increasing function, maximizing <code className="font-mono">log L(θ)</code> produces the exact same parameter <code className="font-mono">θ_MLE</code> as maximizing <code className="font-mono">L(θ)</code>, while turning products into convenient sums!
        </p>

        <Callout type="warning" title="The Vulnerability of Pure MLE">
          Imagine flipping a coin 3 times and getting 3 Heads. MLE computes <code className="font-mono">θ_MLE = 3/3 = 1.0</code>! It boldly concludes the coin will NEVER land on Tails. This sample size fragility is why priors are essential.
        </Callout>
      </div>

      <div className={S}>
        <p className={LBL}>1.2 — Maximum A Posteriori (MAP) &amp; Prior as Regularization</p>
        <h2 className={H2}>Infusing prior domain knowledge</h2>
        <p className={`${BODY} mb-4`}>
          By Bayes&apos; theorem, the posterior probability of parameter <code className="font-mono">θ</code> given the data is proportional to the product of the likelihood and the prior:
        </p>

        <Callout type="formula" mono="P(θ|D) ∝ P(D|θ) · P(θ)  ⟹  θ_MAP = argmax_θ [ log P(D|θ) + log P(θ) ]">
          The prior <code className="font-mono">P(θ)</code> expresses what values of <code className="font-mono">θ</code> are plausible before observing any data.
        </Callout>

        <p className={`${BODY} mb-4`}>
          Explore the interactive simulation below. Toggle between different priors and adjust the observed Heads and Tails to watch how MAP balances empirical data with prior expectations:
        </p>

        <MleMapCanvas theme={theme} />

        <DeepDive title="Mathematical Equivalence: Gaussian Prior = L2 Ridge Regularization">
          <p className={`text-sm ${BODY} mb-2`}>
            Consider linear regression with Gaussian noise <code className="font-mono">y ~ N(wᵀx, σ²)</code> and place a zero-mean Gaussian prior on the weights <code className="font-mono">w ~ N(0, τ²I)</code>:
          </p>
          <div className={`p-3 rounded-lg font-mono text-xs mb-2 ${theme === 'dark' ? 'bg-slate-950 text-purple-300' : 'bg-slate-100 text-purple-800'}`}>
            log P(w|D) = -½σ² Σᵢ (yᵢ - wᵀxᵢ)² - ½τ² ||w||² + const
          </div>
          <p className={`text-sm ${BODY}`}>
            Maximizing this posterior is <strong>mathematically identical</strong> to minimizing Mean Squared Error plus an L2 penalty <code className="font-mono">λ ||w||²</code> where <code className="font-mono">λ = σ² / τ²</code>! Regularization in machine learning is literally Bayesian MAP inference in disguise.
          </p>
        </DeepDive>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 2: BAYES OPTIMAL CLASSIFIER & BAYES ERROR */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-cyan-500/5 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}`}>
          Part 2 — Decision Theory
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Optimal Bayes Classifier &amp; Irreducible Error
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          The theoretical limit of predictive accuracy on any dataset.
        </p>
      </div>

      <div className={S}>
        <p className={LBL}>2.1 — The Bayes Optimal Classifier</p>
        <h2 className={H2}>Predicting the argmax posterior class</h2>
        <p className={`${BODY} mb-4`}>
          For any query feature vector <code className="font-mono">x</code>, the optimal decision rule that minimizes the probability of misclassification is the <strong>Bayes Optimal Classifier</strong>:
        </p>

        <Callout type="formula" mono="y*(x) = argmax_c P(Y = c | X = x) = argmax_c [ P(X = x | Y = c) · P(Y = c) ]">
          If the true underlying data distribution <code className="font-mono">P(X, Y)</code> were known perfectly, no classifier on Earth could outperform this rule.
        </Callout>

        <TheoryBlock
          title="Bayes Decision Theory Concepts"
          cards={[
            {
              icon: '👑',
              title: 'Optimal Classifier',
              body: 'Selects the class with highest posterior probability P(c|x). Minimizes expected classification loss under 0-1 error.',
              mono: 'y* = argmax P(c|x)'
            },
            {
              icon: '📉',
              title: 'Bayes Error Rate',
              body: 'The theoretical minimum achievable error rate. Arises from overlapping class distributions where both classes can produce the same feature x.',
              mono: 'E_Bayes = 1 - E[max_c P(c|X)]'
            },
            {
              icon: '🧩',
              title: 'Tractability Challenge',
              body: 'In real applications, the true distributions P(x|c) are unknown. Fitting a full covariance joint distribution requires O(d²) parameters, necessitating simplifying assumptions like Naïve Bayes.',
              mono: 'O(d²) full vs O(d) naïve'
            }
          ]}
        />
      </div>

      {/* ========================================================================= */}
      {/* SECTION 3: NAIVE BAYES & GENERATIVE CLASSIFIERS */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-purple-500/5 border-purple-500/20' : 'bg-purple-50 border-purple-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-purple-400' : 'text-purple-600'}`}>
          Part 3 — Generative Classifiers
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Naïve Bayes &amp; Laplace Smoothing
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          The conditional independence assumption and its practical variants.
        </p>
      </div>

      <div className={S}>
        <p className={LBL}>3.1 — Conditional Independence &amp; Laplace Smoothing</p>
        <h2 className={H2}>Breaking high dimensions into independent 1D factors</h2>
        <p className={`${BODY} mb-4`}>
          Modeling the joint distribution of <code className="font-mono">d</code> features directly requires exponential data. Naïve Bayes makes the daring assumption that <em>given the class label <code className="font-mono">y</code>, every feature <code className="font-mono">xᵢ</code> is conditionally independent of every other feature</em>:
        </p>

        <Callout type="formula" mono="P(x₁, x₂, ..., x_d | y) = ∏ᵢ₌₁^d P(xᵢ | y)">
          This reduces the parameter estimation burden from <code className="font-mono">O(2^d)</code> to just <code className="font-mono">O(d)</code>!
        </Callout>

        <p className={`${BODY} mb-4`}>
          Below, observe how <strong>Laplace smoothing</strong> rescues a spam classifier when it encounters an unseen word in a test email:
        </p>

        <LaplaceSmoothingDemo theme={theme} />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">
          {[
            {
              name: 'Gaussian Naïve Bayes',
              desc: 'For continuous features. Models each feature within class c as a 1D normal curve N(μ_ci, σ_ci²).',
              badge: 'Continuous Data',
              color: 'text-indigo-400'
            },
            {
              name: 'Multinomial Naïve Bayes',
              desc: 'For discrete word counts (TF / Bag-of-Words). Models word frequencies via multinomial draws.',
              badge: 'NLP & Text',
              color: 'text-purple-400'
            },
            {
              name: 'Bernoulli Naïve Bayes',
              desc: 'For binary feature occurrences (word present: 1, absent: 0). Ideal for short texts or flags.',
              badge: 'Binary Vectors',
              color: 'text-cyan-400'
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
      </div>

      {/* ========================================================================= */}
      {/* SECTION 4: BAYESIAN LINEAR REGRESSION */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-emerald-50 border-emerald-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600'}`}>
          Part 4 — Probabilistic Regression
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Bayesian Linear Regression &amp; Uncertainty Quantification
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Predicting both the expected value and the confidence envelope of predictions.
        </p>
      </div>

      <div className={S}>
        <p className={LBL}>4.1 — Posterior over Weights &amp; Epistemic Uncertainty</p>
        <h2 className={H2}>A distribution over possible lines, not just one line</h2>
        <p className={`${BODY} mb-4`}>
          In standard OLS, we find a single line <code className="font-mono">y = wᵀx</code>. In <strong>Bayesian Linear Regression</strong>, the weights themselves are random variables governed by a multivariate Gaussian posterior:
        </p>

        <Callout type="formula" mono="P(w|D) = N(m_N, S_N)   where S_N⁻¹ = αI + β ΦᵀΦ   and   m_N = β S_N Φᵀy">
          For any query point <code className="font-mono">x</code>, the predictive distribution is also Gaussian:
          <br />
          <code className="font-mono">P(y|x, D) = N( m_Nᵀφ(x),  1/β + φ(x)ᵀ S_N φ(x) )</code>
        </Callout>

        <p className={`${BODY} mb-4`}>
          Interactive Canvas: Click on the canvas to add training points. Watch the 95% uncertainty envelope tighten where data exists and flare open into trumpet shapes where data is absent:
        </p>

        <BayesianRegressionCanvas theme={theme} />

        <DeepDive title="Aleatoric vs Epistemic Uncertainty">
          <p className={`text-sm ${BODY} mb-2`}>
            Look at the predictive variance formula: <code className="font-mono">σ²(x) = (1/β) + φ(x)ᵀ S_N φ(x)</code>:
          </p>
          <ul className={`list-disc list-inside space-y-2 text-xs ${BODY}`}>
            <li>
              <strong>Aleatoric Uncertainty (1/β):</strong> Inherent measurement noise in the physical world. Irreducible even if you collect infinite data points.
            </li>
            <li>
              <strong>Epistemic Uncertainty (φ(x)ᵀ S_N φ(x)):</strong> Lack of knowledge about the true parameter values. Reducible by collecting more training data in that region!
            </li>
          </ul>
        </DeepDive>
      </div>

      {/* Python Code Section */}
      <div className={S}>
        <p className={LBL}>Python Scikit-Learn &amp; Scipy Implementations</p>
        <h2 className={H2}>Hands-on code examples</h2>
        <p className={`${BODY} mb-3`}>
          1. <strong>MLE vs MAP Estimation on Coin Toss:</strong>
        </p>
        <CodeBlock code={PYTHON_CODE_MLE_MAP} />

        <p className={`${BODY} mt-6 mb-3`}>
          2. <strong>Multinomial Naïve Bayes with Laplace Smoothing:</strong>
        </p>
        <CodeBlock code={PYTHON_CODE_NAIVE_BAYES} />

        <p className={`${BODY} mt-6 mb-3`}>
          3. <strong>Bayesian Ridge Regression with Uncertainty Bounds:</strong>
        </p>
        <CodeBlock code={PYTHON_CODE_BAYESIAN_REGRESSION} />
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
              10 questions covering MLE, MAP, Bayes Optimal Classifier, Naïve Bayes, and Bayesian Regression • +100 XP
            </p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="bayesian-learning" />
      </div>
    </motion.div>
  )
}
