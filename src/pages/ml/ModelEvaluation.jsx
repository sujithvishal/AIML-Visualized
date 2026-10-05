import { useState, useMemo } from 'react'
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
const PYTHON_CODE_METRICS_CV = `from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate imbalanced classification dataset (90% negative, 10% positive)
X, y = make_classification(
    n_samples=1000, n_features=12, weights=[0.9, 0.1], random_state=42
)

# 1. Stratified K-Fold: preserves identical class proportions across every fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(clf, X, y, cv=skf, scoring=scoring)

print("--- 5-Fold Stratified Cross-Validation Results ---")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.upper():12s}: Mean = {scores.mean():.4f} (±{scores.std():.4f})")

# 2. Confusion Matrix & Report
clf.fit(X, y)
y_pred = clf.predict(X)
cm = confusion_matrix(y, y_pred)
print(f"\\nConfusion Matrix (Threshold 0.5):\\nTN={cm[0,0]}  FP={cm[0,1]}\\nFN={cm[1,0]}  TP={cm[1,1]}")`

const PYTHON_CODE_SHAP_FAIRNESS = `import numpy as np
# 1. SHAP (SHapley Additive exPlanations) for Model Interpretability
# Computes game-theoretic Shapley values satisfying efficiency and additivity
try:
    import shap
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    model = GradientBoostingClassifier(random_state=42).fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X.iloc[:10])

    print("--- SHAP Feature Attribution for Sample #0 ---")
    print(f"Base Value (E[f(x)]): {shap_values.base_values[0]:.4f}")
    # Top 3 feature contributions
    top_indices = np.argsort(np.abs(shap_values.values[0]))[::-1][:3]
    for idx in top_indices:
        feat = X.columns[idx]
        val = shap_values.values[0][idx]
        print(f"  {feat:24s}: {val:+.4f} ({'Supports Positive' if val > 0 else 'Opposes Positive'})")
except ImportError:
    print("SHAP package optional in environment.")

# 2. Fairness Metric: Demographic Parity Difference
# Evaluates P(Y_hat = 1 | Group A) - P(Y_hat = 1 | Group B)
def demographic_parity_diff(y_pred, sensitive_attr):
    rate_a = np.mean(y_pred[sensitive_attr == 0])
    rate_b = np.mean(y_pred[sensitive_attr == 1])
    return abs(rate_a - rate_b)

print("\\nDemographic Parity metric ready for production auditing.")`

const QUIZ_QUESTIONS = [
  {
    question: 'Why is Stratified K-Fold cross-validation strongly preferred over standard K-Fold for classification problems?',
    options: [
      'It runs twice as fast by skipping gradient checks',
      'It preserves the exact class percentage ratio in every single fold, preventing rare positive classes from being underrepresented or missing in training/validation splits',
      'It requires zero memory allocation',
      'It works exclusively on regression targets'
    ],
    correct: 1,
    explanation: 'Standard k-fold splits data purely at random. If a dataset has 2% fraud, random folds might contain 0% fraud, rendering validation metrics meaningless. Stratified k-fold ensures every fold mirrors the global class ratio.'
  },
  {
    question: 'In medical diagnosis for a rare, deadly disease, which type of error is typically far more dangerous to commit?',
    options: [
      'False Positive (Type I error): diagnosing a healthy patient as sick',
      'False Negative (Type II error): failing to detect a patient who actually has the disease, missing life-saving treatment',
      'Both errors always carry identical real-world cost',
      'Neither error matters if overall accuracy is 99%'
    ],
    correct: 1,
    explanation: 'A False Negative means a sick patient is sent home untreated with fatal consequences. In high-stakes medicine, models are calibrated to maximize Recall (minimizing False Negatives), accepting higher False Positives.'
  },
  {
    question: 'Why can the Area Under the ROC Curve (ROC-AUC) provide a deceptively optimistic score on severely imbalanced datasets?',
    options: [
      'ROC curves cannot be plotted for binary classification',
      'The False Positive Rate (FPR = FP / (FP + TN)) has True Negatives (TN) in the denominator. When the negative class is massive, even a huge number of false alarms yields a tiny FPR, inflating the ROC curve',
      'ROC-AUC scores are strictly capped at 0.5',
      'ROC curves ignore True Positives entirely'
    ],
    correct: 1,
    explanation: 'Because TN is massive in imbalanced datasets (e.g. 99.9% negatives), FPR stays tiny even with hundreds of false alarms. The Precision-Recall (PR) curve does not involve TN and exposes the drop in precision.'
  },
  {
    question: 'What is the theoretical foundation of SHAP (SHapley Additive exPlanations) values in machine learning explainability?',
    options: [
      'Principal Component Analysis',
      'Cooperative game theory (Lloyd Shapley), where features act as players in a coalition and are credited with their fair marginal contribution to the prediction',
      'Bayesian hyperparameter optimization',
      'Gradient clipping'
    ],
    correct: 1,
    explanation: 'SHAP computes Shapley values from game theory. It evaluates the marginal contribution of a feature across all possible subsets of features, uniquely satisfying mathematical axioms like Efficiency and Symmetry.'
  },
  {
    question: 'What is the fundamental difference between LIME and SHAP for local model explainability?',
    options: [
      'LIME works only on neural networks, while SHAP works only on decision trees',
      'LIME fits an interpretable surrogate model (e.g. sparse linear regression) locally around a single perturbed prediction, while SHAP calculates exact axiomatic game-theoretic feature attributions',
      'SHAP is an unregularized method that ignores feature correlation',
      'LIME requires retraining the global model from scratch'
    ],
    correct: 1,
    explanation: 'LIME perturbs the input around a query point and fits a local surrogate linear model. SHAP evaluates game-theoretic marginal contributions across coalitions, providing consistent additive attributions.'
  },
  {
    question: 'What does the fairness criterion &ldquo;Equalized Odds&rdquo; require across sensitive groups (e.g., race, gender)?',
    options: [
      'Every demographic group must receive the exact same percentage of positive outcomes, regardless of qualified rates',
      'The model must have equal True Positive Rates (TPR) AND equal False Positive Rates (FPR) across all demographic groups',
      'The training data must have sensitive attributes permanently deleted',
      'All features must have zero correlation with the target'
    ],
    correct: 1,
    explanation: 'Equalized Odds dictates that the predictor Y_hat and sensitive attribute A are conditionally independent given the true outcome Y: P(Y_hat=1 | Y=y, A=0) = P(Y_hat=1 | Y=y, A=1) for both y=0 and y=1.'
  },
  {
    question: 'What is the &ldquo;Impossibility Theorem of Fairness&rdquo; in algorithmic justice?',
    options: [
      'Machine learning algorithms can never achieve more than 50% accuracy on demographic data',
      'Except in trivial cases (e.g. 100% accurate models or equal base rates), Demographic Parity, Equalized Odds, and Predictive Parity are mathematically incompatible and cannot all be satisfied simultaneously',
      'Fairness can only be achieved using quantum computing',
      'Deep learning models cannot be audited'
    ],
    correct: 1,
    explanation: 'Chouldechova (2017) and Kleinberg et al. mathematically proved that when base rates of true outcomes differ between demographic groups, you cannot simultaneously achieve equal calibration and equal false positive/negative rates.'
  },
  {
    question: 'What is an Adversarial Perturbation (such as created by FGSM: Fast Gradient Sign Method)?',
    options: [
      'A massive corruption that turns 50% of image pixels completely black',
      'An imperceptibly tiny, mathematically crafted perturbation added to the input along the direction of the loss gradient that completely fools the model into high-confidence errors',
      'A software bug in the model checkpoint loader',
      'An intentional deletion of training labels'
    ],
    correct: 1,
    explanation: 'FGSM adds x_adv = x + ε · sign(∇_x L(θ, x, y)). Even when the human eye cannot detect the microscopic change ε, the high-dimensional linear accumulation flips the classifier\'s prediction with high confidence.'
  },
  {
    question: 'When comparing two classification models across K folds, why is a Paired t-test on standard cross-validation folds technically invalid without corrections?',
    options: [
      'The folds do not contain continuous numbers',
      'The test errors across folds are not independent because training sets overlap substantially across cross-validation folds, violating the independence assumption of the t-test',
      'The t-test requires GPU acceleration',
      'Classification errors follow a uniform distribution'
    ],
    correct: 1,
    explanation: 'In 5-fold CV, any two training sets share 75% of their training samples! This high correlation artificially deflates the variance estimate and produces inflated Type I false positive significance (use 5x2cv paired t-test instead).'
  },
  {
    question: 'What is a &ldquo;Model Card&rdquo; in Responsible AI governance?',
    options: [
      'A physical plastic ID badge worn by data scientists',
      'A standardized, structured document detailing a model\'s architecture, intended use cases, performance across demographic subgroups, training data limitations, and ethical considerations',
      'A proprietary benchmark score used exclusively by cloud providers',
      'The compiled binary file of a neural network'
    ],
    correct: 1,
    explanation: 'Pioneered by Margaret Mitchell et al. (Google), Model Cards are short, structured documents that disclose model purpose, performance across sensitive groups, out-of-scope usages, and data biases to foster accountability.'
  }
]

// ── Interactive Component 1: Threshold Tuner & Dynamic Confusion Matrix ───────
function ConfusionMatrixThresholdDemo({ theme }) {
  const [threshold, setThreshold] = useState(0.5)

  // 100 synthetic test samples with known probabilities:
  // 30 Positives (y=1) centered around prob ~ 0.72
  // 70 Negatives (y=0) centered around prob ~ 0.28
  const samples = useMemo(() => {
    const rand = makePRNG(7771)
    const list = []
    // 30 true positives
    for (let i = 0; i < 30; i++) {
      const p = Math.min(0.99, Math.max(0.05, 0.72 + (rand() - 0.5) * 0.42))
      list.push({ y: 1, prob: p })
    }
    // 70 true negatives
    for (let i = 0; i < 70; i++) {
      const p = Math.min(0.95, Math.max(0.01, 0.28 + (rand() - 0.5) * 0.45))
      list.push({ y: 0, prob: p })
    }
    return list
  }, [])

  // Calculate confusion matrix counts based on current threshold
  const { tp, fp, tn, fn } = useMemo(() => {
    let tpCount = 0, fpCount = 0, tnCount = 0, fnCount = 0
    samples.forEach(s => {
      const pred = s.prob >= threshold ? 1 : 0
      if (s.y === 1 && pred === 1) tpCount++
      else if (s.y === 0 && pred === 1) fpCount++
      else if (s.y === 0 && pred === 0) tnCount++
      else fnCount++
    })
    return { tp: tpCount, fp: fpCount, tn: tnCount, fn: fnCount }
  }, [samples, threshold])

  const total = tp + fp + tn + fn
  const accuracy = (tp + tn) / total
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0
  const specificity = tn + fp > 0 ? tn / (tn + fp) : 0
  const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0

  return (
    <div>
      <div className="mb-4">
        <div className="flex justify-between items-center mb-1.5">
          <label className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
            Classification Decision Threshold (τ): <span className="font-mono text-cyan-400 font-bold text-sm">{threshold.toFixed(2)}</span>
          </label>
          <span className={`text-xs font-mono ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>
            {threshold < 0.35 ? 'Aggressive (High Recall, Lower Precision)' : threshold > 0.65 ? 'Conservative (High Precision, Misses Positives)' : 'Balanced Operating Point'}
          </span>
        </div>
        <input
          type="range"
          min="0.1"
          max="0.9"
          step="0.02"
          value={threshold}
          onChange={e => setThreshold(parseFloat(e.target.value))}
          className="w-full accent-cyan-500 cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-gray-500 font-mono mt-1">
          <span>τ = 0.1 (Flag everything)</span>
          <span>τ = 0.5 (Default)</span>
          <span>τ = 0.9 (Flag only absolute certainties)</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {/* 2x2 Confusion Matrix Grid */}
        <div className={`p-4 rounded-2xl border ${theme === 'dark' ? 'bg-slate-900/90 border-white/10' : 'bg-white border-gray-200'}`}>
          <h4 className={`text-xs font-bold uppercase tracking-wider mb-3 ${theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}`}>
            Confusion Matrix (N = {total})
          </h4>

          <div className="grid grid-cols-2 gap-2 text-center text-xs">
            {/* TP */}
            <div className={`p-3 rounded-xl border ${theme === 'dark' ? 'bg-emerald-500/15 border-emerald-500/30' : 'bg-emerald-50 border-emerald-200'}`}>
              <span className="text-[10px] text-emerald-400 font-bold block uppercase">True Positive (TP)</span>
              <span className="text-xl font-mono font-bold text-emerald-400">{tp}</span>
              <p className="text-[10px] text-gray-400 mt-0.5">Sick correctly detected</p>
            </div>

            {/* FP */}
            <div className={`p-3 rounded-xl border ${theme === 'dark' ? 'bg-amber-500/15 border-amber-500/30' : 'bg-amber-50 border-amber-200'}`}>
              <span className="text-[10px] text-amber-400 font-bold block uppercase">False Positive (FP)</span>
              <span className="text-xl font-mono font-bold text-amber-400">{fp}</span>
              <p className="text-[10px] text-gray-400 mt-0.5">Type I: False alarm</p>
            </div>

            {/* FN */}
            <div className={`p-3 rounded-xl border ${theme === 'dark' ? 'bg-red-500/15 border-red-500/30' : 'bg-red-50 border-red-200'}`}>
              <span className="text-[10px] text-red-400 font-bold block uppercase">False Negative (FN)</span>
              <span className="text-xl font-mono font-bold text-red-400">{fn}</span>
              <p className="text-[10px] text-gray-400 mt-0.5">Type II: Critical miss!</p>
            </div>

            {/* TN */}
            <div className={`p-3 rounded-xl border ${theme === 'dark' ? 'bg-indigo-500/15 border-indigo-500/30' : 'bg-indigo-50 border-indigo-200'}`}>
              <span className="text-[10px] text-indigo-400 font-bold block uppercase">True Negative (TN)</span>
              <span className="text-xl font-mono font-bold text-indigo-400">{tn}</span>
              <p className="text-[10px] text-gray-400 mt-0.5">Healthy correctly cleared</p>
            </div>
          </div>
        </div>

        {/* Live Metrics Dashboard */}
        <div className={`p-4 rounded-2xl border flex flex-col justify-between ${
          theme === 'dark' ? 'bg-slate-900/90 border-white/10' : 'bg-white border-gray-200'
        }`}>
          <div>
            <h4 className={`text-xs font-bold uppercase tracking-wider mb-2 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>
              Derived Performance Metrics
            </h4>

            <div className="space-y-2.5 mt-2">
              {[
                { name: 'Recall (Sensitivity)', val: recall, desc: 'TP / (TP + FN)', color: 'text-emerald-400' },
                { name: 'Precision', val: precision, desc: 'TP / (TP + FP)', color: 'text-cyan-400' },
                { name: 'F1-Score', val: f1, desc: 'Harmonic mean of P & R', color: 'text-indigo-400' },
                { name: 'Specificity', val: specificity, desc: 'TN / (TN + FP)', color: 'text-purple-400' },
                { name: 'Accuracy', val: accuracy, desc: '(TP + TN) / Total', color: 'text-gray-300' },
              ].map(m => (
                <div key={m.name} className="flex justify-between items-center text-xs">
                  <div>
                    <span className={`font-semibold ${theme === 'dark' ? 'text-gray-200' : 'text-gray-800'}`}>{m.name}</span>
                    <span className="text-[10px] text-gray-500 font-mono ml-2">({m.desc})</span>
                  </div>
                  <span className={`font-mono font-bold ${m.color}`}>{(m.val * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          <div className={`p-2.5 rounded-lg text-[11px] font-mono mt-3 ${
            theme === 'dark' ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/20' : 'bg-cyan-50 text-cyan-800 border border-cyan-200'
          }`}>
            Slide threshold τ to see the Precision-Recall seesaw in real time!
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Interactive Component 2: SHAP Feature Attribution Waterfall ───────────────
function ShapWaterfallDemo({ theme }) {
  const [profile, setProfile] = useState('approved') // 'approved' vs 'rejected'

  const baseValue = 0.42 // E[f(x)]: global baseline approval probability

  const features = useMemo(() => {
    if (profile === 'approved') {
      return [
        { name: 'Annual Income ($120k)', value: +0.22, positive: true },
        { name: 'Credit Score (780)', value: +0.18, positive: true },
        { name: 'Debt-to-Income (18%)', value: +0.08, positive: true },
        { name: 'Missed Payments (0)', value: +0.05, positive: true },
        { name: 'Recent Inquiries (3)', value: -0.06, positive: false },
        { name: 'Employment Length (1 yr)', value: -0.04, positive: false }
      ]
    } else {
      return [
        { name: 'Recent Delinquencies (2)', value: -0.28, positive: false },
        { name: 'Debt-to-Income (54%)', value: -0.16, positive: false },
        { name: 'Credit Score (590)', value: -0.12, positive: false },
        { name: 'Annual Income ($45k)', value: -0.05, positive: false },
        { name: 'Stable Employment (8 yrs)', value: +0.10, positive: true },
        { name: 'Savings Balance ($15k)', value: +0.06, positive: true }
      ]
    }
  }, [profile])

  const totalShap = features.reduce((acc, f) => acc + f.value, 0)
  const finalPrediction = Math.min(0.99, Math.max(0.01, baseValue + totalShap))

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <label className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
          Select Loan Applicant Profile:
        </label>
        <div className="flex gap-2">
          <button
            onClick={() => setProfile('approved')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              profile === 'approved'
                ? 'bg-emerald-500 text-white shadow-sm'
                : theme === 'dark' ? 'bg-slate-800 text-gray-400' : 'bg-gray-100 text-gray-700'
            }`}
          >
            Applicant A (Approved: 85%)
          </button>
          <button
            onClick={() => setProfile('rejected')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              profile === 'rejected'
                ? 'bg-red-500 text-white shadow-sm'
                : theme === 'dark' ? 'bg-slate-800 text-gray-400' : 'bg-gray-100 text-gray-700'
            }`}
          >
            Applicant B (Rejected: 15%)
          </button>
        </div>
      </div>

      <div className={`p-5 rounded-2xl border mb-3 ${
        theme === 'dark' ? 'bg-slate-900/90 border-white/10' : 'bg-white border-gray-200'
      }`}>
        <div className="flex justify-between items-center text-xs pb-3 border-b border-inherit mb-3">
          <div>
            <span className="text-gray-400">Baseline Expected Probability E[f(x)]:</span>
            <strong className="font-mono text-indigo-400 ml-1.5">{(baseValue * 100).toFixed(0)}%</strong>
          </div>
          <div>
            <span className="text-gray-400">Final Model Output f(x):</span>
            <strong className={`font-mono ml-1.5 text-sm ${finalPrediction >= 0.5 ? 'text-emerald-400' : 'text-red-400'}`}>
              {(finalPrediction * 100).toFixed(1)}% ({finalPrediction >= 0.5 ? 'APPROVED' : 'REJECTED'})
            </strong>
          </div>
        </div>

        <p className={`text-xs font-semibold mb-3 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
          Feature Attributions (Game-Theoretic Shapley Values):
        </p>

        <div className="space-y-2.5">
          {features.map((f) => {
            const isPos = f.value >= 0
            const pct = Math.abs(f.value) * 180 // visual width scale
            return (
              <div key={f.name} className="text-xs">
                <div className="flex justify-between text-[11px] mb-1">
                  <span className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>{f.name}</span>
                  <span className={`font-mono font-bold ${isPos ? 'text-emerald-400' : 'text-red-400'}`}>
                    {isPos ? '+' : ''}{(f.value * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center h-3 w-full bg-slate-800/40 rounded-full overflow-hidden">
                  <div className="w-1/2 flex justify-end">
                    {!isPos && (
                      <motion.div
                        className="h-full bg-red-500 rounded-l-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    )}
                  </div>
                  <div className="w-0.5 h-full bg-white/40 shrink-0" />
                  <div className="w-1/2 flex justify-start">
                    {isPos && (
                      <motion.div
                        className="h-full bg-emerald-500 rounded-r-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    )}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <Callout type="formula" mono="f(x) = E[f(x)] + Σ_{j=1}^d φ_j(x)">
        SHAP satisfies the <strong>Efficiency Axiom</strong>: the sum of all individual feature attributions φ_j(x) exactly equals the difference between the model&apos;s output f(x) and the dataset baseline E[f(x)]. No attribution is lost or double-counted!
      </Callout>
    </div>
  )
}

// ── Interactive Component 3: Fairness & Demographic Parity Simulator ──────────
function FairnessSimulatorDemo({ theme }) {
  const [thresholdGroupA, setThresholdGroupA] = useState(0.5)
  const [thresholdGroupB, setThresholdGroupB] = useState(0.5)

  // Subgroup acceptance rate calculations
  // Group A acceptance rate: 60% at tau=0.5
  // Group B acceptance rate: 40% at tau=0.5 (historical bias disparity)
  const rateA = Math.max(0.05, Math.min(0.95, 0.60 - (thresholdGroupA - 0.5) * 0.8))
  const rateB = Math.max(0.05, Math.min(0.95, 0.40 - (thresholdGroupB - 0.5) * 0.8))

  const demographicParityGap = Math.abs(rateA - rateB) * 100
  const isFair = demographicParityGap < 5

  return (
    <div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              Group A Threshold: <span className="font-mono text-indigo-400 font-bold">{thresholdGroupA.toFixed(2)}</span>
            </span>
            <span className="font-mono text-xs text-emerald-400 font-bold">{(rateA * 100).toFixed(0)}% Selected</span>
          </div>
          <input
            type="range"
            min="0.2"
            max="0.8"
            step="0.05"
            value={thresholdGroupA}
            onChange={e => setThresholdGroupA(parseFloat(e.target.value))}
            className="w-full accent-indigo-500 cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
              Group B Threshold: <span className="font-mono text-purple-400 font-bold">{thresholdGroupB.toFixed(2)}</span>
            </span>
            <span className="font-mono text-xs text-purple-400 font-bold">{(rateB * 100).toFixed(0)}% Selected</span>
          </div>
          <input
            type="range"
            min="0.2"
            max="0.8"
            step="0.05"
            value={thresholdGroupB}
            onChange={e => setThresholdGroupB(parseFloat(e.target.value))}
            className="w-full accent-purple-500 cursor-pointer"
          />
        </div>
      </div>

      <div className={`p-4 rounded-xl border flex flex-wrap items-center justify-between text-xs mb-3 ${
        theme === 'dark' ? 'bg-slate-900/90 border-white/10' : 'bg-white border-gray-200'
      }`}>
        <div>
          <span className="text-gray-400">Demographic Parity Disparity:</span>
          <strong className={`font-mono text-sm ml-2 ${isFair ? 'text-emerald-400' : 'text-amber-400'}`}>
            {demographicParityGap.toFixed(1)}%
          </strong>
        </div>

        <div className={`px-3 py-1 rounded-full font-semibold text-[11px] ${
          isFair
            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
            : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
        }`}>
          {isFair ? '✓ Demographic Parity Satisfied (<5% gap)' : '⚠️ Disparate Impact Detected (>5% gap)'}
        </div>
      </div>

      <p className={`text-xs text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        Notice that achieving Demographic Parity often requires post-processing with separate group thresholds, highlighting the tension between statistical parity and individual treatment!
      </p>
    </div>
  )
}

// ── Main Page Component ───────────────────────────────────────────────────────
export default function ModelEvaluation() {
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
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-500/20 border border-cyan-500/30 mb-4"
        >
          <span className="text-xs text-cyan-400 font-medium">Machine Learning • Rigorous Evaluation &amp; Responsible AI</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Model Evaluation &amp; <span className="gradient-text">Emerging Requirements</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Deploying machine learning models in production requires far more than measuring training loss. It demands <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>rigorous cross-validation, asymmetric cost analysis, game-theoretic interpretability, fairness auditing, and adversarial defense</strong>.
        </p>

        <Callout type="analogy" title="Analogy: The Flight Certification Test">
          You would never board an airplane that was only flight-tested on calm, sunny mornings (training data). Aviation engineers test planes in Category 5 turbulence, lightning storms, and simulated dual-engine failures. <strong>Model Evaluation and Responsible AI</strong> are the flight certifications of machine learning.
        </Callout>
      </div>

      {/* Core Concepts */}
      <TheoryBlock
        title="Evaluation &amp; Governance Pillars"
        cards={[
          {
            icon: '🔀',
            title: 'Stratified K-Fold',
            body: 'Splits data into K folds while strictly preserving class ratios. Protects against catastrophic variance on rare positive classes.',
            mono: 'StratifiedKFold(n_splits=5)'
          },
          {
            icon: '🎯',
            title: 'Precision vs Recall',
            body: 'Precision = TP / (TP + FP) minimizes false alarms. Recall = TP / (TP + FN) catches every true case. Balanced by F1 harmonic mean.',
            mono: 'F1 = 2·P·R / (P + R)'
          },
          {
            icon: '🔍',
            title: 'SHAP & LIME',
            body: 'Game-theoretic Shapley values allocate credit to individual features. LIME fits local surrogate models to explain black-box predictions.',
            mono: 'f(x) = E[f(x)] + Σ φ_i'
          },
          {
            icon: '⚖️',
            title: 'Algorithmic Fairness',
            body: 'Auditing demographic parity and equalized odds. Mitigating historical, representation, and measurement biases across sensitive cohorts.',
            mono: 'P(Ŷ=1|A=0) ≈ P(Ŷ=1|A=1)'
          },
          {
            icon: '🛡️',
            title: 'Adversarial Robustness',
            body: 'Defending against imperceptible gradient perturbations (e.g. FGSM) that fool deep models into confident catastrophic blunders.',
            mono: 'x_adv = x + ε·sign(∇_x L)'
          },
          {
            icon: '📋',
            title: 'Model Cards',
            body: 'Standardized documentation detailing architecture, intended use, subgroup benchmark metrics, and out-of-scope risks.',
            mono: 'Responsible AI Standard'
          }
        ]}
      />

      {/* ========================================================================= */}
      {/* SECTION 1: COMPARING MACHINE LEARNING MODELS */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>
          Part 1 — Rigorous Evaluation
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Cross-Validation &amp; Confusion Matrix Analysis
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Asymmetric error costs, decision thresholds, and statistical significance.
        </p>
      </div>

      <div className={S}>
        <p className={LBL}>1.1 — The Decision Threshold &amp; Error Costs</p>
        <h2 className={H2}>Calibrating for real-world business objectives</h2>
        <p className={`${BODY} mb-4`}>
          A classifier outputs continuous probability scores <code className="font-mono">P(y = 1 | x)</code>. Converting these probabilities into discrete decisions requires selecting a threshold <code className="font-mono">τ</code>. Shifting <code className="font-mono">τ</code> directly rebalances the ratio between <strong>False Positives (Type I)</strong> and <strong>False Negatives (Type II)</strong>:
        </p>

        <ConfusionMatrixThresholdDemo theme={theme} />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <h3 className="font-semibold text-xs text-amber-400 mb-1">Type I Error: False Positive</h3>
            <p className={`text-xs ${BODY}`}>
              Predicting positive when true label is negative. Example: Spam filter flags an important job offer email as spam.
            </p>
          </div>

          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <h3 className="font-semibold text-xs text-red-400 mb-1">Type II Error: False Negative</h3>
            <p className={`text-xs ${BODY}`}>
              Predicting negative when true label is positive. Example: Failing to detect fraudulent bank transaction or tumor.
            </p>
          </div>
        </div>
      </div>

      <div className={S}>
        <p className={LBL}>1.2 — Statistical Significance in Model Comparison</p>
        <h2 className={H2}>Are performance differences real or random noise?</h2>
        <p className={`${BODY} mb-4`}>
          When Model B achieves an accuracy of <code className="font-mono">84.2%</code> versus Model A&apos;s <code className="font-mono">82.8%</code>, is the difference statistically significant?
        </p>

        <TheoryBlock
          title="Significance Testing Methods"
          cards={[
            {
              icon: '⚠️',
              title: 'Standard Paired t-Test (Flawed)',
              body: 'Running standard paired t-tests on 5-fold CV violates independence because training folds overlap by 75%, deflating variance and causing false positives.',
              mono: 'Overlap violates i.i.d.'
            },
            {
              icon: '✓',
              title: '5x2cv Paired t-Test',
              body: 'Dietterich (1998): 5 replications of 2-fold cross-validation. Uses non-overlapping 50-50 splits, producing robust, calibrated Type I error rates.',
              mono: 'Recommended by Dietterich'
            },
            {
              icon: '⚖️',
              title: 'McNemar\'s Test',
              body: 'Contingency test based on instances where Model A and Model B disagree: (b - c)² / (b + c) ~ χ²(1). Fast, non-parametric, requires no re-training.',
              mono: 'Contingency Disagreement'
            }
          ]}
        />
      </div>

      {/* ========================================================================= */}
      {/* SECTION 2: EMERGING REQUIREMENTS & RESPONSIBLE AI */}
      {/* ========================================================================= */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-purple-500/5 border-purple-500/20' : 'bg-purple-50 border-purple-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-purple-400' : 'text-purple-600'}`}>
          Part 2 — Trustworthy Machine Learning
        </p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Interpretability, Fairness &amp; Adversarial Robustness
        </h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Game-theoretic SHAP explainability, demographic equity, and adversarial defense.
        </p>
      </div>

      <div className={S}>
        <p className={LBL}>2.1 — Model Explainability: SHAP &amp; LIME</p>
        <h2 className={H2}>Opening the algorithmic black box</h2>
        <p className={`${BODY} mb-4`}>
          In regulated domains (finance, healthcare, hiring), models must explain <em>why</em> an individual decision was made. <strong>SHAP (SHapley Additive exPlanations)</strong> calculates the exact marginal contribution of each feature across all possible feature coalitions:
        </p>

        <ShapWaterfallDemo theme={theme} />
      </div>

      <div className={S}>
        <p className={LBL}>2.2 — Algorithmic Bias, Fairness &amp; Adversarial Robustness</p>
        <h2 className={H2}>Demographic parity and adversarial defenses</h2>
        <p className={`${BODY} mb-4`}>
          Biases creep into machine learning models through historical training data, unrepresentative sampling, and measurement bias. Adjusting group thresholds can mitigate demographic disparities:
        </p>

        <FairnessSimulatorDemo theme={theme} />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <h3 className="font-semibold text-xs text-amber-400 mb-1">Adversarial Attacks (FGSM)</h3>
            <p className={`text-xs ${BODY} mb-2`}>
              Adding an imperceptible gradient step <code className="font-mono">ε · sign(∇_x L)</code> fools state-of-the-art vision models into misclassifying a panda as a gibbon with 99.3% confidence!
            </p>
            <span className="text-[11px] text-gray-400 font-mono">Defense: Adversarial Training &amp; Gradient Masking</span>
          </div>

          <div className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/40 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <h3 className="font-semibold text-xs text-cyan-400 mb-1">Model Cards for Transparency</h3>
            <p className={`text-xs ${BODY} mb-2`}>
              Standardized metadata schema documenting intended usage, out-of-scope applications, demographic slicing benchmarks, and ethical risks.
            </p>
            <span className="text-[11px] text-gray-400 font-mono">Standard: Mitchell et al. (FAT* 2019)</span>
          </div>
        </div>

        <DeepDive title="The Impossibility Theorem of Fairness">
          <p className={`text-sm ${BODY} mb-2`}>
            Can an algorithm satisfy all fairness definitions simultaneously?
          </p>
          <p className={`text-sm ${BODY}`}>
            <strong>The Mathematical Reality:</strong> Alexandra Chouldechova (2017) and Jon Kleinberg proved that when base rates of the true condition differ between two groups, it is <em>mathematically impossible</em> to satisfy <strong>Demographic Parity</strong>, <strong>Equalized Odds</strong>, and <strong>Predictive Parity</strong> at the same time! System designers must make explicit, transparent ethical choices about which trade-off to prioritize.
          </p>
        </DeepDive>
      </div>

      {/* Python Code Section */}
      <div className={S}>
        <p className={LBL}>Python Scikit-Learn &amp; SHAP Implementations</p>
        <h2 className={H2}>Production evaluation workflows</h2>
        <p className={`${BODY} mb-3`}>
          1. <strong>Stratified 5-Fold Cross-Validation with Multi-Metric Scoring:</strong>
        </p>
        <CodeBlock code={PYTHON_CODE_METRICS_CV} />

        <p className={`${BODY} mt-6 mb-3`}>
          2. <strong>TreeExplainer SHAP Feature Attributions &amp; Fairness Audit:</strong>
        </p>
        <CodeBlock code={PYTHON_CODE_SHAP_FAIRNESS} />
      </div>

      {/* Quiz Section */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${
          theme === 'dark' ? 'bg-cyan-500/10 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'
        }`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
              10 comprehensive questions covering cross-validation, confusion matrices, SHAP, fairness, and adversarial robustness • +100 XP
            </p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="model-evaluation" />
      </div>
    </motion.div>
  )
}
