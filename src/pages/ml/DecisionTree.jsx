import { useState } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

const PYTHON_CODE_BASIC = `from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entropy-based (ID3/C4.5 style)
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf_entropy.fit(X_train, y_train)
print(f"Entropy tree accuracy : {clf_entropy.score(X_test, y_test):.2%}")

# Gini-based (CART style)
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf_gini.fit(X_train, y_train)
print(f"Gini tree accuracy    : {clf_gini.score(X_test, y_test):.2%}")

# Print the tree rules
print(export_text(clf_gini, feature_names=['sepal_l','sepal_w','petal_l','petal_w']))`

const PYTHON_CODE_PRUNING = `from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Pre-pruning via hyperparameters
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,          # limit tree depth
    min_samples_split=10, # need ≥10 samples to split a node
    min_samples_leaf=5,   # leaf must have ≥5 samples
    max_features='sqrt',  # only consider sqrt(n_features) per split
    random_state=42
)

# Post-pruning via cost-complexity (ccp_alpha)
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas[::5]  # sample every 5th alpha

scores = []
for alpha in alphas:
    t = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores.append(cross_val_score(t, X_train, y_train, cv=5).mean())

best_alpha = alphas[np.argmax(scores)]
print(f"Best ccp_alpha: {best_alpha:.5f}")

final = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
final.fit(X_train, y_train)
print(f"Pruned accuracy: {final.score(X_test, y_test):.2%}")
print(f"Num leaves     : {final.get_n_leaves()}")`

const PYTHON_CODE_CONTINUOUS = `from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Continuous attributes — sklearn handles thresholding automatically
# But here's the manual idea for one feature:
def best_threshold(X_col, y):
    """Find best binary split threshold for a continuous feature."""
    sorted_vals = np.sort(np.unique(X_col))
    thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2  # midpoints
    best_gain, best_t = -1, None
    for t in thresholds:
        left_y  = y[X_col <= t]
        right_y = y[X_col >  t]
        if len(left_y) == 0 or len(right_y) == 0:
            continue
        gain = entropy(y) - (len(left_y)/len(y))*entropy(left_y) \
                          - (len(right_y)/len(y))*entropy(right_y)
        if gain > best_gain:
            best_gain, best_t = gain, t
    return best_t, best_gain

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-9))

# Missing values — sklearn uses surrogate splits internally.
# Simple imputation approach:
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('tree',    DecisionTreeClassifier(max_depth=4, random_state=42))
])
# pipe.fit(X_train, y_train)`

const QUIZ_QUESTIONS = [
  { question: 'Entropy of a pure node (all same class) is:', options: ['1', '0.5', '0', 'undefined'], correct: 2, explanation: 'A pure node has only one class so p=1. Entropy = −1·log₂(1) = 0. No uncertainty.' },
  { question: 'Information Gain is defined as:', options: ['Parent entropy − weighted child entropy', 'Child entropy − parent entropy', 'Gini of parent', 'Sum of leaf purities'], correct: 0, explanation: 'IG = H(parent) − Σ (|child|/|parent|)·H(child). We pick the split that maximises this.' },
  { question: 'Gini impurity of 0 means:', options: ['Maximum impurity', 'Node is pure', 'Equal class distribution', 'Root node'], correct: 1, explanation: 'Gini = 1 − Σpᵢ². When one class has p=1, Gini = 1−1 = 0. Pure node.' },
  { question: 'Gain Ratio adjusts Information Gain by:', options: ['Adding node depth', 'Dividing by Split Information', 'Multiplying by sample size', 'Subtracting Gini'], correct: 1, explanation: 'GainRatio = IG / SplitInfo. This penalises attributes with many values (high SplitInfo) to avoid bias toward them.' },
  { question: 'ID3 uses which criterion?', options: ['Gini impurity', 'Variance reduction', 'Information Gain (entropy)', 'Chi-square'], correct: 2, explanation: 'ID3 uses Information Gain. C4.5 improved it with Gain Ratio. CART uses Gini impurity.' },
  { question: 'CART always produces:', options: ['Multi-way splits', 'Binary splits only', 'Depth-limited trees', 'Balanced trees'], correct: 1, explanation: 'CART (Classification and Regression Trees) always makes binary splits — every node has exactly two children.' },
  { question: 'Pre-pruning stops growth by:', options: ['Removing branches after training', 'Setting limits during training', 'Random branch deletion', 'Increasing tree depth'], correct: 1, explanation: 'Pre-pruning uses constraints like max_depth, min_samples_split to stop growing before the tree overfits.' },
  { question: 'MDL principle says the best model:', options: ['Has the largest tree', 'Minimises description length of model + data given model', 'Has no pruning', 'Uses only one feature'], correct: 1, explanation: 'MDL: prefer the model that gives the shortest combined description of itself and the data it explains. Simpler models generalise better.' },
  { question: 'For continuous attributes, CART:', options: ['Ignores them', 'Converts them to categories', 'Tries all midpoint thresholds as binary splits', 'Uses regression only'], correct: 2, explanation: 'CART sorts the feature, evaluates midpoints between adjacent values as candidate thresholds, and picks the one with the best Gini reduction.' },
  { question: 'A common strategy for missing values in decision trees is:', options: ['Drop the row', 'Use surrogate splits', 'Replace with −1', 'Skip the feature permanently'], correct: 1, explanation: 'Surrogate splits find a backup feature that mimics the primary split. When the primary feature is missing, the surrogate is used instead.' },
]

// ── Entropy / Gini calculator ─────────────────────────────────────────────────
function ImpurityCalculator({ theme }) {
  const [nA, setNA] = useState(6)
  const [nB, setNB] = useState(4)
  const total = nA + nB
  const pA = total === 0 ? 0 : nA / total
  const pB = total === 0 ? 0 : nB / total
  const entropy = total === 0 ? 0 : -(pA > 0 ? pA * Math.log2(pA) : 0) - (pB > 0 ? pB * Math.log2(pB) : 0)
  const gini = 1 - pA ** 2 - pB ** 2

  const bar = (p, color) => (
    <motion.div className="h-full rounded-full" style={{ backgroundColor: color }}
      animate={{ width: `${p * 100}%` }} transition={{ duration: 0.2 }} />
  )

  return (
    <div>
      <div className="grid grid-cols-2 gap-4 mb-5">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            Class A = <span className="text-indigo-400 font-bold">{nA}</span>
          </label>
          <input type="range" min="0" max="20" value={nA} onChange={e => setNA(+e.target.value)} className="w-full accent-indigo-500" />
        </div>
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            Class B = <span className="text-emerald-400 font-bold">{nB}</span>
          </label>
          <input type="range" min="0" max="20" value={nB} onChange={e => setNB(+e.target.value)} className="w-full accent-emerald-500" />
        </div>
      </div>
      <div className="mb-3">
        <div className="flex justify-between text-xs mb-1">
          <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Class distribution</span>
          <span className={`font-mono ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{nA}A + {nB}B = {total}</span>
        </div>
        <div className={`h-5 rounded-full overflow-hidden flex ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-200'}`}>
          {bar(pA, '#6366F1')}{bar(pB, '#10B981')}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 mt-4">
        {[
          { label: 'Entropy H', val: entropy.toFixed(4), unit: 'bits', color: 'text-indigo-400', formula: '−Σ pᵢ log₂(pᵢ)', note: entropy < 0.3 ? 'Low — nearly pure' : entropy > 0.9 ? 'High — very mixed' : 'Moderate impurity' },
          { label: 'Gini', val: gini.toFixed(4), unit: '', color: 'text-emerald-400', formula: '1 − Σ pᵢ²', note: gini < 0.2 ? 'Low — nearly pure' : gini > 0.4 ? 'High — very mixed' : 'Moderate impurity' },
        ].map(m => (
          <div key={m.label} className={`rounded-xl p-3 border ${theme === 'dark' ? 'bg-slate-800 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className={`text-xs font-semibold mb-0.5 ${m.color}`}>{m.label}</p>
            <p className={`font-mono text-xl font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{m.val}</p>
            <p className={`font-mono text-xs mt-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{m.formula}</p>
            <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>{m.note}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Information Gain demo ─────────────────────────────────────────────────────
function InfoGainDemo({ theme }) {
  // Fixed dataset: weather → play tennis (classic)
  // Outlook: Sunny(5), Overcast(4), Rain(5) | Play: Yes(9), No(5)
  const [splitAttr, setSplitAttr] = useState('outlook')
  const H = (pos, neg) => {
    const t = pos + neg; if (t === 0) return 0
    const p = pos / t, q = neg / t
    return -(p > 0 ? p * Math.log2(p) : 0) - (q > 0 ? q * Math.log2(q) : 0)
  }
  const WH = (branches) => branches.reduce((s, [pos, neg]) => s + ((pos + neg) / 14) * H(pos, neg), 0)

  const parentH = H(9, 5)
  const splits = {
    outlook:    { branches: [['Sunny', 2, 3], ['Overcast', 4, 0], ['Rain', 3, 2]], wH: WH([[2,3],[4,0],[3,2]]) },
    humidity:   { branches: [['High', 3, 4],  ['Normal', 6, 1]],                  wH: WH([[3,4],[6,1]]) },
    wind:       { branches: [['Weak', 6, 2],  ['Strong', 3, 3]],                  wH: WH([[6,2],[3,3]]) },
    temperature:{ branches: [['Hot', 2, 2],   ['Mild', 4, 2],  ['Cool', 3, 1]],   wH: WH([[2,2],[4,2],[3,1]]) },
  }
  const cur = splits[splitAttr]
  const ig = (parentH - cur.wH).toFixed(4)
  const splitInfo = -cur.branches.reduce((s, [, pos, neg]) => {
    const n = pos + neg, t = 14; const p = n / t; return s + (p > 0 ? p * Math.log2(p) : 0)
  }, 0).toFixed(4)
  const gainRatio = (parseFloat(ig) / parseFloat(splitInfo)).toFixed(4)

  const attrs = ['outlook', 'humidity', 'wind', 'temperature']

  return (
    <div>
      <div className="flex gap-2 flex-wrap mb-4">
        {attrs.map(a => (
          <button key={a} onClick={() => setSplitAttr(a)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold border capitalize transition-all ${
              splitAttr === a
                ? theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-indigo-100 border-indigo-300 text-indigo-700'
                : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400 hover:text-gray-200' : 'bg-gray-100 border-gray-200 text-gray-500 hover:text-gray-700'
            }`}>{a}</button>
        ))}
      </div>
      <div className="space-y-2 mb-4">
        {cur.branches.map(([name, pos, neg]) => {
          const t = pos + neg
          const h = H(pos, neg)
          return (
            <div key={name} className={`rounded-xl p-3 border ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
              <div className="flex items-center justify-between mb-1.5">
                <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{name}</span>
                <span className={`text-xs font-mono ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>{pos}✅ {neg}❌ ({t} total) H={h.toFixed(3)}</span>
              </div>
              <div className={`h-3 rounded-full overflow-hidden flex ${theme === 'dark' ? 'bg-slate-700' : 'bg-gray-200'}`}>
                <motion.div className="h-full bg-emerald-500 rounded-full" animate={{ width: `${(pos / t) * 100}%` }} transition={{ duration: 0.3 }} />
              </div>
            </div>
          )
        })}
      </div>
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: 'Info Gain', val: ig, color: 'text-indigo-400' },
          { label: 'Split Info', val: splitInfo, color: 'text-amber-400' },
          { label: 'Gain Ratio', val: gainRatio, color: 'text-emerald-400' },
        ].map(m => (
          <div key={m.label} className={`rounded-xl p-3 border text-center ${theme === 'dark' ? 'bg-slate-800 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className={`text-xs mb-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{m.label}</p>
            <p className={`font-mono font-bold text-lg ${m.color}`}>{m.val}</p>
          </div>
        ))}
      </div>
      <p className={`text-xs mt-3 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        Parent entropy: H = {parentH.toFixed(4)} bits &nbsp;•&nbsp; Dataset: 14 samples, 9 Yes / 5 No (Play Tennis)
      </p>
    </div>
  )
}

// ── Tree builder visualiser ───────────────────────────────────────────────────
function TreeVisualiser({ theme }) {
  const [maxDepth, setMaxDepth] = useState(3)
  const [criterion, setCriterion] = useState('gini')
  const [minSamples, setMinSamples] = useState(2)

  // Render a static illustrative tree SVG that responds to depth/criterion
  const nodeColor = theme === 'dark' ? '#1E293B' : '#F8FAFC'
  const nodeBorder = theme === 'dark' ? '#334155' : '#E2E8F0'
  const textColor = theme === 'dark' ? '#CBD5E1' : '#475569'
  const lineColor = theme === 'dark' ? '#334155' : '#CBD5E1'
  const rootColor = criterion === 'gini' ? '#6366F1' : '#8B5CF6'
  const leafColor = '#10B981'

  const nodes = [
    // depth 0
    { id: 'root', x: 190, y: 30, label: 'petal_l ≤ 2.45', impurity: criterion === 'gini' ? 'Gini=0.667' : 'H=1.585', depth: 0, isLeaf: false },
    // depth 1
    { id: 'n1', x: 80, y: 100, label: 'Class: Setosa', impurity: criterion === 'gini' ? 'Gini=0.000' : 'H=0.000', depth: 1, isLeaf: true },
    { id: 'n2', x: 300, y: 100, label: 'petal_w ≤ 1.75', impurity: criterion === 'gini' ? 'Gini=0.499' : 'H=1.000', depth: 1, isLeaf: false },
    // depth 2
    { id: 'n3', x: 210, y: 170, label: 'petal_l ≤ 4.95', impurity: criterion === 'gini' ? 'Gini=0.168' : 'H=0.445', depth: 2, isLeaf: maxDepth < 3 },
    { id: 'n4', x: 370, y: 170, label: 'petal_l ≤ 4.85', impurity: criterion === 'gini' ? 'Gini=0.043' : 'H=0.151', depth: 2, isLeaf: maxDepth < 3 },
    // depth 3
    { id: 'n5', x: 150, y: 240, label: 'Versicolor', impurity: 'Gini=0.000', depth: 3, isLeaf: true },
    { id: 'n6', x: 270, y: 240, label: 'Versicolor', impurity: 'Gini=0.444', depth: 3, isLeaf: true },
    { id: 'n7', x: 330, y: 240, label: 'Virginica', impurity: 'Gini=0.444', depth: 3, isLeaf: true },
    { id: 'n8', x: 410, y: 240, label: 'Virginica', impurity: 'Gini=0.000', depth: 3, isLeaf: true },
  ]
  const edges = [
    { from: 'root', to: 'n1', label: '≤ 2.45' }, { from: 'root', to: 'n2', label: '> 2.45' },
    { from: 'n2', to: 'n3', label: '≤ 1.75' }, { from: 'n2', to: 'n4', label: '> 1.75' },
    ...(maxDepth >= 3 ? [
      { from: 'n3', to: 'n5', label: '≤ 4.95' }, { from: 'n3', to: 'n6', label: '> 4.95' },
      { from: 'n4', to: 'n7', label: '≤ 4.85' }, { from: 'n4', to: 'n8', label: '> 4.85' },
    ] : []),
  ]
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]))
  const visibleNodes = nodes.filter(n => n.depth < maxDepth || n.depth === 0 || n.id === 'n1' || n.id === 'n2' || (maxDepth >= 2 && (n.id === 'n3' || n.id === 'n4')) || (maxDepth >= 3 && n.depth === 3))

  return (
    <div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>Max depth = <span className="text-indigo-400 font-bold">{maxDepth}</span></label>
          <input type="range" min="1" max="3" value={maxDepth} onChange={e => setMaxDepth(+e.target.value)} className="w-full accent-indigo-500" />
        </div>
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>Min samples = <span className="text-purple-400 font-bold">{minSamples}</span></label>
          <input type="range" min="1" max="10" value={minSamples} onChange={e => setMinSamples(+e.target.value)} className="w-full accent-purple-500" />
        </div>
        <div className="flex gap-1 items-end pb-1">
          {['gini', 'entropy'].map(c => (
            <button key={c} onClick={() => setCriterion(c)}
              className={`flex-1 py-1.5 rounded-lg text-xs font-semibold border capitalize transition-all ${
                criterion === c
                  ? theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-indigo-100 border-indigo-300 text-indigo-700'
                  : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'
              }`}>{c}</button>
          ))}
        </div>
      </div>
      <div className={`rounded-xl border overflow-auto ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
        <svg width="480" height={maxDepth >= 3 ? 290 : 220} viewBox={`0 0 480 ${maxDepth >= 3 ? 290 : 220}`} className="w-full">
          {edges.map((e, i) => {
            const f = byId[e.from], t = byId[e.to]
            if (!f || !t) return null
            const mx = (f.x + t.x) / 2, my = (f.y + t.y) / 2
            return (
              <g key={i}>
                <line x1={f.x} y1={f.y + 18} x2={t.x} y2={t.y - 18} stroke={lineColor} strokeWidth="1.5" />
                <text x={mx} y={my} textAnchor="middle" fontSize="9" fill={textColor} fontFamily="monospace">{e.label}</text>
              </g>
            )
          })}
          {visibleNodes.map(n => {
            const isLeaf = n.isLeaf || (n.id === 'n3' && maxDepth < 3) || (n.id === 'n4' && maxDepth < 3)
            const fill = isLeaf ? leafColor : n.depth === 0 ? rootColor : nodeColor
            const textFill = isLeaf || n.depth === 0 ? 'white' : textColor
            return (
              <g key={n.id}>
                <rect x={n.x - 55} y={n.y - 18} width={110} height={36} rx={6} fill={fill} stroke={nodeBorder} strokeWidth="1" />
                <text x={n.x} y={n.y - 4} textAnchor="middle" fontSize="9" fill={textFill} fontWeight="600" fontFamily="system-ui">{n.label}</text>
                <text x={n.x} y={n.y + 10} textAnchor="middle" fontSize="8" fill={isLeaf || n.depth === 0 ? 'rgba(255,255,255,0.7)' : textColor} fontFamily="monospace">{n.impurity}</text>
              </g>
            )
          })}
        </svg>
      </div>
      <p className={`text-xs mt-2 text-center ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>
        Iris dataset tree — {criterion} criterion — depth {maxDepth} — 🟢 = leaf node
      </p>
    </div>
  )
}

// ── Pruning demo ──────────────────────────────────────────────────────────────
function PruningDemo({ theme }) {
  const [mode, setMode] = useState('unpruned')

  const b = `text-xs leading-relaxed mt-1 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`

  const stats = {
    unpruned: { depth: '∞', leaves: 32, trainAcc: '100%', testAcc: '74%', label: 'Overfit', color: 'text-red-400' },
    prepruned: { depth: '4', leaves: 11, trainAcc: '91%', testAcc: '88%', label: 'Balanced', color: 'text-emerald-400' },
    postpruned: { depth: '5', leaves: 8, trainAcc: '89%', testAcc: '90%', label: 'Best generalisation', color: 'text-indigo-400' },
  }
  const s = stats[mode]

  return (
    <div>
      <div className="flex gap-2 mb-4">
        {[['unpruned', '🌿 Unpruned'], ['prepruned', '✂️ Pre-pruned'], ['postpruned', '🔬 Post-pruned']].map(([k, label]) => (
          <button key={k} onClick={() => setMode(k)}
            className={`flex-1 py-2 rounded-lg text-xs font-semibold border transition-all ${
              mode === k
                ? theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-indigo-100 border-indigo-300 text-indigo-700'
                : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'
            }`}>{label}</button>
        ))}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        {[
          { label: 'Max Depth', val: s.depth },
          { label: 'Leaves', val: s.leaves },
          { label: 'Train Acc', val: s.trainAcc },
          { label: 'Test Acc', val: s.testAcc },
        ].map(m => (
          <div key={m.label} className={`rounded-xl p-3 border text-center ${theme === 'dark' ? 'bg-slate-800 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className={`text-xs mb-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{m.label}</p>
            <p className={`font-mono font-bold text-lg ${s.color}`}>{m.val}</p>
          </div>
        ))}
      </div>
      <div className={`rounded-xl p-4 border ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
        {mode === 'unpruned' && <p className={b}>Tree grows until every leaf is pure. Memorises training data — test accuracy suffers significantly. The gap between train (100%) and test (74%) is classic overfitting.</p>}
        {mode === 'prepruned' && <p className={b}>Constraints like <span className="font-mono">max_depth=4, min_samples_leaf=5</span> prevent the tree from growing too deep. Stops early during training. Faster but may underfit if limits are too strict.</p>}
        {mode === 'postpruned' && <p className={b}>Tree grows fully, then branches are removed bottom-up if they don't improve generalisation. Cost-complexity pruning (ccp_alpha) finds the best subtree via cross-validation. Best test accuracy.</p>}
      </div>
    </div>
  )
}

// ── Continuous threshold demo ──────────────────────────────────────────────────
function ThresholdDemo({ theme }) {
  const [threshold, setThreshold] = useState(3.0)
  // Toy dataset: petal_length → class (0=Setosa, 1=Other)
  const data = [
    { v: 1.4, c: 0 }, { v: 1.5, c: 0 }, { v: 1.3, c: 0 }, { v: 1.6, c: 0 }, { v: 1.7, c: 0 },
    { v: 3.9, c: 1 }, { v: 4.2, c: 1 }, { v: 3.5, c: 1 }, { v: 5.1, c: 1 }, { v: 4.7, c: 1 },
    { v: 2.8, c: 0 }, { v: 3.3, c: 1 },
  ]

  const H = (items) => {
    if (items.length === 0) return 0
    const pos = items.filter(d => d.c === 1).length
    const neg = items.length - pos
    const p = pos / items.length, q = neg / items.length
    return -(p > 0 ? p * Math.log2(p) : 0) - (q > 0 ? q * Math.log2(q) : 0)
  }
  const left = data.filter(d => d.v <= threshold)
  const right = data.filter(d => d.v > threshold)
  const parentH = H(data)
  const weightedH = (left.length / data.length) * H(left) + (right.length / data.length) * H(right)
  const ig = (parentH - weightedH).toFixed(4)

  const correct = [...left.filter(d => d.c === 0), ...right.filter(d => d.c === 1)].length
  const acc = Math.round((correct / data.length) * 100)

  return (
    <div>
      <div className="mb-4">
        <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
          Split threshold: petal_length ≤ <span className="text-yellow-400 font-bold">{threshold.toFixed(1)}</span>
        </label>
        <input type="range" min="1.0" max="6.0" step="0.1" value={threshold}
          onChange={e => setThreshold(parseFloat(e.target.value))} className="w-full accent-yellow-500" />
      </div>
      <div className={`rounded-xl border p-3 mb-3 relative overflow-hidden ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`} style={{ height: 80 }}>
        {data.map((d, i) => {
          const x = ((d.v - 1) / 5) * 100
          return (
            <motion.div key={i} animate={{ left: `${x}%` }} transition={{ duration: 0.2 }}
              className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full border-2 border-white"
              style={{ backgroundColor: d.c === 0 ? '#6366F1' : '#10B981' }} />
          )
        })}
        <motion.div animate={{ left: `${((threshold - 1) / 5) * 100}%` }} transition={{ duration: 0.2 }}
          className="absolute top-0 bottom-0 w-0.5 bg-yellow-400" style={{ transform: 'translateX(-50%)' }} />
      </div>
      <div className="flex gap-3 flex-wrap text-xs">
        {[
          { label: 'Left node', val: `${left.length} samples`, sub: `H=${H(left).toFixed(3)}` },
          { label: 'Right node', val: `${right.length} samples`, sub: `H=${H(right).toFixed(3)}` },
          { label: 'Info Gain', val: ig, sub: 'bits', color: 'text-indigo-400' },
          { label: 'Accuracy', val: `${acc}%`, sub: 'at this split', color: acc >= 80 ? 'text-emerald-400' : 'text-amber-400' },
        ].map(m => (
          <div key={m.label} className={`flex-1 px-3 py-2 rounded-lg min-w-[80px] ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-100'}`}>
            <p className={`font-semibold text-xs ${m.color || (theme === 'dark' ? 'text-gray-300' : 'text-gray-700')}`}>{m.val}</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{m.label}</p>
            <p className={`font-mono text-xs ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>{m.sub}</p>
          </div>
        ))}
      </div>
      <p className={`text-xs mt-2 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        🟣 Setosa (class 0) &nbsp; 🟢 Other (class 1) &nbsp; Yellow line = current threshold
      </p>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function DecisionTree() {
  const { theme } = useApp()
  const S = `rounded-2xl border p-6 mb-6 ${theme === 'dark' ? 'bg-slate-900/80 border-white/10' : 'bg-white border-gray-200'}`
  const LBL = `text-xs font-semibold uppercase tracking-wider mb-3 ${theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600'}`
  const H2 = `text-xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`
  const BODY = `text-sm leading-relaxed ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-3xl">
      {/* Hero */}
      <div className="mb-8">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/20 border border-emerald-500/30 mb-4">
          <span className="text-xs text-emerald-400 font-medium">Machine Learning • Supervised Learning</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Decision <span className="gradient-text">Trees</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          A tree-structured classifier that learns a sequence of <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>if-then-else rules</strong> from data — one of the most interpretable ML models.
        </p>
        <Callout type="analogy" title="Analogy: 20 Questions">
          A decision tree plays a game of 20 Questions. Each internal node asks a question about a feature. Each branch is an answer. Each leaf is a final classification. The model learns which questions to ask — and in what order — to most efficiently narrow down the class.
        </Callout>
      </div>

      <TheoryBlock title="Core Concepts" cards={[
        { icon: '🌳', title: 'Tree Structure', body: 'Root node (best split), internal nodes (feature tests), branches (outcomes), and leaf nodes (class labels or values). Predictions = path from root to leaf.', mono: 'depth = longest root-to-leaf path' },
        { icon: '📐', title: 'Splitting Criterion', body: 'At each node, pick the feature/threshold that maximally reduces impurity. Two main criteria: Information Gain (entropy-based) or Gini impurity reduction.', mono: 'best split = argmax IG(S, A)' },
        { icon: '✂️', title: 'Pruning', body: 'Full trees overfit. Pruning removes branches that provide little power. Pre-pruning: stop early. Post-pruning: grow then cut back.', mono: 'ccp_alpha controls cost-complexity' },
        { icon: '🔢', title: 'Continuous Attributes', body: 'For numeric features, try all midpoints between sorted adjacent values as thresholds. The best threshold is chosen by the same impurity criterion.', mono: 'threshold = (vᵢ + vᵢ₊₁) / 2' },
        { icon: '❓', title: 'Missing Values', body: 'Handled via surrogate splits (find a proxy feature that mimics the primary split) or by routing missing-value samples to the most probable child.', mono: 'P(child|missing) = P(child)' },
        { icon: '⚖️', title: 'Bias in Splitting', body: 'Information Gain favours features with many values (e.g. IDs). Gain Ratio and Gini impurity are more balanced and less prone to this bias.', mono: 'GainRatio = IG / SplitInfo' },
      ]} />

      {/* ── Section 6.1 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-emerald-50 border-emerald-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600'}`}>Section 6.1</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Information Theory Foundations</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>The mathematical underpinnings — entropy, information gain, gain ratio, and Gini impurity.</p>
      </div>

      {/* 6.1.1 Entropy */}
      <div className={S}>
        <p className={LBL}>6.1.1 — Entropy: Definition and Properties</p>
        <h2 className={H2}>Measuring uncertainty in a node</h2>
        <p className={`${BODY} mb-4`}>
          Entropy quantifies the impurity of a set. A pure node (all one class) has entropy 0. A maximally mixed node (equal classes) has maximum entropy. Decision trees seek splits that maximally reduce entropy.
        </p>
        <TheoryBlock title="Entropy properties" cards={[
          { icon: '📏', title: 'Definition', body: 'For a set S with K classes, each with probability pᵢ, entropy is the expected number of bits needed to encode a randomly drawn label.', mono: 'H(S) = −Σᵢ pᵢ · log₂(pᵢ)' },
          { icon: '0️⃣', title: 'Pure node (H = 0)', body: 'All samples belong to one class. p = 1 for one class, 0 for others. H = −1·log₂(1) = 0. No uncertainty — no bits needed.', mono: 'H = 0 when all pᵢ ∈ {0,1}' },
          { icon: '📈', title: 'Max entropy', body: 'Entropy is maximised when all K classes are equally likely (pᵢ = 1/K). For binary classification, max H = 1 bit (when p = 0.5).', mono: 'H_max = log₂(K)  bits' },
        ]} />
        <Callout type="formula" mono="H(S) = −p₊ · log₂(p₊) − p₋ · log₂(p₋)">
          For binary classification: if p₊ samples are positive and p₋ are negative, entropy measures how surprised you'd be by a random draw. The −log₂(p) is the surprise of seeing an event with probability p.
        </Callout>
        <p className={`${BODY} mb-4`}>Use the slider below to see how entropy and Gini change as the class balance shifts:</p>
        <ImpurityCalculator theme={theme} />
        <DeepDive title="Entropy vs Gini: which to use?">
          <p className={`text-sm ${BODY} mb-2`}>Both measure impurity and tend to produce similar trees in practice. Key differences:</p>
          <TheoryBlock title="" cards={[
            { icon: '📊', title: 'Entropy', body: 'Logarithmic — slightly more sensitive to class probability differences. Used in ID3 and C4.5. Slightly more expensive to compute (log vs multiply).', mono: '−Σ pᵢ log₂(pᵢ)' },
            { icon: '🎯', title: 'Gini', body: 'Quadratic — measures probability of misclassifying a randomly drawn sample. Used in CART and sklearn default. Faster to compute.', mono: '1 − Σ pᵢ²' },
          ]} />
        </DeepDive>
      </div>

      {/* 6.1.2 Information Gain */}
      <div className={S}>
        <p className={LBL}>6.1.2 — Information Gain and Gain Ratio</p>
        <h2 className={H2}>Which feature to split on?</h2>
        <p className={`${BODY} mb-4`}>
          Information Gain measures the reduction in entropy achieved by splitting on a feature. The ID3 algorithm greedily picks the feature with the highest IG at each node. Gain Ratio corrects for IG's bias toward features with many distinct values.
        </p>
        <TheoryBlock title="Information gain family" cards={[
          { icon: '📉', title: 'Information Gain', body: 'Parent entropy minus the weighted average entropy of child nodes after the split. Higher IG = better split — more uncertainty resolved.', mono: 'IG(S,A) = H(S) − Σᵥ (|Sᵥ|/|S|)·H(Sᵥ)' },
          { icon: '📐', title: 'Split Information', body: 'Measures how broadly the feature spreads samples across branches. A feature with many values has high SplitInfo — IG is artificially inflated for such features.', mono: 'SplitInfo(A) = −Σᵥ (|Sᵥ|/|S|)·log₂(|Sᵥ|/|S|)' },
          { icon: '⚖️', title: 'Gain Ratio (C4.5)', body: 'Divides IG by SplitInfo, penalising features with many values. More robust attribute selection — avoids spurious splits on high-cardinality features like IDs.', mono: 'GainRatio = IG(S,A) / SplitInfo(A)' },
        ]} />
        <Callout type="warning" title="IG bias toward many-valued attributes">
          A feature like "customer ID" (unique per row) would get IG ≈ H(S) — perfect split — but it's completely useless for generalisation. Gain Ratio penalises this by dividing by SplitInfo, which is maximum for such features.
        </Callout>
        <p className={`${BODY} mb-4`}>Select an attribute to see its information gain and gain ratio on the classic Play Tennis dataset:</p>
        <InfoGainDemo theme={theme} />
      </div>

      {/* 6.1.3 Gini impurity */}
      <div className={S}>
        <p className={LBL}>6.1.3 — Gini Impurity</p>
        <h2 className={H2}>The probability of misclassification</h2>
        <p className={`${BODY} mb-4`}>
          Gini impurity measures the probability that a randomly chosen sample from the node would be incorrectly classified if given a random label drawn from the node's class distribution. CART (used by sklearn) minimises weighted Gini impurity across the two children.
        </p>
        <TheoryBlock title="Gini impurity details" cards={[
          { icon: '🎯', title: 'Definition', body: 'Pick a random sample from the node, then a random class label from the same distribution. Gini is the probability they disagree — i.e. the expected misclassification rate.', mono: 'Gini(S) = 1 − Σ pᵢ²' },
          { icon: '✂️', title: 'CART split criterion', body: 'For a binary split (left/right), CART picks the split minimising the weighted sum of child Gini values.', mono: 'ΔGini = Gini(S) − (nₗ/n)Gini(Sₗ) − (nᵣ/n)Gini(Sᵣ)' },
          { icon: '📊', title: 'Range', body: 'Gini ranges from 0 (pure) to 1−1/K (maximally impure, K classes). For binary: max Gini = 0.5. Much like entropy but computed without logarithms.', mono: 'Gini ∈ [0, 1−1/K]' },
        ]} />
        <Callout type="info" title="Gini vs Entropy in practice">
          Both criteria produce nearly identical trees on most datasets. Gini is the sklearn default (it's slightly faster). Entropy can occasionally produce slightly better-balanced trees due to its logarithmic sensitivity. Run cross-validation to choose between them for your dataset.
        </Callout>
      </div>

      {/* ── Section 6.2 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 6.2</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Tree Construction and Overfitting</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>ID3, C4.5, CART — and strategies to prevent overfitting via pruning and MDL.</p>
      </div>

      {/* 6.2.1 ID3 / C4.5 */}
      <div className={S}>
        <p className={LBL}>6.2.1 — Entropy-Based Construction: ID3 and C4.5</p>
        <h2 className={H2}>Building a tree with information gain</h2>
        <p className={`${BODY} mb-4`}>
          ID3 (Quinlan, 1986) was the first practical decision tree algorithm. C4.5 extended it with continuous attributes, missing values, and Gain Ratio. Both follow the same greedy recursive strategy.
        </p>
        <TheoryBlock title="ID3 / C4.5 algorithm" cards={[
          { icon: '1️⃣', title: 'Base cases', body: 'If all samples have the same class → return a leaf with that class. If no features remain → return a leaf with the majority class.', mono: 'if |classes|=1 → leaf' },
          { icon: '2️⃣', title: 'Pick best attribute', body: 'Compute IG (ID3) or Gain Ratio (C4.5) for every remaining feature. Select the feature A* that maximises the criterion.', mono: 'A* = argmax_A IG(S, A)' },
          { icon: '3️⃣', title: 'Recurse', body: 'Create a child node for each value of A*. Recurse on the subset of samples with each value. C4.5 uses thresholding for continuous features.', mono: 'for v in values(A*): recurse(Sᵥ)' },
        ]} />
        <Callout type="info" title="C4.5 improvements over ID3">
          C4.5 adds: (1) Gain Ratio to reduce bias, (2) continuous attribute thresholding, (3) missing value handling via fractional counts, and (4) post-pruning based on pessimistic error estimation. C5.0 is the commercial successor.
        </Callout>
        <TreeVisualiser theme={theme} />
      </div>

      {/* 6.2.2 CART */}
      <div className={S}>
        <p className={LBL}>6.2.2 — CART Algorithm: Binary Splits and Gini</p>
        <h2 className={H2}>The algorithm behind sklearn's DecisionTreeClassifier</h2>
        <p className={`${BODY} mb-4`}>
          CART (Classification and Regression Trees, Breiman 1984) differs from ID3/C4.5 in one key way: it always makes <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>binary splits</strong>. Every internal node has exactly two children, regardless of how many values the feature has.
        </p>
        <TheoryBlock title="CART vs ID3/C4.5" cards={[
          { icon: '✌️', title: 'Binary splits only', body: 'For categorical features, CART searches all possible subsets for a binary partition. For continuous features, it tries all midpoint thresholds. Always exactly 2 children.', mono: 'left: X[j] ≤ t  |  right: X[j] > t' },
          { icon: '🎯', title: 'Gini criterion', body: 'CART minimises the weighted Gini impurity of the split (for classification) or variance reduction (for regression). No separate Gain Ratio needed since binary splits naturally limit spread.', mono: 'ΔGini = Gini(S) − nₗ/n·Gini(Sₗ) − nᵣ/n·Gini(Sᵣ)' },
          { icon: '🔢', title: 'Regression variant', body: 'For regression targets, replace Gini with variance. Pick the split that maximally reduces the weighted variance of the two child nodes. Leaf value = mean(y) in leaf.', mono: 'loss = Σ(yᵢ − ȳ_leaf)²' },
        ]} />
        <Callout type="formula" mono="best split = argmin_{j,t} [ nₗ·Gini(Sₗ) + nᵣ·Gini(Sᵣ) ]">
          For every feature j and every candidate threshold t, compute the weighted Gini of the two resulting subsets. The (j, t) pair that minimises this quantity is the split chosen at this node.
        </Callout>
        <DeepDive title="CART for regression">
          <p className={`text-sm ${BODY} mb-2`}>The same binary split structure applies to regression. Instead of minimising Gini, minimise the weighted variance of the two children. The prediction at a leaf is the mean of all training samples that reach it.</p>
          <Callout type="formula" mono="split loss = Σᵢ∈left (yᵢ−ȳₗ)² + Σᵢ∈right (yᵢ−ȳᵣ)²">
            This is equivalent to minimising the MSE of a step-function approximation. CART regression trees are the building blocks of Gradient Boosting and Random Forests.
          </Callout>
        </DeepDive>
      </div>

      {/* 6.2.3 Pruning */}
      <div className={S}>
        <p className={LBL}>6.2.3 — Avoiding Overfitting: Pre-Pruning and Post-Pruning</p>
        <h2 className={H2}>Why full trees overfit and how to fix it</h2>
        <p className={`${BODY} mb-4`}>
          A fully grown tree perfectly fits training data but memorises noise. The training accuracy is 100% but test accuracy degrades. Pruning finds the right complexity trade-off.
        </p>
        <Callout type="warning" title="Bias–variance trade-off">
          Deep trees: low bias (fit training data well), high variance (sensitive to noise). Shallow/pruned trees: higher bias (may miss patterns), lower variance (generalise better). Pruning moves right along this curve.
        </Callout>
        <PruningDemo theme={theme} />
        <TheoryBlock title="Pruning strategies" cards={[
          { icon: '✋', title: 'Pre-pruning (early stopping)', body: 'Set constraints before training: max_depth, min_samples_split, min_samples_leaf, max_features. Fast but requires good hyperparameter tuning.', mono: 'max_depth, min_samples_leaf' },
          { icon: '✂️', title: 'Post-pruning (cost-complexity)', body: 'Grow full tree, then prune bottom-up. For each subtree, compute: does removing it improve generalisation? ccp_alpha controls the regularisation strength.', mono: 'ccp_alpha: larger → more pruning' },
          { icon: '📊', title: 'Reduced Error Pruning', body: 'Hold out a validation set. Remove each subtree; if accuracy doesn\'t drop, keep it pruned. Simple and effective. Used in C4.5.', mono: 'prune if val_acc(pruned) ≥ val_acc(full)' },
        ]} />
      </div>

      {/* 6.2.4 MDL */}
      <div className={S}>
        <p className={LBL}>6.2.4 — Minimum Description Length (MDL) Principle</p>
        <h2 className={H2}>Occam's Razor as an information-theoretic principle</h2>
        <p className={`${BODY} mb-4`}>
          MDL formalises the intuition that simpler models generalise better. It says: choose the model H that minimises the total description length of both the model itself and the data encoded using that model.
        </p>
        <TheoryBlock title="MDL for decision trees" cards={[
          { icon: '📦', title: 'Two-part MDL', body: 'Total cost = bits to describe the tree structure + bits to encode training errors given the tree. A bigger tree costs more to describe; a smaller tree makes more errors.', mono: 'L(H) + L(D|H) → minimise' },
          { icon: '⚖️', title: 'Trade-off', body: 'Adding a split costs bits (model gets more complex). It saves bits only if the impurity reduction is worth the complexity cost. MDL prunes when the saving is insufficient.', mono: 'add split only if ΔL(D|H) > L(split)' },
          { icon: '🔗', title: 'Connection to regularisation', body: 'MDL is a principled Bayesian justification for regularisation. Minimising L(H)+L(D|H) is equivalent to maximising the posterior P(H|D) ∝ P(D|H)·P(H).', mono: 'MDL ≡ MAP estimation' },
        ]} />
        <Callout type="analogy" title="Compression analogy">
          Imagine you need to transmit the dataset to a friend. You could send the raw data (long), or send a model + corrections. A good model compresses the data well — the corrections are small. MDL says: pick the model that gives the shortest total transmission.
        </Callout>
        <DeepDive title="MDL in practice">
          <p className={`text-sm ${BODY} mb-2`}>Pure MDL-based tree induction is complex to implement. In practice, cost-complexity pruning (ccp_alpha in sklearn) approximates the MDL objective. The pruning parameter α corresponds directly to the cost of adding one node to the tree description.</p>
          <Callout type="info">Cross-validation is the standard way to set ccp_alpha. Generate a sequence of pruned trees (via <span className="font-mono">cost_complexity_pruning_path</span>), then pick the alpha that maximises validation accuracy.</Callout>
        </DeepDive>
      </div>

      {/* ── Section 6.3 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-amber-500/5 border-amber-500/20' : 'bg-amber-50 border-amber-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-amber-400' : 'text-amber-600'}`}>Section 6.3</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Handling Special Attributes</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Real-world data is messy — continuous values, missing entries, and high-cardinality features all need special handling.</p>
      </div>

      {/* 6.3.1 Continuous attributes */}
      <div className={S}>
        <p className={LBL}>6.3.1 — Handling Continuous-Valued Attributes</p>
        <h2 className={H2}>Thresholding: turning continuous into binary splits</h2>
        <p className={`${BODY} mb-4`}>
          A continuous feature has infinitely many possible split points. The standard approach sorts the values and evaluates midpoints between adjacent pairs as candidate thresholds. Only n−1 midpoints need evaluation for n distinct values.
        </p>
        <TheoryBlock title="Threshold selection procedure" cards={[
          { icon: '🔢', title: 'Sort unique values', body: 'Sort all distinct values of the feature in ascending order. Adjacent pairs with different class labels are the only candidate threshold locations.', mono: 'v₁ < v₂ < ... < vₙ' },
          { icon: '✂️', title: 'Evaluate midpoints', body: 'For each adjacent pair (vᵢ, vᵢ₊₁), try threshold t = (vᵢ + vᵢ₊₁)/2. Compute IG (or ΔGini) for the resulting binary split.', mono: 't* = argmax_{tᵢ} IG(S, X ≤ tᵢ)' },
          { icon: '♻️', title: 'Reuse across nodes', body: 'A continuous feature can be used again in deeper nodes with a different threshold. Unlike categorical features in ID3, continuous features are never "used up".', mono: 'same feature, different threshold per node' },
        ]} />
        <Callout type="info" title="Complexity note">
          Evaluating all thresholds for one continuous feature takes O(n log n) time (sorting). With m features and n samples, total split-search cost per node is O(m·n·log n). sklearn uses optimised routines but this is why trees on high-n datasets can be slow.
        </Callout>
        <p className={`${BODY} mb-4`}>Drag the threshold below to see how information gain changes across split points:</p>
        <ThresholdDemo theme={theme} />
      </div>

      {/* 6.3.2 Missing values */}
      <div className={S}>
        <p className={LBL}>6.3.2 — Handling Missing Attribute Values</p>
        <h2 className={H2}>Three strategies when data is incomplete</h2>
        <p className={`${BODY} mb-4`}>
          Real datasets often have missing values. Decision trees can handle them gracefully using several strategies, without requiring imputation before training.
        </p>
        <TheoryBlock title="Missing value strategies" cards={[
          { icon: '🔄', title: 'Surrogate splits', body: 'Find a secondary feature that best mimics the primary split. When the primary feature is missing, use the surrogate. CART uses this by default.', mono: 'backup: best correlated feature' },
          { icon: '📊', title: 'Fractional routing (C4.5)', body: 'During training, route a sample with a missing value to ALL children, weighted by the proportion of non-missing samples that took each branch.', mono: 'wᵢ = |Sᵥ| / |S_non-missing|' },
          { icon: '⬇️', title: 'Majority routing', body: 'Send missing-value samples to the child that has the most training samples. Simple and fast — often good enough for small amounts of missingness.', mono: 'child = argmax |Sᵥ|' },
        ]} />
        <Callout type="warning" title="Imputation vs native handling">
          SimpleImputer (mean/mode fill) before training is common but introduces bias — especially if missingness is not random. Native tree handling (surrogate splits or fractional routing) is more statistically correct and can even use the missingness itself as a signal.
        </Callout>
        <DeepDive title="Using missingness as a feature">
          <p className={`text-sm ${BODY} mb-2`}>Sometimes, <em>whether</em> a value is missing is itself informative (e.g. income is missing more often for high earners who don't disclose). You can add a binary indicator feature "X_j is missing" alongside X_j. The tree can then split on this indicator directly.</p>
          <Callout type="info">This is the basis for how LightGBM and XGBoost handle missing values — they learn the optimal direction to send missing-value samples at each split.</Callout>
        </DeepDive>
      </div>

      {/* 6.3.3 Multi-valued attributes */}
      <div className={S}>
        <p className={LBL}>6.3.3 — Multi-Valued Attributes and Attribute Selection</p>
        <h2 className={H2}>High-cardinality features and how to handle them</h2>
        <p className={`${BODY} mb-4`}>
          Categorical features with many unique values (e.g. city, user_id) cause two problems: ID3's IG metric artificially favours them, and creating one branch per value leads to data fragmentation — tiny, unreliable subsets.
        </p>
        <TheoryBlock title="Multi-valued attribute strategies" cards={[
          { icon: '⚖️', title: 'Gain Ratio (C4.5)', body: 'Divides IG by SplitInfo. Features that split samples into many branches have high SplitInfo, penalising them. Naturally discourages splitting on high-cardinality features.', mono: 'GainRatio = IG / SplitInfo' },
          { icon: '✌️', title: 'Binary grouping (CART)', body: 'For a categorical feature with K values, CART searches all 2^(K−1) − 1 binary partitions of the values and picks the best one. Avoids fragmentation entirely.', mono: 'search all subsets → best binary split' },
          { icon: '🔢', title: 'Ordinal encoding + threshold', body: 'Encode categories as integers (e.g. by target mean — target encoding) then treat as a continuous feature. Reduces K values to a single threshold search.', mono: 'target_encode → sort → threshold' },
        ]} />
        <Callout type="warning" title="Data fragmentation problem">
          If a feature has 100 unique values and you create 100 branches, each branch has very few samples. The impurity estimates become unreliable — the tree appears to learn a lot but is actually fitting noise. Pre-pruning (min_samples_split) helps, as does preferring binary splits.
        </Callout>
        <Callout type="analogy" title="Why not just use customer ID?">
          A "customer_id" feature has one unique value per sample. It would give IG = H(S) — perfect split. But each leaf has one sample with a memorised label. It's 100% training accuracy and 0% generalisation. Gain Ratio fixes this: SplitInfo = log₂(n), making GainRatio ≈ 0.
        </Callout>
      </div>

      {/* Code blocks */}
      <div className={S}>
        <p className={LBL}>Python Example — Tree Construction</p>
        <h2 className={H2}>Entropy vs Gini on the Iris dataset</h2>
        <p className={`${BODY} mb-3`}>Train and compare entropy-based and Gini-based decision trees, then print the learned rules.</p>
        <CodeBlock code={PYTHON_CODE_BASIC} />
      </div>

      <div className={S}>
        <p className={LBL}>Python Example — Pruning</p>
        <h2 className={H2}>Pre-pruning and cost-complexity post-pruning</h2>
        <p className={`${BODY} mb-3`}>Find the optimal ccp_alpha via cross-validation, then compare pruned vs unpruned tree accuracy.</p>
        <CodeBlock code={PYTHON_CODE_PRUNING} />
      </div>

      <div className={S}>
        <p className={LBL}>Python Example — Special Attributes</p>
        <h2 className={H2}>Continuous thresholding and missing value handling</h2>
        <p className={`${BODY} mb-3`}>Manual threshold search for a continuous feature, and a pipeline with SimpleImputer for missing data.</p>
        <CodeBlock code={PYTHON_CODE_CONTINUOUS} />
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme === 'dark' ? 'bg-emerald-500/10 border-emerald-500/20' : 'bg-emerald-50 border-emerald-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>10 questions covering all sections • +100 XP on completion</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="decision-tree" />
      </div>
    </motion.div>
  )
}
