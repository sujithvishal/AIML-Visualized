import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

const ML_TYPES = [
  {
    type: 'Supervised Learning',
    icon: '🎯',
    color: 'from-indigo-500 to-blue-500',
    border: 'border-indigo-500/30',
    bg: 'bg-indigo-500/10',
    desc: 'Model learns from labeled training data to make predictions.',
    detail: 'You provide pairs of (input, correct output). The algorithm adjusts its internal parameters until its predictions closely match the known outputs. Think of a teacher grading homework — the student learns from the corrections.',
    examples: ['House Price Prediction', 'Spam Detection', 'Image Classification', 'Medical Diagnosis'],
  },
  {
    type: 'Unsupervised Learning',
    icon: '🔍',
    color: 'from-purple-500 to-pink-500',
    border: 'border-purple-500/30',
    bg: 'bg-purple-500/10',
    desc: 'Model finds patterns in unlabeled data without guidance.',
    detail: 'No correct answers are provided. The algorithm explores the data and discovers hidden structure on its own — grouping similar items, reducing dimensions, or detecting anomalies.',
    examples: ['Customer Segmentation', 'Anomaly Detection', 'Topic Modeling', 'Data Compression'],
  },
  {
    type: 'Reinforcement Learning',
    icon: '🎮',
    color: 'from-cyan-500 to-emerald-500',
    border: 'border-cyan-500/30',
    bg: 'bg-cyan-500/10',
    desc: 'Agent learns by interacting with an environment, receiving rewards.',
    detail: 'An agent takes actions, receives reward or penalty signals, and learns a policy that maximises cumulative reward. No dataset needed — the agent generates its own experience by exploring.',
    examples: ['Self-Driving Cars', 'AlphaGo', 'Robot Navigation', 'Game Playing AI'],
  },
]

const FLOW_STEPS = [
  { label: 'Data', icon: '📦', desc: 'Raw information: features & labels', color: '#6366F1' },
  { label: 'Algorithm', icon: '⚙️', desc: 'Finds patterns & optimizes weights', color: '#8B5CF6' },
  { label: 'Model', icon: '🤖', desc: 'Learned mathematical function', color: '#06B6D4' },
  { label: 'Prediction', icon: '💡', desc: 'Output for new unseen inputs', color: '#10B981' },
]

const QUIZ_QUESTIONS = [
  {
    question: 'What is Machine Learning?',
    options: [
      'A type of robot that can walk',
      'A subset of AI where systems learn from data without explicit programming',
      'A programming language for data analysis',
      'A database management system',
    ],
    correct: 1,
    explanation: 'ML is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed for every scenario.',
  },
  {
    question: 'Which type of ML uses labeled training data?',
    options: ['Reinforcement Learning', 'Unsupervised Learning', 'Supervised Learning', 'Transfer Learning'],
    correct: 2,
    explanation: 'Supervised Learning uses labeled data — each training example has both input features and the correct output label.',
  },
  {
    question: 'AlphaGo is an example of which type of ML?',
    options: ['Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning', 'Semi-supervised Learning'],
    correct: 2,
    explanation: 'AlphaGo uses Reinforcement Learning — it learned to play Go by playing millions of games and receiving rewards for winning moves.',
  },
  {
    question: 'Customer segmentation is typically done using:',
    options: ['Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning', 'Deep Learning only'],
    correct: 1,
    explanation: 'Customer segmentation uses Unsupervised Learning (clustering) to group customers by similar behaviour without predefined labels.',
  },
  {
    question: 'In the ML pipeline, what immediately follows data collection?',
    options: ['Deploy the model', 'Apply the learning algorithm', 'Make predictions', 'Delete the data'],
    correct: 1,
    explanation: 'After collecting data, you feed it into a learning algorithm that optimizes model parameters to fit the data.',
  },
]

const PYTHON_CODE = `from sklearn.linear_model import LogisticRegression
import numpy as np

# Training data: [study_hours, sleep_hours]
X = [[1, 4], [2, 5], [3, 6], [5, 7], [6, 8]]
y = [0, 0, 0, 1, 1]  # 0 = fail, 1 = pass

# Step 1 — choose model
model = LogisticRegression()

# Step 2 — train (fit) on labeled data
model.fit(X, y)

# Step 3 — predict on new input
prediction = model.predict([[4, 7]])
probability = model.predict_proba([[4, 7]])[0][1]

print(f"Result : {'Pass' if prediction[0] == 1 else 'Fail'}")
print(f"Confidence: {probability:.1%}")
# Output:
# Result : Pass
# Confidence: 72.4%`

function InteractiveClassifier({ theme }) {
  const canvasRef = useRef(null)
  const [points, setPoints] = useState([
    { x: 80, y: 80, cls: 0 }, { x: 120, y: 100, cls: 0 },
    { x: 100, y: 150, cls: 0 }, { x: 200, y: 200, cls: 1 },
    { x: 250, y: 220, cls: 1 }, { x: 230, y: 170, cls: 1 },
  ])
  const [activeClass, setActiveClass] = useState(0)
  const W = 340, H = 260

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.06)'
    ctx.lineWidth = 1
    for (let i = 0; i < W; i += 30) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, H); ctx.stroke() }
    for (let i = 0; i < H; i += 30) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(W, i); ctx.stroke() }

    const cls0 = points.filter(p => p.cls === 0)
    const cls1 = points.filter(p => p.cls === 1)
    if (cls0.length > 0 && cls1.length > 0) {
      const m0x = cls0.reduce((s, p) => s + p.x, 0) / cls0.length
      const m0y = cls0.reduce((s, p) => s + p.y, 0) / cls0.length
      const m1x = cls1.reduce((s, p) => s + p.x, 0) / cls1.length
      const m1y = cls1.reduce((s, p) => s + p.y, 0) / cls1.length
      const mx = (m0x + m1x) / 2, my = (m0y + m1y) / 2
      const dx = m1x - m0x, dy = m1y - m0y
      const len = Math.sqrt(dx * dx + dy * dy)
      if (len > 0) {
        const nx = -dy / len * 220, ny = dx / len * 220
        ctx.beginPath(); ctx.moveTo(mx - nx, my - ny); ctx.lineTo(mx + nx, my + ny)
        ctx.strokeStyle = '#6366F1'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.stroke(); ctx.setLineDash([])
      }
    }
    points.forEach(p => {
      ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2)
      ctx.fillStyle = p.cls === 0 ? '#6366F1' : '#06B6D4'; ctx.fill()
      ctx.strokeStyle = 'white'; ctx.lineWidth = 1.5; ctx.stroke()
    })
  }, [points, theme])

  useEffect(() => { draw() }, [draw])

  const handleClick = (e) => {
    const rect = canvasRef.current.getBoundingClientRect()
    setPoints(prev => [...prev, {
      x: (e.clientX - rect.left) * (W / rect.width),
      y: (e.clientY - rect.top) * (H / rect.height),
      cls: activeClass,
    }])
  }

  return (
    <div>
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        <span className={`text-sm font-medium ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>Place points:</span>
        {[0, 1].map(c => (
          <button key={c} onClick={() => setActiveClass(c)}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-all ${
              activeClass === c
                ? c === 0 ? 'bg-indigo-500 text-white' : 'bg-cyan-500 text-white'
                : theme === 'dark' ? 'bg-slate-800 text-gray-400' : 'bg-gray-100 text-gray-600'
            }`}>
            ● Class {String.fromCharCode(65 + c)}
          </button>
        ))}
        <button onClick={() => setPoints([])}
          className={`ml-auto px-3 py-1 rounded-lg text-xs ${theme === 'dark' ? 'bg-slate-800 text-gray-400' : 'bg-gray-100 text-gray-500'}`}>
          Clear
        </button>
      </div>
      <canvas ref={canvasRef} width={W} height={H} onClick={handleClick}
        className={`w-full rounded-xl border cursor-crosshair ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
      <p className={`text-xs mt-2 text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        Click to add points • Dashed line = decision boundary learned from data
      </p>
    </div>
  )
}

export default function WhatIsML() {
  const { theme } = useApp()
  const [activeCard, setActiveCard] = useState(null)
  const [flowStep, setFlowStep] = useState(0)

  useEffect(() => {
    const t = setInterval(() => setFlowStep(s => (s + 1) % FLOW_STEPS.length), 1200)
    return () => clearInterval(t)
  }, [])

  const S = `rounded-2xl border p-6 mb-6 ${theme === 'dark' ? 'bg-slate-900/80 border-white/10' : 'bg-white border-gray-200'}`
  const LBL = `text-xs font-semibold uppercase tracking-wider mb-3 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`
  const H2 = `text-xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`
  const BODY = `text-sm leading-relaxed ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="max-w-3xl">

      {/* Hero */}
      <div className="mb-8">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/20 border border-indigo-500/30 mb-4">
          <span className="text-xs text-indigo-400 font-medium">Machine Learning • Beginner</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          What is <span className="gradient-text">Machine Learning?</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          A field of AI that gives computers the ability to <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>learn from data</strong> and improve from experience — without being explicitly programmed for every task.
        </p>
        <Callout type="analogy" title="Real-world analogy">
          Traditional programming: you write rules for every scenario. Machine Learning: you show thousands of examples, and the computer figures out the rules itself — just like how you learned to recognise cats without someone defining every feature of a cat.
        </Callout>
      </div>

      {/* Core Concepts */}
      <TheoryBlock title="Core Concepts" cards={[
        { icon: '📊', title: 'What is a Model?', body: 'A mathematical function mapping inputs → outputs. Training adjusts its internal parameters to approximate the true pattern in data.', mono: 'f(x) = ŷ  (prediction)' },
        { icon: '🏷️', title: 'Features & Labels', body: 'Features are measurable inputs (e.g. house size, age). Labels are the known outputs used in training (e.g. price, spam/not-spam).' },
        { icon: '🔁', title: 'Training vs Inference', body: 'Training: model sees labeled data and updates weights to minimise error. Inference: the trained model predicts on new, unseen inputs.' },
        { icon: '📐', title: 'Overfitting', body: 'When a model memorises training data instead of learning patterns. It scores perfectly on training data but fails on new data.', mono: 'high train acc, low test acc' },
        { icon: '⚖️', title: 'Bias vs Variance', body: 'Bias = error from wrong assumptions (underfitting). Variance = error from sensitivity to training data (overfitting). Good models balance both.' },
        { icon: '🧪', title: 'Train / Test Split', body: 'Always hold out a portion of data for testing. Evaluate the model only on data it has never seen during training to get honest accuracy.' },
      ]} />

      {/* ML Pipeline */}
      <div className={S}>
        <p className={LBL}>The ML Pipeline — How It Works</p>
        <p className={`${BODY} mb-4`}>Every ML project follows the same fundamental cycle, regardless of the algorithm or domain.</p>
        <div className="flex flex-col sm:flex-row items-center justify-between gap-2 mb-4">
          {FLOW_STEPS.map((step, i) => (
            <div key={i} className="flex flex-col sm:flex-row items-center gap-2">
              <motion.div animate={{ scale: flowStep === i ? 1.05 : 1, boxShadow: flowStep === i ? `0 0 20px ${step.color}40` : 'none' }}
                className={`flex flex-col items-center gap-2 px-5 py-4 rounded-xl border transition-all ${flowStep === i ? 'border-indigo-500/50 bg-indigo-500/15' : theme === 'dark' ? 'border-white/10 bg-slate-800/60' : 'border-gray-200 bg-gray-50'}`}>
                <span className="text-2xl">{step.icon}</span>
                <span className={`text-sm font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{step.label}</span>
                <span className={`text-xs text-center max-w-[120px] ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'}`}>{step.desc}</span>
              </motion.div>
              {i < FLOW_STEPS.length - 1 && (
                <motion.div animate={{ opacity: flowStep > i ? 1 : 0.3 }} className="text-2xl text-indigo-500 hidden sm:block">→</motion.div>
              )}
            </div>
          ))}
        </div>
        <Callout type="info" title="Why does this cycle repeat?">
          After evaluating on test data, you often go back and improve: collect more data, try a different algorithm, tune hyperparameters. This iterative loop is how real ML projects work.
        </Callout>
        <DeepDive title="What are hyperparameters?">
          <p className={`text-sm ${BODY} mb-2`}>Parameters are learned <em>from</em> data (weights, biases). Hyperparameters are set <em>before</em> training — they control how learning happens.</p>
          <TheoryBlock title="" cards={[
            { icon: '🔢', title: 'Learning Rate (α)', body: 'How big a step gradient descent takes. Too large → overshoots. Too small → trains very slowly.', mono: 'typical: 0.001 – 0.1' },
            { icon: '🔄', title: 'Epochs', body: 'Number of complete passes through the training dataset. More epochs = more learning, but risks overfitting.', mono: 'typical: 10 – 1000' },
            { icon: '📦', title: 'Batch Size', body: 'How many samples to process before updating weights. Smaller batches = noisier but more frequent updates.', mono: 'typical: 32, 64, 128' },
          ]} />
        </DeepDive>
      </div>

      {/* Types of ML */}
      <div className={S}>
        <p className={LBL}>Types of Machine Learning</p>
        <p className={`${BODY} mb-4`}>The three paradigms differ in <em>how</em> the model receives feedback during training. Click each to expand.</p>
        <div className="grid gap-4">
          {ML_TYPES.map((item, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.08 }}
              onClick={() => setActiveCard(activeCard === i ? null : i)}
              className={`rounded-xl border p-4 cursor-pointer transition-all ${item.border} ${activeCard === i ? item.bg : theme === 'dark' ? 'bg-slate-800/60 hover:bg-slate-800' : 'bg-gray-50 hover:bg-gray-100'}`}>
              <div className="flex items-center gap-3">
                <span className="text-2xl">{item.icon}</span>
                <div className="flex-1">
                  <h3 className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{item.type}</h3>
                  <p className={`text-xs ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{item.desc}</p>
                </div>
                <motion.span animate={{ rotate: activeCard === i ? 90 : 0 }} className="text-gray-500 text-lg">›</motion.span>
              </div>
              <AnimatePresence>
                {activeCard === i && (
                  <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                    <div className="pt-3 mt-3 border-t border-white/10 space-y-3">
                      <p className={`text-sm leading-relaxed ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>{item.detail}</p>
                      <div>
                        <p className={`text-xs font-medium mb-2 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>Real-world examples:</p>
                        <div className="flex flex-wrap gap-2">
                          {item.examples.map(ex => (
                            <span key={ex} className={`px-2.5 py-1 rounded-lg text-xs font-medium bg-gradient-to-r ${item.color} text-white`}>{ex}</span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>
      </div>

      {/* How models learn */}
      <div className={S}>
        <p className={LBL}>How Models Actually Learn</p>
        <h2 className={H2}>The Optimisation Loop</h2>
        <p className={`${BODY} mb-4`}>At its core, every ML training process is a loop that makes the model incrementally less wrong.</p>
        <div className="grid sm:grid-cols-2 gap-3 mb-4">
          {[
            { step: '1', title: 'Forward Pass', body: 'Feed input through the model to get a prediction ŷ.', color: '#6366F1' },
            { step: '2', title: 'Compute Loss', body: 'Measure how wrong ŷ is compared to the true label y.', color: '#8B5CF6' },
            { step: '3', title: 'Backpropagation', body: 'Calculate how each parameter contributed to the error (gradients).', color: '#06B6D4' },
            { step: '4', title: 'Update Weights', body: 'Adjust parameters slightly in the direction that reduces loss.', color: '#10B981' },
          ].map(s => (
            <div key={s.step} className={`p-4 rounded-xl border ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
              <div className="flex items-center gap-2 mb-2">
                <span className="w-6 h-6 rounded-full text-white text-xs font-bold flex items-center justify-center" style={{ background: s.color }}>{s.step}</span>
                <span className={`font-semibold text-sm ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{s.title}</span>
              </div>
              <p className={`text-xs leading-relaxed ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{s.body}</p>
            </div>
          ))}
        </div>
        <Callout type="formula" mono="Loss ↓ → Model improves → Repeat for N epochs">
          This loop runs thousands of times. Each iteration the model gets slightly better at predicting the training data.
        </Callout>
        <DeepDive title="What loss functions are there?">
          <TheoryBlock title="" cards={[
            { icon: '📉', title: 'MSE', body: 'Mean Squared Error — used for regression. Penalises large errors heavily.', mono: '(1/n)Σ(y−ŷ)²' },
            { icon: '🔢', title: 'Cross-Entropy', body: 'Used for classification. Measures the difference between predicted probability distribution and true labels.', mono: '−Σ y·log(ŷ)' },
            { icon: '📏', title: 'MAE', body: 'Mean Absolute Error — more robust to outliers than MSE. Treats all errors equally.', mono: '(1/n)Σ|y−ŷ|' },
          ]} />
        </DeepDive>
      </div>

      {/* Interactive Classifier */}
      <div className={S}>
        <p className={LBL}>Interactive Classifier Playground</p>
        <h2 className={H2}>Build a dataset — watch the model adapt</h2>
        <p className={`${BODY} mb-3`}>Place data points from two classes. The dashed line is the decision boundary — the model's "best guess" of how to separate the two groups given only the points you've added.</p>
        <Callout type="info" title="What you're seeing">
          The boundary shifts each time you add a point because the model recomputes the optimal separation line from scratch using the centroid of each class. In real ML this update happens via gradient descent, not centroid distance.
        </Callout>
        <InteractiveClassifier theme={theme} />
      </div>

      {/* Python Code */}
      <div className={S}>
        <p className={LBL}>Python Example — Supervised Learning</p>
        <h2 className={H2}>Predict pass/fail with scikit-learn</h2>
        <p className={`${BODY} mb-3`}>
          The three-step pattern below — <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>choose → fit → predict</strong> — is universal across every scikit-learn model.
        </p>
        <CodeBlock code={PYTHON_CODE} />
        <DeepDive title="What is scikit-learn?">
          <p className={`text-sm ${BODY} mb-2`}>scikit-learn is Python's most popular ML library. It provides a consistent API for dozens of algorithms: you always call <code className="font-mono text-xs px-1 py-0.5 rounded bg-slate-700">.fit(X, y)</code> to train and <code className="font-mono text-xs px-1 py-0.5 rounded bg-slate-700">.predict(X)</code> to infer. Other popular libraries include TensorFlow, PyTorch, and XGBoost.</p>
        </DeepDive>
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme === 'dark' ? 'bg-indigo-500/10 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>5 questions • Earn 100 XP + ML Beginner badge on completion</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="what-is-ml" />
      </div>
    </motion.div>
  )
}
