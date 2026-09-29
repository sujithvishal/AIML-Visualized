import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

const PYTHON_CODE = `from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Study hours → pass/fail
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]]).reshape(-1,1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# Probability predictions
probs = model.predict_proba(X)[:, 1]  # P(pass)
preds = model.predict(X)

print(f"Accuracy     : {accuracy_score(y, preds):.0%}")
print(f"Coefficient  : {model.coef_[0][0]:.4f}")
print(f"Intercept    : {model.intercept_[0]:.4f}")

# Decision boundary: where P = 0.5
boundary = -model.intercept_[0] / model.coef_[0][0]
print(f"Boundary     : {boundary:.2f} hours")

# Predict a new student (5.5 hours)
p = model.predict_proba([[5.5]])[0][1]
print(f"P(pass|5.5h) : {p:.2%}")`

const QUIZ_QUESTIONS = [
  { question: 'What does the Sigmoid function output?', options: ['Any real number','A value between 0 and 1','Only 0 or 1','A negative number'], correct: 1, explanation: 'Sigmoid squashes any real z to (0,1), making it ideal for probability outputs.' },
  { question: 'Logistic Regression is primarily used for:', options: ['Predicting continuous values','Classification problems','Clustering','Dimensionality reduction'], correct: 1, explanation: 'Logistic Regression predicts the probability of class membership — it\'s a classification algorithm.' },
  { question: 'The decision boundary is where:', options: ['Loss = 0','P(y=1) = 0.5','Weight = 0','Bias = 1'], correct: 1, explanation: 'At the boundary σ(z)=0.5, meaning z=0. Points with z>0 → class 1; z<0 → class 0.' },
  { question: 'What loss function does Logistic Regression minimise?', options: ['MSE','Binary Cross-Entropy (Log Loss)','Hinge Loss','MAE'], correct: 1, explanation: 'Binary Cross-Entropy penalises confident wrong predictions heavily: −[y·log(p)+(1−y)·log(1−p)].' },
  { question: 'If σ(z) = 0.85, the model:', options: ['Is uncertain','Predicts class 1 with 85% confidence','Predicts class 0','Has an error'], correct: 1, explanation: 'σ(z)=0.85 > 0.5 → predict class 1. The model is 85% confident the sample belongs to class 1.' },
]

const STUDENTS = [
  {hours:1,pass:false},{hours:2,pass:false},{hours:2.5,pass:false},
  {hours:3,pass:false},{hours:4,pass:false},{hours:5,pass:true},
  {hours:5.5,pass:true},{hours:6,pass:true},{hours:7,pass:true},{hours:8,pass:true},
]

function SigmoidCanvas({ theme }) {
  const [weight, setWeight] = useState(1)
  const [bias, setBias] = useState(0)
  const canvasRef = useRef(null)
  const W=380, H=240

  useEffect(()=>{
    const canvas=canvasRef.current; if(!canvas) return
    const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,W,H)
    const sigmoid=z=>1/(1+Math.exp(-z))
    const toY=p=>H-p*H*0.8-H*0.1
    ctx.strokeStyle=theme==='dark'?'rgba(255,255,255,0.04)':'rgba(0,0,0,0.05)'; ctx.lineWidth=1
    for(let x=0;x<W;x+=W/6){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}
    for(const p of[0,0.25,0.5,0.75,1]){ctx.beginPath();ctx.moveTo(0,toY(p));ctx.lineTo(W,toY(p));ctx.stroke()}
    ctx.beginPath();ctx.moveTo(0,toY(0.5));ctx.lineTo(W,toY(0.5))
    ctx.strokeStyle='rgba(239,68,68,0.35)';ctx.lineWidth=1.5;ctx.setLineDash([4,4]);ctx.stroke();ctx.setLineDash([])
    const grad=ctx.createLinearGradient(0,0,W,0); grad.addColorStop(0,'#6366F1'); grad.addColorStop(0.5,'#8B5CF6'); grad.addColorStop(1,'#06B6D4')
    ctx.beginPath()
    for(let px=0;px<=W;px++){const x=(px/W)*12-6;const p=sigmoid(weight*x+bias);if(px===0)ctx.moveTo(px,toY(p));else ctx.lineTo(px,toY(p))}
    ctx.strokeStyle=grad; ctx.lineWidth=3; ctx.stroke()
    ctx.fillStyle=theme==='dark'?'#9CA3AF':'#6B7280'; ctx.font='10px monospace'; ctx.textAlign='center'
    for(const p of[{v:1,l:'P=1.0'},{v:0.5,l:'P=0.5'},{v:0,l:'P=0.0'}])ctx.fillText(p.l,28,toY(p.v)+4)
  },[weight,bias,theme])

  return (
    <div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme==='dark'?'text-gray-300':'text-gray-600'}`}>Weight (w) = <span className="text-indigo-400 font-bold">{weight.toFixed(1)}</span></label>
          <input type="range" min="-3" max="3" step="0.1" value={weight} onChange={e=>setWeight(parseFloat(e.target.value))} className="w-full accent-indigo-500"/>
          <p className={`text-xs mt-1 ${theme==='dark'?'text-gray-600':'text-gray-400'}`}>{Math.abs(weight)>1.5?'Steep S-curve (sharp decision)':'Gentle S-curve (gradual transition)'}</p>
        </div>
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme==='dark'?'text-gray-300':'text-gray-600'}`}>Bias (b) = <span className="text-purple-400 font-bold">{bias.toFixed(1)}</span></label>
          <input type="range" min="-4" max="4" step="0.1" value={bias} onChange={e=>setBias(parseFloat(e.target.value))} className="w-full accent-purple-500"/>
          <p className={`text-xs mt-1 ${theme==='dark'?'text-gray-600':'text-gray-400'}`}>{bias>0?'Boundary shifted left (easier to predict 1)':bias<0?'Boundary shifted right (harder to predict 1)':'Boundary at x=0'}</p>
        </div>
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme==='dark'?'bg-slate-900 border-white/10':'bg-gray-50 border-gray-200'}`}
        style={{maxWidth:W,height:H}} />
      <p className={`text-xs mt-2 text-center font-mono ${theme==='dark'?'text-gray-500':'text-gray-400'}`}>
        σ(z) = 1/(1+e⁻ᶻ) where z = {weight.toFixed(1)}x + {bias.toFixed(1)} &nbsp;•&nbsp; Red = decision boundary (P=0.5)
      </p>
    </div>
  )
}

function ClassificationDemo({ theme }) {
  const canvasRef = useRef(null)
  const [boundary, setBoundary] = useState(4.5)
  const W=380, H=200
  const students=STUDENTS
  const correct=students.filter(s=>(s.hours>=boundary)===s.pass).length
  const accuracy=Math.round((correct/students.length)*100)

  useEffect(()=>{
    const canvas=canvasRef.current; if(!canvas) return
    const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,W,H)
    const toX=h=>(h/10)*W; const yFail=H*0.3,yPass=H*0.7
    const bx=toX(boundary)
    ctx.fillStyle='rgba(239,68,68,0.07)'; ctx.fillRect(0,0,bx,H)
    ctx.fillStyle='rgba(16,185,129,0.07)'; ctx.fillRect(bx,0,W-bx,H)
    ctx.beginPath();ctx.moveTo(bx,0);ctx.lineTo(bx,H);ctx.strokeStyle='#6366F1';ctx.lineWidth=2.5;ctx.setLineDash([6,3]);ctx.stroke();ctx.setLineDash([])
    ctx.fillStyle='rgba(239,68,68,0.6)'; ctx.font='12px sans-serif'; ctx.fillText('Fail zone',8,20)
    ctx.fillStyle='rgba(16,185,129,0.6)'; ctx.fillText('Pass zone',bx+8,20)
    students.forEach(s=>{
      const x=toX(s.hours),y=s.pass?yPass:yFail
      const pred=s.hours>=boundary
      ctx.beginPath();ctx.arc(x,y,9,0,Math.PI*2)
      ctx.fillStyle=s.pass?'#10B981':'#EF4444'; ctx.fill()
      if(pred!==s.pass){ctx.strokeStyle='#F59E0B';ctx.lineWidth=3;ctx.stroke()}
      else{ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke()}
    })
    ctx.beginPath();ctx.moveTo(0,H-20);ctx.lineTo(W,H-20)
    ctx.strokeStyle=theme==='dark'?'rgba(255,255,255,0.1)':'rgba(0,0,0,0.1)';ctx.lineWidth=1;ctx.stroke()
    ctx.fillStyle=theme==='dark'?'#6B7280':'#9CA3AF'; ctx.font='10px sans-serif'
    for(let h=1;h<=9;h+=2)ctx.fillText(h+'h',toX(h)-6,H-6)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[boundary,theme])

  return (
    <div>
      <div className="mb-4">
        <label className={`text-xs font-medium block mb-1.5 ${theme==='dark'?'text-gray-300':'text-gray-600'}`}>
          Decision Boundary = <span className="text-indigo-400 font-bold">{boundary.toFixed(1)} study hours</span>
        </label>
        <input type="range" min="1" max="9" step="0.5" value={boundary} onChange={e=>setBoundary(parseFloat(e.target.value))} className="w-full accent-indigo-500"/>
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme==='dark'?'bg-slate-900 border-white/10':'bg-gray-50 border-gray-200'}`}
        style={{maxWidth:W,height:H}} />
      <div className="flex flex-wrap gap-3 mt-3 text-xs">
        <div className={`px-3 py-1.5 rounded-lg ${theme==='dark'?'bg-slate-800':'bg-gray-100'}`}>
          <span className={theme==='dark'?'text-gray-500':'text-gray-400'}>Accuracy: </span>
          <span className={`font-bold ${accuracy>=80?'text-emerald-400':accuracy>=60?'text-yellow-400':'text-red-400'}`}>{accuracy}%</span>
        </div>
        <div className={`px-3 py-1.5 rounded-lg ${theme==='dark'?'bg-slate-800':'bg-gray-100'} ${theme==='dark'?'text-gray-400':'text-gray-500'}`}>
          🟢 Pass &nbsp; 🔴 Fail &nbsp; 🟡 Misclassified
        </div>
      </div>
    </div>
  )
}

function LogisticFlow({ theme }) {
  const steps = [
    {label:'Linear Output',icon:'📏',formula:'z = wx + b',color:'#6366F1',desc:'Compute weighted sum of inputs — same as linear regression.'},
    {label:'Sigmoid',icon:'σ',formula:'σ(z) = 1/(1+e⁻ᶻ)',color:'#8B5CF6',desc:'Squash z into a probability between 0 and 1.'},
    {label:'Probability',icon:'📊',formula:'P(y=1|x) ∈ [0,1]',color:'#06B6D4',desc:'Interpret output as the probability of belonging to class 1.'},
    {label:'Decision',icon:'✅',formula:'ŷ = 1 if P ≥ 0.5',color:'#10B981',desc:'Apply threshold to convert probability to a class label.'},
  ]
  return (
    <div className="flex flex-col items-center gap-2">
      {steps.map((step,i)=>(
        <div key={i} className="flex flex-col items-center w-full max-w-sm">
          <motion.div initial={{opacity:0,x:-20}} animate={{opacity:1,x:0}} transition={{delay:i*0.1}}
            className={`w-full px-5 py-3 rounded-xl border text-center ${theme==='dark'?'bg-slate-800/60 border-white/10':'bg-gray-50 border-gray-200'}`}>
            <div className="flex items-center justify-center gap-2 mb-1">
              <span className="text-xl">{step.icon}</span>
              <span className={`font-semibold text-sm ${theme==='dark'?'text-white':'text-gray-900'}`}>{step.label}</span>
            </div>
            <p className="font-mono text-xs mb-1" style={{color:step.color}}>{step.formula}</p>
            <p className={`text-xs ${theme==='dark'?'text-gray-500':'text-gray-400'}`}>{step.desc}</p>
          </motion.div>
          {i<steps.length-1&&<div className="text-indigo-500 text-xl leading-none py-1">↓</div>}
        </div>
      ))}
    </div>
  )
}

export default function LogisticRegression() {
  const { theme } = useApp()
  const S=`rounded-2xl border p-6 mb-6 ${theme==='dark'?'bg-slate-900/80 border-white/10':'bg-white border-gray-200'}`
  const LBL=`text-xs font-semibold uppercase tracking-wider mb-3 ${theme==='dark'?'text-purple-400':'text-purple-600'}`
  const H2=`text-xl font-bold mb-2 ${theme==='dark'?'text-white':'text-gray-900'}`
  const BODY=`text-sm leading-relaxed ${theme==='dark'?'text-gray-400':'text-gray-600'}`

  return (
    <motion.div initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} className="max-w-3xl">
      {/* Hero */}
      <div className="mb-8">
        <motion.div initial={{opacity:0,scale:0.9}} animate={{opacity:1,scale:1}}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-purple-500/20 border border-purple-500/30 mb-4">
          <span className="text-xs text-purple-400 font-medium">Machine Learning • Classification</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme==='dark'?'text-white':'text-gray-900'}`}>
          Logistic <span className="gradient-text">Regression</span>
        </h1>
        <p className={`text-lg mb-4 ${theme==='dark'?'text-gray-400':'text-gray-600'}`}>
          Despite its name, it's a <strong className={theme==='dark'?'text-white':'text-gray-900'}>classification</strong> algorithm. It uses the Sigmoid function to output a probability between 0 and 1.
        </p>
        <Callout type="analogy" title="Why 'Regression' in the name?">
          Logistic Regression first computes a linear combination of features (like linear regression), then maps the result through the sigmoid function to get a probability. The "regression" refers to that internal linear step.
        </Callout>
      </div>

      <TheoryBlock title="Core Theory" cards={[
        { icon: 'σ', title: 'Sigmoid Function', body: 'Maps any real z to (0,1). Large positive z → close to 1. Large negative z → close to 0. The "S" shape gives smooth, continuous probability outputs.', mono: 'σ(z) = 1 / (1 + e⁻ᶻ)' },
        { icon: '✂️', title: 'Decision Boundary', body: 'The hyperplane where P(y=1) = 0.5 (i.e. z=0). Points on one side are class 1, the other class 0. For linear logistic regression, it\'s always a straight line.', mono: 'boundary: wx + b = 0' },
        { icon: '📋', title: 'Log Loss (Cross-Entropy)', body: 'Penalises confident wrong predictions exponentially. If the model predicts P=0.99 for class 0 (true label), the loss is very high.', mono: 'L = −[y·log(p)+(1−y)·log(1−p)]' },
        { icon: '📏', title: 'Linear Component', body: 'First compute z = wx + b — a weighted sum identical to linear regression. Then sigmoid transforms this to a probability.', mono: 'z = w₁x₁ + w₂x₂ + b' },
        { icon: '📊', title: 'Multi-class Extension', body: 'For more than 2 classes, use Softmax instead of Sigmoid. Softmax outputs a probability distribution summing to 1 across all classes.', mono: 'softmax(zᵢ) = eᶻⁱ / Σeᶻʲ' },
        { icon: '🔍', title: 'Regularisation', body: 'To prevent overfitting, add L1 (Lasso) or L2 (Ridge) penalty to the loss. L2 shrinks weights; L1 can zero them out entirely (feature selection).', mono: 'Loss + λ·Σw²  (L2/Ridge)' },
      ]} />

      {/* Flow */}
      <div className={S}>
        <p className={LBL}>How Logistic Regression Works</p>
        <p className={`${BODY} mb-4`}>Four steps transform raw features into a class prediction:</p>
        <LogisticFlow theme={theme} />
        <Callout type="info" title="Key insight" className="mt-4">
          The sigmoid output is a probability, not a decision. The threshold (0.5 by default) converts probability to class. You can adjust the threshold depending on the problem — e.g. lower it to 0.3 to catch more true positives in medical diagnosis.
        </Callout>
      </div>

      {/* Sigmoid Playground */}
      <div className={S}>
        <p className={LBL}>Sigmoid Function Playground</p>
        <h2 className={H2}>See how weight and bias reshape the curve</h2>
        <p className={`${BODY} mb-3`}>
          <strong className={theme==='dark'?'text-white':'text-gray-900'}>Weight</strong> controls steepness — high weight = sharp, confident decision. <strong className={theme==='dark'?'text-white':'text-gray-900'}>Bias</strong> shifts the curve left or right, moving the boundary.
        </p>
        <Callout type="warning" title="Try this">
          Set weight = 0. The curve flattens to P=0.5 everywhere — the model can't distinguish classes. This is why features need to be informative.
        </Callout>
        <SigmoidCanvas theme={theme} />
        <DeepDive title="Why not just use a step function?">
          <p className={`text-sm ${BODY} mb-2`}>A step function outputs exactly 0 or 1 — it has no gradient, so gradient descent cannot update weights. Sigmoid has a smooth, continuous gradient everywhere, enabling learning.</p>
          <Callout type="formula" mono="σ'(z) = σ(z) · (1 − σ(z))">The sigmoid derivative is always positive and reaches max at z=0. This is used in backpropagation to compute weight updates.</Callout>
        </DeepDive>
      </div>

      {/* Classification Demo */}
      <div className={S}>
        <p className={LBL}>Pass / Fail Classifier</p>
        <h2 className={H2}>Slide the decision boundary — watch predictions change</h2>
        <p className={`${BODY} mb-3`}>Students above the boundary are predicted to pass. Yellow dots are misclassifications. Notice how accuracy peaks at a specific boundary value.</p>
        <Callout type="warning" title="Try this">
          Move the boundary to 1 hour (all predicted pass). Then to 9 hours (all predicted fail). Notice accuracy drops dramatically — demonstrating why the optimal boundary matters.
        </Callout>
        <ClassificationDemo theme={theme} />
        <DeepDive title="Precision, Recall, and F1 Score">
          <p className={`text-sm ${BODY} mb-2`}>Accuracy alone can be misleading on imbalanced datasets. Use these metrics instead:</p>
          <TheoryBlock title="" cards={[
            { icon: '🎯', title: 'Precision', body: 'Of all predicted positives, how many were actually positive? High precision = few false alarms.', mono: 'TP / (TP + FP)' },
            { icon: '🔍', title: 'Recall', body: 'Of all actual positives, how many did we catch? High recall = few misses.', mono: 'TP / (TP + FN)' },
            { icon: '⚖️', title: 'F1 Score', body: 'Harmonic mean of precision and recall. Best single metric when classes are imbalanced.', mono: '2 · P·R / (P + R)' },
          ]} />
        </DeepDive>
      </div>

      {/* Code */}
      <div className={S}>
        <p className={LBL}>Python Example</p>
        <h2 className={H2}>Full workflow with metrics</h2>
        <p className={`${BODY} mb-3`}>This example trains a logistic regression model and computes accuracy, the decision boundary, and probability for a new student.</p>
        <CodeBlock code={PYTHON_CODE} />
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme==='dark'?'bg-purple-500/10 border-purple-500/20':'bg-purple-50 border-purple-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme==='dark'?'text-white':'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme==='dark'?'text-gray-400':'text-gray-600'}`}>5 questions • +100 XP on completion</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="logistic-regression" />
      </div>
    </motion.div>
  )
}
